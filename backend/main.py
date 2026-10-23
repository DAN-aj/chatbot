"""FastAPI routes and durable single-worker job processor."""
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event, Thread
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

import upload
from config import DATA_DIR
from rag import pipeline, state
from rag.errors import public_error

logger = logging.getLogger(__name__)
worker_threads: list[Thread] = []


def _worker_loop(kinds: tuple[str, ...], stop_worker: Event) -> None:
    """Claim jobs atomically so every request has persisted progress and errors."""
    while not stop_worker.is_set():
        job = None
        try:
            job = state.claim_next_job(kinds)
            if job is None:
                stop_worker.wait(0.5)
                continue
            report = lambda percent, stage: state.progress(job["id"], percent, stage)
            if job["kind"] == "index":
                result = pipeline.index_document(job["source"], report)
            elif job["kind"] == "chat":
                result = pipeline.answer_question(job["payload"], report)
            elif job["kind"] == "search":
                result = pipeline.search_question(job["payload"], report)
            else:
                raise ValueError(f"Unknown job kind: {job['kind']}")
            state.complete(job, result)
        except Exception as exc:
            logger.exception("Background job failed")
            if job is not None:
                try:
                    state.fail(job, public_error(exc))
                except Exception:
                    logger.exception("Could not record failed job; retry after restart")
                    stop_worker.wait(1)
                job = None
            else:
                stop_worker.wait(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Use separate durable index and query workers to isolate API latency."""
    state.initialize()
    stop_worker = Event()
    worker_threads[:] = [
        Thread(target=_worker_loop, args=(("index",), stop_worker),
               name="rag-index-worker", daemon=True),
        Thread(target=_worker_loop, args=(("chat", "search"), stop_worker),
               name="rag-query-worker", daemon=True),
    ]
    for thread in worker_threads:
        thread.start()
    try:
        yield
    finally:
        stop_worker.set()
        for thread in worker_threads:
            thread.join(timeout=3)


app = FastAPI(title="Talk to Documents", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=public_error(exc))


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={
        "code": "VALIDATION_ERROR", "message": "Request fields are invalid.",
        "suggested_action": "Check required fields and allowed values."})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    logger.exception("Unhandled API error")
    return JSONResponse(status_code=500, content=public_error(exc))


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    source: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)
    threshold: float = Field(default=0.55, ge=0, le=1)
    history: list[ConversationTurn] = Field(default_factory=list, max_length=6)

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty")
        return value


def _queue_question(kind: str, body: Question) -> dict:
    try:
        job_id = state.queue_question(kind, body.model_dump())
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job_id": job_id, "status": "queued"}


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/documents")
def documents():
    return state.list_documents()


@app.get("/documents/{source}")
def document(source: str):
    record = state.get_document(source)
    if record is None:
        raise HTTPException(404, "Document not found")
    return record


@app.get("/documents/{source}/file")
def document_file(source: str):
    """Serve the original file; PDF page coordinates refer to these bytes."""
    record = state.get_document(source)
    if record is None:
        raise HTTPException(404, "Document not found")
    path = upload.UPLOAD_DIR / source
    if not path.is_file():
        raise HTTPException(404, "Original file is missing")
    media_type = "application/pdf" if record["kind"] == "pdf" else "text/plain; charset=utf-8"
    return FileResponse(path, media_type=media_type, filename=record["filename"],
                        content_disposition_type="inline")


@app.get("/documents/{source}/figures/{figure_name}")
def document_figure(source: str, figure_name: str):
    """Serve only a generated figure belonging to a registered document."""
    if state.get_document(source) is None:
        raise HTTPException(404, "Document not found")
    if not re.fullmatch(r"p\d{4}-figure-\d{3}\.png", figure_name):
        raise HTTPException(404, "Figure not found")
    path = DATA_DIR / "figures" / source / figure_name
    if not path.is_file():
        raise HTTPException(404, "Figure not found")
    return FileResponse(path, media_type="image/png")


@app.post("/upload", status_code=202)
async def file_upload(file: UploadFile):
    """Persist upload first, then register its index job in SQLite."""
    try:
        saved = await upload.save_file(file)
    finally:
        await file.close()
    try:
        job_id = await run_in_threadpool(state.register_upload, saved)
    except Exception:
        path = upload.UPLOAD_DIR / saved["stored_filename"]
        path.unlink(missing_ok=True)
        upload.metadata_path(path).unlink(missing_ok=True)
        raise
    return {**saved, "document_id": saved["stored_filename"],
            "job_id": job_id, "status": "queued"}


@app.post("/documents/{source}/index", status_code=202)
def index(source: str):
    try:
        job_id = state.queue_index(source)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job_id": job_id, "document_id": source, "status": "queued"}


@app.post("/chat", status_code=202)
def chat(body: Question):
    return _queue_question("chat", body)


@app.post("/search", status_code=202)
def search(body: Question):
    return _queue_question("search", body)


@app.get("/jobs")
def jobs(limit: int = 50):
    return state.list_jobs(max(1, min(limit, 100)))


@app.get("/jobs/{job_id}")
def job(job_id: str):
    record = state.get_job(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    return record
