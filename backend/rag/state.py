"""Durable SQLite registry for documents and background jobs."""
import json
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

import upload
from config import DATABASE_PATH
from .embedding import MODEL_ID


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATABASE_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=30000")
    return db


def initialize() -> None:
    """Create tables, recover interrupted jobs, and register existing uploads."""
    with connection() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                kind TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL,
                embedding_model TEXT,
                page_count INTEGER,
                chunk_count INTEGER,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                source TEXT,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL,
                result_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS jobs_status_created
                ON jobs(status, created_at);
        """)
        # A crashed worker can safely retry indexing because point IDs are stable.
        db.execute("UPDATE jobs SET status='queued', stage='Recovered after restart', "
                   "progress=0, updated_at=? WHERE status='processing'", (now(),))
        db.execute("UPDATE documents SET status='pending', updated_at=? "
                   "WHERE status='processing'", (now(),))

        # Existing user PDFs/TXTs are retained; the new spatial collection needs reindexing.
        upload.UPLOAD_DIR.mkdir(exist_ok=True)
        paths = sorted((*upload.UPLOAD_DIR.glob("*.pdf"), *upload.UPLOAD_DIR.glob("*.txt")))
        for path in paths:
            if path.suffix == ".txt" and path.with_suffix(".pdf").exists():
                continue
            db.execute(
                "INSERT OR IGNORE INTO documents "
                "(id, filename, kind, size_bytes, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (path.name, upload.original_name(path), path.suffix[1:],
                 path.stat().st_size, now(), now()))
        # Schema changes require rebuilding child vectors and parent payloads.
        rows = db.execute("SELECT id, status, embedding_model FROM documents").fetchall()
        for row in rows:
            if row["status"] == "completed" and row["embedding_model"] == MODEL_ID:
                continue
            if row["status"] == "failed":
                continue
            if not (upload.UPLOAD_DIR / row["id"]).is_file():
                continue
            active = db.execute(
                "SELECT 1 FROM jobs WHERE kind='index' AND source=? "
                "AND status IN ('queued','processing') LIMIT 1", (row["id"],)).fetchone()
            if active:
                continue
            db.execute("UPDATE documents SET status='pending', error=NULL, updated_at=? "
                       "WHERE id=?", (now(), row["id"]))
            _insert_job(db, "index", row["id"], {})


def _insert_job(db, kind: str, source: str | None, payload: dict) -> str:
    job_id = uuid4().hex
    timestamp = now()
    db.execute(
        "INSERT INTO jobs (id, kind, source, payload_json, status, stage, progress, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, 'queued', 'Waiting for worker', 0, ?, ?)",
        (job_id, kind, source, json.dumps(payload, ensure_ascii=False), timestamp, timestamp))
    return job_id


def register_upload(saved: dict) -> str:
    """Register a saved file and its indexing job in one transaction."""
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        source = saved["stored_filename"]
        timestamp = now()
        db.execute(
            "INSERT INTO documents (id, filename, kind, size_bytes, status, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', ?, ?)",
            (source, saved["filename"], source.rsplit(".", 1)[-1],
             saved["size_bytes"], timestamp, timestamp))
        return _insert_job(db, "index", source, {})


def queue_index(source: str) -> str:
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        doc = db.execute("SELECT id FROM documents WHERE id=?", (source,)).fetchone()
        if doc is None:
            raise LookupError("Document not found")
        active = db.execute(
            "SELECT 1 FROM jobs WHERE kind='index' AND source=? "
            "AND status IN ('queued','processing') LIMIT 1", (source,)).fetchone()
        if active:
            raise RuntimeError("This document is already being indexed")
        questions = db.execute(
            "SELECT 1 FROM jobs WHERE kind IN ('chat','search') "
            "AND status IN ('queued','processing') "
            "AND (source=? OR source IS NULL) LIMIT 1", (source,)).fetchone()
        if questions:
            raise RuntimeError("Wait for active questions before reindexing this document")
        db.execute("UPDATE documents SET status='pending', error=NULL, updated_at=? "
                   "WHERE id=?", (now(), source))
        return _insert_job(db, "index", source, {})


def assert_ready(db: sqlite3.Connection, source: str | None) -> list[str]:
    """Validate readiness atomically with queuing a question."""
    if source:
        row = db.execute("SELECT status, embedding_model FROM documents WHERE id=?",
                         (source,)).fetchone()
        if row is None:
            raise LookupError("Document not found")
        if row["status"] != "completed" or row["embedding_model"] != MODEL_ID:
            raise RuntimeError("Document indexing has not completed")
        return [source]
    active = db.execute("SELECT 1 FROM jobs WHERE kind='index' "
                        "AND status IN ('queued','processing') LIMIT 1").fetchone()
    if active:
        raise RuntimeError("Document indexing is in progress")
    rows = db.execute("SELECT id FROM documents WHERE status='completed' "
                      "AND embedding_model=?", (MODEL_ID,)).fetchall()
    if not rows:
        raise RuntimeError("No indexed documents are ready")
    return [row["id"] for row in rows]


def queue_question(kind: str, payload: dict) -> str:
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        sources = assert_ready(db, payload.get("source"))
        return _insert_job(db, kind, payload.get("source"),
                           {**payload, "ready_sources": sources})


def ready_sources(source: str | None) -> list[str]:
    with connection() as db:
        return assert_ready(db, source)


def claim_next_job(kinds: tuple[str, ...] | None = None) -> dict | None:
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        clause = (" AND kind IN (" + ",".join("?" for _ in kinds) + ")") if kinds else ""
        row = db.execute("SELECT * FROM jobs WHERE status='queued'" + clause +
                         " ORDER BY created_at, rowid LIMIT 1", kinds or ()).fetchone()
        if row is None:
            return None
        db.execute("UPDATE jobs SET status='processing', stage='Starting', "
                   "progress=1, updated_at=? WHERE id=?", (now(), row["id"]))
        if row["kind"] == "index":
            db.execute("UPDATE documents SET status='processing', updated_at=? "
                       "WHERE id=?", (now(), row["source"]))
        job = dict(row)
        job["payload"] = json.loads(job.pop("payload_json"))
        return job


def progress(job_id: str, percent: int, stage: str) -> None:
    with connection() as db:
        db.execute("UPDATE jobs SET progress=?, stage=?, updated_at=? "
                   "WHERE id=? AND status='processing'",
                   (max(0, min(99, percent)), stage, now(), job_id))


def complete(job: dict, result: dict) -> None:
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("UPDATE jobs SET status='completed', stage='Completed', progress=100, "
                   "result_json=?, updated_at=? WHERE id=?",
                   (json.dumps(result, ensure_ascii=False), now(), job["id"]))
        if job["kind"] == "index":
            db.execute("UPDATE documents SET status='completed', embedding_model=?, "
                       "page_count=?, chunk_count=?, error=NULL, updated_at=? WHERE id=?",
                       (MODEL_ID, result.get("pages"), result.get("imported"),
                        now(), job["source"]))


def fail(job: dict, error: dict | str) -> None:
    detail = error if isinstance(error, dict) else {
        "code": "PROCESSING_ERROR", "message": str(error),
        "suggested_action": "Check server logs and retry."}
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("UPDATE jobs SET status='failed', stage='Failed', error=?, "
                   "updated_at=? WHERE id=?",
                   (json.dumps(detail, ensure_ascii=False), now(), job["id"]))
        if job["kind"] == "index":
            db.execute("UPDATE documents SET status='failed', error=?, updated_at=? "
                       "WHERE id=?", (detail["message"], now(), job["source"]))


def get_document(source: str) -> dict | None:
    with connection() as db:
        row = db.execute("SELECT * FROM documents WHERE id=?", (source,)).fetchone()
        return dict(row) if row else None


def list_documents() -> list[dict]:
    with connection() as db:
        rows = db.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]


def get_job(job_id: str) -> dict | None:
    with connection() as db:
        row = db.execute("SELECT id, kind, source, status, stage, progress, "
                         "result_json, error, created_at, updated_at "
                         "FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        job = dict(row)
        job["result"] = json.loads(job.pop("result_json")) if job["result_json"] else None
    if job["error"]:
        try:
            job["error_detail"] = json.loads(job["error"])
            job["error"] = job["error_detail"]["message"]
        except (ValueError, KeyError, TypeError):
            job["error_detail"] = None
    else:
        job["error_detail"] = None
    return job


def list_jobs(limit: int = 50) -> list[dict]:
    with connection() as db:
        rows = db.execute("SELECT id FROM jobs ORDER BY created_at DESC LIMIT ?",
                          (limit,)).fetchall()
    return [get_job(row["id"]) for row in rows]
