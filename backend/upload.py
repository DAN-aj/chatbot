"""Kiểm tra và lưu PDF/TXT bất đồng bộ vào uploads/."""
import codecs
import json
from pathlib import Path
from uuid import uuid4

import aiofiles
from aiofiles.os import makedirs, remove
from fastapi import HTTPException, UploadFile

from config import MAX_UPLOAD_BYTES

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
CHUNK_SIZE = 1024 * 1024
MAX_FILE_SIZE = MAX_UPLOAD_BYTES
ALLOWED_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
}


def metadata_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def read_metadata(path: Path) -> dict:
    meta = metadata_path(path)
    if meta.is_file():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            pass
    return {}


def original_name(path: Path) -> str:
    return read_metadata(path).get("filename", path.name)


async def save_file(file: UploadFile) -> dict:
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_TYPES:
        raise HTTPException(400, "Only PDF or TXT files are supported")
    if file.content_type not in ALLOWED_TYPES[suffix]:
        raise HTTPException(400, "Content-Type does not match the file format")

    await makedirs(UPLOAD_DIR, exist_ok=True)
    path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    meta = metadata_path(path)
    size = 0
    first = True
    decoder = codecs.getincrementaldecoder("utf-8-sig")() if suffix == ".txt" else None
    completed = False
    try:
        async with aiofiles.open(path, "xb") as output:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(413, "File exceeds the configured size limit")
                if first and suffix == ".pdf" and not chunk.startswith(b"%PDF-"):
                    raise HTTPException(400, "Invalid PDF header")
                if decoder is not None:
                    if b"\x00" in chunk:
                        raise HTTPException(400, "TXT contains NUL characters")
                    try:
                        decoder.decode(chunk)
                    except UnicodeError as exc:
                        raise HTTPException(400, "TXT must be UTF-8") from exc
                await output.write(chunk)
                first = False
            if size == 0:
                raise HTTPException(400, "File is empty")
            if decoder is not None:
                try:
                    decoder.decode(b"", final=True)
                except UnicodeError as exc:
                    raise HTTPException(400, "TXT must be UTF-8") from exc
        async with aiofiles.open(meta, "w", encoding="utf-8") as output:
            await output.write(json.dumps({"filename": filename}, ensure_ascii=False))
        completed = True
    finally:
        if not completed:
            for candidate in (path, meta):
                try:
                    await remove(candidate)
                except FileNotFoundError:
                    pass
    return {"filename": filename, "stored_filename": path.name,
            "size_bytes": size, "path": f"uploads/{path.name}"}
