"""Layout-aware parent blocks and semantic child search chunks."""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from threading import RLock
from uuid import NAMESPACE_URL, uuid5

import aiofiles

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSIONS = 384
MAX_MODEL_TOKENS = 128
CHILD_WORDS = 85
PARENT_WORDS = 420
_model = None
_tokenizer = None
_model_lock = RLock()


async def load(filepath: str) -> str:
    async with aiofiles.open(filepath, "r", encoding="utf-8-sig") as file:
        return await file.read()


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def normalized_key(text: str) -> str:
    return clean(unicodedata.normalize("NFKC", text)).casefold()


def get_tokenizer():
    global _tokenizer
    with _model_lock:
        if _tokenizer is None:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        return _tokenizer


def _sentences(text: str) -> list[str]:
    """Prefer sentence boundaries; split oversized prose at word boundaries."""
    units = re.split(r"(?<=[.!?。])\s+|\n\s*\n", text)
    result = []
    for unit in units:
        words = unit.split()
        for offset in range(0, len(words), CHILD_WORDS):
            part = " ".join(words[offset:offset + CHILD_WORDS])
            if part:
                result.append(part)
    return result


def _child_texts(parent: dict) -> list[str]:
    """Keep tables, code, and chart summaries indivisible in the child index."""
    kind = parent["kind"]
    if kind == "table":
        rows = [line for line in parent["text"].splitlines() if line.startswith("|")]
        header = rows[0] if rows else ""
        children = [f"Table in {parent['section']}. Columns: {header}. "
                    f"Rows: {max(0, len(rows) - 2)}."]
        children.extend(f"Table in {parent['section']}. {header} {row}"
                        for row in rows[2:])
        return children
    if kind == "code":
        names = re.findall(r"\b(?:def|class|function)\s+([\w_]+)", parent["text"])
        return [f"Code in {parent['section']}. Symbols: {', '.join(names) or 'none'}. "
                f"{clean(parent['text'])}" if len(parent["text"].split()) <= CHILD_WORDS
                else f"Code in {parent['section']}. Symbols: {', '.join(names) or 'none'}."]
    if kind == "figure":
        summary = parent.get("visual_summary") or parent["text"]
        table = parent.get("markdown_table") or ""
        children = [f"Figure in {parent['section']}. {summary}".strip()]
        rows = [line for line in table.splitlines() if line.startswith("|")]
        if rows:
            children.extend(f"Chart in {parent['section']}. {rows[0]} {row}"
                            for row in rows[2:])
        return children
    pieces, current = [], []
    for sentence in _sentences(parent["text"]):
        if current and len((" ".join(current) + " " + sentence).split()) > CHILD_WORDS:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _parent(blocks: list[dict], title: str, section: str, ordinal: int,
            source: str) -> dict:
    first = blocks[0]
    kind = first["kind"]
    text = "\n\n".join(block["text"] for block in blocks)
    boxes = [block["bbox"] for block in blocks if block.get("bbox")]
    box = ([min(item[0] for item in boxes), min(item[1] for item in boxes),
            max(item[2] for item in boxes), max(item[3] for item in boxes)]
           if boxes else None)
    parent_id = str(uuid5(NAMESPACE_URL, f"{source}/{section}/{ordinal}/{text}"))
    return {"parent_id": parent_id, "kind": kind, "text": text,
            "document_title": title, "section": section,
            "page_number": first["page_number"], "bbox": box,
            "page_width": first["page_width"], "page_height": first["page_height"],
            "paragraph_ids": [block["paragraph_id"] for block in blocks],
            "image_path": first.get("image_path"),
            "visual_summary": first.get("visual_summary"),
            "markdown_table": first.get("markdown_table")}


def build_chunks(segments: list[dict], document_title: str,
                 source: str | None = None) -> list[dict]:
    """Group nearby prose into parents, then index smaller semantic children."""
    title = document_title
    for segment in segments:
        if segment["kind"] == "heading" and segment["text"]:
            title = segment["text"]
            break
    parents = []
    source_id = source or document_title
    pending: list[dict] = []
    section = title
    pending_section = section

    def flush() -> None:
        nonlocal pending
        if pending:
            parents.append(_parent(pending, title, pending_section,
                                   len(parents), source_id))
            pending = []

    for segment in segments:
        kind = segment["kind"]
        if kind == "heading":
            flush()
            section = segment["text"]
            continue
        if pending and (segment["page_number"] != pending[0]["page_number"]
                        or section != pending_section
                        or sum(len(item["text"].split()) for item in pending)
                        + len(segment["text"].split()) > PARENT_WORDS):
            flush()
        if kind in {"table", "code", "figure"}:
            flush()
            parents.append(_parent([segment], title, section,
                                   len(parents), source_id))
            continue
        if not pending:
            pending_section = section
        pending.append(segment)
    flush()

    unique: dict[str, dict] = {}
    for parent in parents:
        location = {key: parent[key] for key in (
            "page_number", "bbox", "page_width", "page_height", "paragraph_ids")}
        for child_text in _child_texts(parent):
            key = hashlib.sha256(normalized_key(child_text).encode("utf-8")).hexdigest()
            if key in unique:
                if location not in unique[key]["locations"]:
                    unique[key]["locations"].append(location)
                continue
            unique[key] = {
                "original_text": child_text, "text_hash": key,
                "type": parent["kind"], "parent_id": parent["parent_id"],
                "parent_text": parent["text"], "document_title": title,
                "section": parent["section"], "image_path": parent["image_path"],
                "page_number": parent["page_number"], "bbox": parent["bbox"],
                "page_width": parent["page_width"],
                "page_height": parent["page_height"],
                "paragraph_ids": parent["paragraph_ids"],
                "locations": [location],
            }
    chunks = list(unique.values())
    for number, chunk in enumerate(chunks):
        chunk["chunk_id"] = number
    return chunks


def chunk_segments(segments: list[dict], document_title: str = "Document",
                   source: str | None = None) -> list[dict]:
    """Compatibility entry point for the parent-child chunking strategy."""
    return build_chunks(segments, document_title, source)


def embed_many(texts: list[str]) -> list[list[float]]:
    """Run the local multilingual MiniLM adapter in a bounded batch."""
    global _model
    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("Cannot embed empty text")
    with _model_lock:
        import torch
        from transformers import AutoModel
        if _model is None:
            device = os.getenv("EMBEDDING_DEVICE", "cpu")
            if device == "mps" and not torch.backends.mps.is_available():
                raise ValueError("MPS is unavailable; use EMBEDDING_DEVICE=cpu")
            _model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
        encoded = get_tokenizer()(texts, return_tensors="pt", padding=True,
                                  truncation=True, max_length=MAX_MODEL_TOKENS)
        encoded = {key: value.to(_model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = _model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().tolist()
        if any(len(vector) != EMBEDDING_DIMENSIONS for vector in vectors):
            raise ValueError("Local embedding dimension mismatch")
        return vectors


def embed(text: str) -> list[float]:
    return embed_many([text])[0]
