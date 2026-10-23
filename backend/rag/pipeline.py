"""Indexing, unique retrieval, and map-reduce answers."""
import re

import upload
from . import repository, state
from .extractor import extract_document
from .llm import answer
from .embedding import get_embedder
from .transform import chunk_segments
from .vision import get_vision_model

SUMMARY_PATTERN = re.compile(
    r"tóm tắt|tổng quan|khái quát|nội dung chính|điểm chính|"
    r"tài liệu.{0,40}(nói về|đề cập|trình bày)|"
    r"summari[sz]e|overview|main points|key takeaways|"
    r"document.{0,40}(about|discuss)",
    re.IGNORECASE)


def index_document(source: str, report=lambda percent, stage: None) -> dict:
    """Extract one document, embed unique chunks, and replace its Qdrant points."""
    document = state.get_document(source)
    if document is None:
        raise LookupError("Document not found")
    path = upload.UPLOAD_DIR / source
    if not path.is_file():
        raise FileNotFoundError(path)
    report(3, "Opening document")
    extracted = extract_document(path, report, get_vision_model())
    report(36, "Building unique spatial chunks")
    chunks = chunk_segments(extracted["segments"], document["filename"], source)
    if not chunks:
        raise ValueError("No searchable text was extracted")

    embedder = get_embedder()
    for offset in range(0, len(chunks), embedder.batch_size):
        batch = chunks[offset:offset + embedder.batch_size]
        vectors = embedder.embed_documents([chunk["original_text"] for chunk in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding provider returned an incomplete batch")
        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = vector
        report(38 + round(47 * (offset + len(batch)) / len(chunks)),
               f"Embedding {offset + len(batch)}/{len(chunks)} chunks")

    report(88, "Updating Qdrant")
    stored = repository.store_chunks(chunks, source, document["filename"])
    report(97, "Finalizing index")
    return {**stored, "source": source, "filename": document["filename"],
            "pages": extracted["pages"], "characters": extracted["characters"],
            "text_path": extracted["text_path"],
            "warnings": extracted.get("warnings", [])}


def _ready(payload: dict) -> list[str]:
    snapshot = payload.get("ready_sources")
    if snapshot:
        for source in snapshot:
            state.ready_sources(source)
        return snapshot
    return state.ready_sources(payload.get("source"))


def search_question(payload: dict, report=lambda percent, stage: None) -> dict:
    """Run exactly one vector query and return unique grounded passages."""
    sources = _ready(payload)
    report(25, "Embedding question")
    vector = get_embedder().embed_query(payload["question"])
    report(55, "Searching Qdrant")
    matches = repository.search(
        vector, sources, payload.get("top_k", 3), payload.get("threshold", 0.55))
    report(95, "Preparing citations")
    return {"sources": matches}


def _groups_for_summary(chunks: list[dict]) -> list[list[dict]]:
    """Keep each map group on one source and page for precise citation locations."""
    groups = []
    current = []
    previous = None
    for chunk in chunks:
        location = (chunk["source"], chunk["page_number"])
        if current and (location != previous or len(current) >= 3 or
                        sum(len(item["text"]) for item in current)
                        + len(chunk["text"]) > 6000):
            groups.append(current)
            current = []
        current.append(chunk)
        previous = location
    if current:
        groups.append(current)
    return groups


def _summary_note(group: list[dict], text: str) -> dict:
    """Carry every underlying location through reduction for the PDF viewer."""
    first = group[0]
    locations = []
    for item in group:
        for location in item.get("locations", []):
            if location not in locations:
                locations.append(location)
    return {
        "source": first["source"], "filename": first["filename"],
        "chunk_id": first["chunk_id"], "text": text, "score": 1.0,
        "page_number": first["page_number"], "bbox": first["bbox"],
        "page_width": first["page_width"], "page_height": first["page_height"],
        "paragraph_ids": first.get("paragraph_ids", []),
        "locations": locations, "type": "summary_note",
    }


def _strip_local_citations(text: str) -> str:
    return re.sub(r"\[\d+\]", "", text).strip()


def summarize_question(payload: dict, report=lambda percent, stage: None) -> dict:
    """Map every chunk, then reduce notes until they fit one final answer prompt."""
    sources = _ready(payload)
    chunks = repository.all_chunks(sources)
    if not chunks:
        return {"answer": "No indexed content is available to summarize.",
                "sources": [], "mode": "summary"}
    # Small documents fit in one prompt, so preserve their original passages.
    if len(chunks) <= 6:
        report(75, "Reading the complete document")
        result = answer(payload["question"], chunks, payload.get("history", []))
        result["mode"] = "summary"
        report(96, "Preparing summary citations")
        return result
    groups = _groups_for_summary(chunks)
    notes = []
    for index, group in enumerate(groups, 1):
        mapped = answer(
            "Liệt kê ngắn các ý chính thực sự có trong các nguồn này. "
            "Giữ nguyên tên riêng, số liệu và sự kiện; không suy diễn hoặc "
            "nhận xét về phần tài liệu không được cung cấp.",
            group)
        notes.append(_summary_note(group, _strip_local_citations(mapped["answer"])))
        report(5 + round(65 * index / len(groups)),
               f"Summarizing section {index}/{len(groups)}")

    # Hierarchical reduction preserves coverage when the document has many pages.
    level = 0
    while len(notes) > 6:
        next_notes = []
        for offset in range(0, len(notes), 6):
            group = notes[offset:offset + 6]
            reduced = answer(
                "Gộp các bản tóm tắt này thành một bản ngắn, giữ đủ chủ đề và số liệu.",
                group)
            next_notes.append(_summary_note(
                group, _strip_local_citations(reduced["answer"])))
        notes = next_notes
        level += 1
        report(min(90, 72 + level * 6), f"Combining summary level {level}")

    final = answer(payload["question"], notes, payload.get("history", []))
    final["mode"] = "summary"
    report(96, "Preparing summary citations")
    return final


def answer_question(payload: dict, report=lambda percent, stage: None) -> dict:
    """Use map-reduce for broad questions; otherwise retrieve once and answer."""
    if SUMMARY_PATTERN.search(payload["question"]):
        return summarize_question(payload, report)
    retrieved = search_question(payload, report)
    report(70, "Generating answer")
    result = answer(payload["question"], retrieved["sources"],
                    payload.get("history", []))
    result["mode"] = "retrieval"
    report(96, "Preparing citations")
    return result
