"""Qdrant storage with stable IDs, source filtering, and result deduplication."""
from contextlib import closing
import math
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from config import COLLECTION, QDRANT_URL
from .embedding import get_embedder
from .transform import normalized_key


def connect() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=60)


def ensure_collection(client: QdrantClient) -> None:
    dimensions = get_embedder().dimensions
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=dimensions, distance=models.Distance.COSINE))
        return
    vectors = client.get_collection(COLLECTION).config.params.vectors
    if (not isinstance(vectors, models.VectorParams)
            or vectors.size != dimensions
            or vectors.distance != models.Distance.COSINE):
        raise ValueError("Qdrant collection has incompatible vector dimensions or distance")


def source_filter(sources: list[str]) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(
        key="source", match=models.MatchAny(any=sources))])


def _point_id(source: str, text_hash: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{COLLECTION}/{source}/{text_hash}"))


def store_chunks(chunks: list[dict], source: str, filename: str) -> dict:
    """Upsert every unique chunk, then remove stale points from this source."""
    if not chunks:
        raise ValueError("The document produced no searchable chunks")
    points = []
    seen_hashes = set()
    for chunk in chunks:
        text_hash = chunk["text_hash"]
        if text_hash in seen_hashes:
            raise ValueError("Duplicate chunk reached Qdrant storage")
        seen_hashes.add(text_hash)
        vector = chunk["embedding"]
        if (len(vector) != get_embedder().dimensions
                or not all(isinstance(value, (int, float)) and math.isfinite(value)
                           for value in vector)
                or not any(vector)):
            raise ValueError("Embedding dimension mismatch")
        payload = {key: value for key, value in chunk.items() if key != "embedding"}
        payload.update(source=source, filename=filename)
        payload["locations"] = [
            {**location, "source": source, "filename": filename}
            for location in chunk.get("locations", [])]
        points.append(models.PointStruct(
            id=_point_id(source, text_hash), vector=vector, payload=payload))

    new_ids = {point.id for point in points}
    with closing(connect()) as client:
        ensure_collection(client)
        for offset in range(0, len(points), 64):
            client.upsert(COLLECTION, points=points[offset:offset + 64], wait=True)

        # Scroll old point IDs instead of constructing an unbounded HasId filter.
        stale_ids = []
        cursor = None
        while True:
            batch, cursor = client.scroll(
                COLLECTION, scroll_filter=source_filter([source]),
                with_payload=False, with_vectors=False, limit=256, offset=cursor)
            stale_ids.extend(point.id for point in batch if point.id not in new_ids)
            if cursor is None:
                break
        for offset in range(0, len(stale_ids), 256):
            client.delete(COLLECTION, points_selector=models.PointIdsList(
                points=stale_ids[offset:offset + 256]), wait=True)
    return {"imported": len(points), "removed_stale": len(stale_ids)}


def _result(payload: dict, score: float) -> dict:
    """Expand a matched child to its intact parent for generation and citation."""
    image_path = payload.get("image_path")
    figure_url = (f"/documents/{payload.get('source')}/figures/{Path(image_path).name}"
                  if image_path else None)
    return {
        "source": payload.get("source"), "filename": payload.get("filename"),
        "chunk_id": payload.get("chunk_id"), "parent_id": payload.get("parent_id"),
        "text": payload.get("parent_text", payload.get("original_text", "")),
        "search_text": payload.get("original_text", ""), "score": score,
        "type": payload.get("type", "text"),
        "document_title": payload.get("document_title"),
        "section": payload.get("section"), "figure_url": figure_url,
        "page_number": payload.get("page_number"), "bbox": payload.get("bbox"),
        "page_width": payload.get("page_width"),
        "page_height": payload.get("page_height"),
        "paragraph_ids": payload.get("paragraph_ids", []),
        "locations": payload.get("locations", []),
    }


def search(vector: list[float], sources: list[str], top_k: int = 3,
           threshold: float = 0.55) -> list[dict]:
    """Fetch extra candidates and return at most one result per normalized text."""
    if not sources:
        return []
    with closing(connect()) as client:
        if not client.collection_exists(COLLECTION):
            return []
        found = client.query_points(
            collection_name=COLLECTION, query=vector,
            query_filter=source_filter(sources),
            limit=min(max(top_k * 8, 24), 100),
            score_threshold=threshold, with_payload=True).points
    results = []
    seen = set()
    for point in found:
        payload = point.payload or {}
        text = payload.get("parent_text", payload.get("original_text", ""))
        key = payload.get("parent_id") or normalized_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(_result(payload, float(point.score)))
        if len(results) == top_k:
            break
    return results


def all_chunks(sources: list[str]) -> list[dict]:
    """Read content in document order for the map-reduce summary path."""
    if not sources:
        return []
    rows = []
    seen = set()
    with closing(connect()) as client:
        if not client.collection_exists(COLLECTION):
            return []
        cursor = None
        while True:
            batch, cursor = client.scroll(
                COLLECTION, scroll_filter=source_filter(sources),
                with_payload=True, with_vectors=False, limit=256, offset=cursor)
            for point in batch:
                payload = point.payload or {}
                parent_id = payload.get("parent_id") or payload.get("text_hash")
                if parent_id not in seen and payload.get("parent_text"):
                    seen.add(parent_id)
                    rows.append(_result(payload, 1.0))
            if cursor is None:
                break
    order = {source: index for index, source in enumerate(sources)}
    rows.sort(key=lambda row: (order.get(row["source"], 999999),
                               row["page_number"] or 0, row["chunk_id"] or 0))
    return rows
