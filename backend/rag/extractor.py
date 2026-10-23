"""Layout-aware PDF/TXT parser producing atomic text, table, and visual blocks."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import pdfplumber
import pymupdf

from config import DATA_DIR
from .vision import VisionModel

logger = logging.getLogger(__name__)


def _warn(warnings: list[dict], source: str, page: int, code: str, exc: object) -> None:
    message = str(exc)
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
        secret = os.getenv(name)
        if secret:
            message = message.replace(secret, "[redacted]")
    warning = {"source": source, "page_number": page, "code": code,
               "message": message[:300]}
    warnings.append(warning)
    logger.warning("ingestion_warning %s", json.dumps(warning, ensure_ascii=False))


def _base(kind: str, text: str, page: int | None, box: list[float] | None,
          width: float | None, height: float | None, block_id: str) -> dict:
    return {"kind": kind, "text": text.strip(), "page_number": page,
            "bbox": box, "page_width": width, "page_height": height,
            "paragraph_id": block_id, "image_path": None}


def _box(words: list[dict]) -> list[float]:
    return [round(min(word["x0"] for word in words), 2),
            round(min(word["top"] for word in words), 2),
            round(max(word["x1"] for word in words), 2),
            round(max(word["bottom"] for word in words), 2)]


def _intersects(a: tuple | list, b: tuple | list) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _markdown_table(rows: list[list[str | None]]) -> str:
    """Preserve ragged table rows in one valid Markdown table."""
    width = max((len(row) for row in rows), default=0)
    if width < 2 or len(rows) < 2:
        return ""
    clean = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", str(value or "")).strip()
                 .replace("|", "\\|") for value in row]
        clean.append(cells + [""] * (width - len(cells)))
    if not any(cell for row in clean for cell in row):
        return ""
    return "\n".join([
        "| " + " | ".join(clean[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
        *("| " + " | ".join(row) + " |" for row in clean[1:]),
    ])


def _table_blocks(page, number: int) -> tuple[list[dict], list[tuple]]:
    blocks, boxes = [], []
    for index, table in enumerate(page.find_tables(), 1):
        markdown = _markdown_table(table.extract())
        if not markdown:
            continue
        box = tuple(float(value) for value in table.bbox)
        boxes.append(box)
        blocks.append(_base("table", markdown, number, list(box),
                            float(page.width), float(page.height),
                            f"p{number:04d}-table-{index:03d}"))
    return blocks, boxes


def _paragraph_blocks(page, number: int, excluded: list[tuple]) -> list[dict]:
    """Group measured words into paragraphs, headings, and fenced code blocks."""
    words = page.extract_words(x_tolerance=2, y_tolerance=3,
                               extra_attrs=["fontname", "size"])
    unique = {}
    for word in words:
        if not word["text"].strip():
            continue
        box = (word["x0"], word["top"], word["x1"], word["bottom"])
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        if any(region[0] <= center[0] <= region[2] and
               region[1] <= center[1] <= region[3] for region in excluded):
            continue
        key = (word["text"], *(round(value, 2) for value in box))
        unique.setdefault(key, word)
    ordered = sorted(unique.values(), key=lambda item: (round(item["top"] / 3), item["x0"]))
    if not ordered:
        return []
    sizes = sorted(float(item.get("size") or 10) for item in ordered)
    median_size = sizes[len(sizes) // 2]
    lines: list[list[dict]] = []
    for word in ordered:
        if lines and abs(word["top"] - lines[-1][0]["top"]) <= 3:
            lines[-1].append(word)
        else:
            lines.append([word])
    groups: list[tuple[str, list[dict]]] = []
    current: list[dict] = []
    current_kind = "text"
    previous_bottom = None
    for line in lines:
        text = " ".join(word["text"] for word in line)
        size = max(float(word.get("size") or 10) for word in line)
        fonts = " ".join(str(word.get("fontname") or "") for word in line).lower()
        kind = ("code" if "courier" in fonts or "mono" in fonts else
                "heading" if len(text.split()) <= 14 and size >= median_size * 1.18
                else "text")
        top = min(word["top"] for word in line)
        bottom = max(word["bottom"] for word in line)
        gap = top - previous_bottom if previous_bottom is not None else 0
        if current and (kind != current_kind or kind == "heading" or
                        gap > max(5, (bottom - top) * 0.8)):
            groups.append((current_kind, current))
            current = []
        current_kind = kind
        current.extend(line)
        previous_bottom = bottom
    if current:
        groups.append((current_kind, current))
    blocks = []
    for index, (kind, group) in enumerate(groups, 1):
        text = " ".join(word["text"] for word in group)
        if kind == "code":
            text = "```\n" + text + "\n```"
        blocks.append(_base(kind, text, number, _box(group),
                            float(page.width), float(page.height),
                            f"p{number:04d}-{index:04d}"))
    return blocks


def _visual_regions(page, tables: list[tuple], has_text: bool) -> list[tuple]:
    """Find raster figures, scanned pages, and sizeable vector drawing groups."""
    if not has_text:
        return [tuple(page.rect)]
    regions = []
    for image in page.get_image_info():
        box = tuple(float(value) for value in image["bbox"])
        if box[2] - box[0] >= 80 and box[3] - box[1] >= 65:
            if not any(_intersects(box, table) for table in tables):
                regions.append(box)
    clusters: list[list] = []
    for drawing in page.get_drawings():
        box = tuple(float(value) for value in drawing["rect"])
        if any(_intersects(box, table) for table in tables):
            continue
        for cluster in clusters:
            bounds = cluster[0]
            padded = (bounds[0] - 12, bounds[1] - 12,
                      bounds[2] + 12, bounds[3] + 12)
            if _intersects(box, padded):
                bounds[:] = [min(bounds[0], box[0]), min(bounds[1], box[1]),
                             max(bounds[2], box[2]), max(bounds[3], box[3])]
                cluster[1] += 1
                break
        else:
            clusters.append([[box[0], box[1], box[2], box[3]], 1])
    for box, count in clusters:
        if count >= 5 and box[2] - box[0] >= 120 and box[3] - box[1] >= 80:
            if not any(_intersects(box, existing) for existing in regions):
                regions.append(tuple(box))
    return regions[:12]


def _visual_blocks(page, number: int, source: str, regions: list[tuple],
                   vision: VisionModel | None, warnings: list[dict]) -> list[dict]:
    blocks = []
    if regions and vision is None:
        _warn(warnings, source, number, "vision_unconfigured",
              "Set GEMINI_API_KEY to analyze figures or scanned pages")
        return blocks
    for index, region in enumerate(regions, 1):
        try:
            box = pymupdf.Rect(region) & page.rect
            if box.is_empty:
                continue
            image = page.get_pixmap(clip=box, dpi=144, alpha=False).tobytes("png")
            observation = vision.describe(image)
            text = "\n\n".join(part for part in (
                observation["summary"], observation["markdown_table"]) if part)
            figure_dir = DATA_DIR / "figures" / source
            figure_dir.mkdir(parents=True, exist_ok=True)
            figure_path = figure_dir / f"p{number:04d}-figure-{index:03d}.png"
            figure_path.write_bytes(image)
            block = _base("figure", text, number, list(box),
                          float(page.rect.width), float(page.rect.height),
                          f"p{number:04d}-figure-{index:03d}")
            block["image_path"] = str(figure_path)
            block["visual_summary"] = observation["summary"]
            block["markdown_table"] = observation["markdown_table"]
            blocks.append(block)
        except Exception as exc:
            _warn(warnings, source, number, "figure_failed", exc)
    return blocks


def _txt_blocks(text: str) -> list[dict]:
    blocks, code, paragraph, table = [], [], [], []
    def emit(kind: str, lines: list[str]) -> None:
        if lines:
            blocks.append(_base(kind, "\n".join(lines), None, None, None, None,
                                f"txt-{len(blocks) + 1:04d}"))
            lines.clear()
    in_code = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            emit("text", paragraph)
            emit("table", table)
            code.append(line)
            in_code = not in_code
            if not in_code:
                emit("code", code)
        elif in_code:
            code.append(line)
        elif stripped.startswith("#"):
            emit("text", paragraph)
            emit("table", table)
            emit("heading", [stripped.lstrip("# ")])
        elif stripped.startswith("|") and stripped.endswith("|"):
            emit("text", paragraph)
            table.append(line)
        elif not stripped:
            emit("text", paragraph)
            emit("table", table)
        else:
            emit("table", table)
            paragraph.append(line)
    emit("code" if in_code else "text", code)
    emit("text", paragraph)
    emit("table", table)
    return [block for block in blocks if block["text"].strip()]


def extract_document(path: Path, report=lambda percent, stage: None,
                     vision: VisionModel | None = None) -> dict:
    """Continue after damaged pages and return valid blocks plus warnings."""
    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8-sig")
        blocks = _txt_blocks(text)
        if not blocks:
            raise ValueError("The TXT file contains no readable text")
        return {"segments": blocks, "pages": None, "characters": len(text),
                "text_path": path.name, "warnings": []}
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF and UTF-8 TXT files are supported")
    warnings: list[dict] = []
    blocks: list[dict] = []
    page_texts: list[str] = []
    with pymupdf.open(path) as image_pdf:
        pages = len(image_pdf)
        if not pages:
            raise ValueError("The PDF contains no pages")
        try:
            text_pdf = pdfplumber.open(path)
        except Exception as exc:
            text_pdf = None
            _warn(warnings, path.name, 0, "layout_parser_unavailable", exc)
        try:
            for offset in range(pages):
                number = offset + 1
                page_blocks, table_boxes = [], []
                try:
                    if text_pdf is None:
                        raise ValueError("Layout parser unavailable")
                    text_page = text_pdf.pages[offset]
                    tables, table_boxes = _table_blocks(text_page, number)
                    page_blocks.extend(tables)
                    page_blocks.extend(_paragraph_blocks(text_page, number, table_boxes))
                except Exception as exc:
                    _warn(warnings, path.name, number, "page_text_failed", exc)
                    try:
                        for index, item in enumerate(image_pdf[offset].get_text("blocks"), 1):
                            if item[4].strip():
                                page_blocks.append(_base(
                                    "text", item[4], number, list(item[:4]),
                                    float(image_pdf[offset].rect.width),
                                    float(image_pdf[offset].rect.height),
                                    f"p{number:04d}-fallback-{index:03d}"))
                    except Exception as fallback_exc:
                        _warn(warnings, path.name, number, "page_fallback_failed", fallback_exc)
                try:
                    page = image_pdf[offset]
                    regions = _visual_regions(page, table_boxes, bool(page_blocks))
                    page_blocks.extend(_visual_blocks(
                        page, number, path.name, regions, vision, warnings))
                except Exception as exc:
                    _warn(warnings, path.name, number, "page_visual_failed", exc)
                page_blocks.sort(key=lambda item: (
                    item["bbox"][1] if item["bbox"] else 0,
                    item["bbox"][0] if item["bbox"] else 0))
                blocks.extend(page_blocks)
                page_texts.append("\n\n".join(item["text"] for item in page_blocks))
                report(5 + round(30 * number / pages), f"Parsing page {number}/{pages}")
        finally:
            if text_pdf is not None:
                text_pdf.close()
    if not blocks:
        raise ValueError("No readable text, tables, or figures were recovered")
    text_path = path.with_suffix(".txt")
    temporary = text_path.with_suffix(".txt.tmp")
    try:
        temporary.write_text("\n\n\f\n\n".join(page_texts), encoding="utf-8")
        temporary.replace(text_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"segments": blocks, "pages": pages,
            "characters": sum(len(text) for text in page_texts),
            "text_path": text_path.name, "warnings": warnings}
