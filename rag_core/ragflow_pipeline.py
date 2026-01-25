"""RagFlow-based parsing and splitting pipeline."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image

from rag_core.config import DEFAULT_IMAGE_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "ragflow_slim"
DEFAULT_RAG_PROJECT_BASE = PROJECT_ROOT / "data" / "index" / "ragflow"
LEGACY_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "ragflow"
LEGACY_MODEL_DIR = LEGACY_VENDOR_ROOT / "rag" / "res" / "deepdoc"

if "RAG_PROJECT_BASE" not in os.environ:
    if LEGACY_MODEL_DIR.exists():
        # Reuse existing local models to avoid forced downloads.
        os.environ["RAG_PROJECT_BASE"] = str(LEGACY_VENDOR_ROOT)
    else:
        DEFAULT_RAG_PROJECT_BASE.mkdir(parents=True, exist_ok=True)
        os.environ["RAG_PROJECT_BASE"] = str(DEFAULT_RAG_PROJECT_BASE)
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from deepdoc.parser.pdf_parser import RAGFlowPdfParser  # noqa: E402
from deepdoc.parser.docx_parser import RAGFlowDocxParser  # noqa: E402
from deepdoc.vision import OCR  # noqa: E402
from rag.nlp import naive_merge_with_images  # noqa: E402


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".md",
    ".markdown",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
}


def _ensure_image_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_image(image: Image.Image, image_dir: Path, prefix: str) -> str:
    _ensure_image_dir(image_dir)
    safe_prefix = "".join(ch if ch.isalnum() else "_" for ch in prefix) or "image"
    filename = f"{safe_prefix}_{uuid.uuid4().hex}.jpg"
    target = image_dir / filename
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(target, format="JPEG")
    image.close()
    return str(target)


def _parse_pdf(path: Path) -> Tuple[list[dict], dict]:
    parser = RAGFlowPdfParser()
    bboxes = parser.parse_into_bboxes(str(path))
    sections: list[dict] = []
    for b in bboxes:
        text = b.get("text") or ""
        position_tag = b.get("position_tag") or ""
        image = b.get("image")
        layout = (b.get("layout_type") or "").lower()
        doc_type = ""
        if layout == "table":
            doc_type = "table"
        elif layout == "figure":
            doc_type = "image"
        elif image is not None and not text.strip():
            doc_type = "image"
        if doc_type not in {"table", "image"} and text.strip():
            image = None
        sections.append(
            {
                "text": text,
                "position_tag": position_tag,
                "image": image,
                "doc_type_kwd": doc_type,
            }
        )
    page_count = len(getattr(parser, "page_images", []) or [])
    return sections, {"page_count": page_count}


def _parse_docx(path: Path) -> Tuple[list[dict], dict]:
    parser = RAGFlowDocxParser()
    lines, tables = parser(str(path))
    sections: list[dict] = []
    for text, _style in lines:
        sections.append(
            {
                "text": text or "",
                "position_tag": "",
                "image": None,
                "doc_type_kwd": "",
            }
        )
    for table in tables:
        table_text = "\n".join(table) if isinstance(table, list) else str(table)
        sections.append(
            {
                "text": table_text,
                "position_tag": "",
                "image": None,
                "doc_type_kwd": "table",
            }
        )
    return sections, {"paragraph_count": len(lines), "table_count": len(tables)}


def _parse_markdown(path: Path) -> Tuple[list[dict], dict]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    return (
        [
            {
                "text": content,
                "position_tag": "",
                "image": None,
                "doc_type_kwd": "",
            }
        ],
        {"char_count": len(content)},
    )


def _parse_image(path: Path) -> Tuple[list[dict], dict]:
    img = Image.open(path).convert("RGB")
    ocr = OCR()
    boxes = ocr(np.array(img))
    text = "\n".join([t[0] for _, t in boxes if t and t[0]])
    return (
        [
            {
                "text": text,
                "position_tag": "",
                "image": img,
                "doc_type_kwd": "image",
            }
        ],
        {"char_count": len(text)},
    )


def _parse_sections(path: Path) -> Tuple[list[dict], dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix in {".docx", ".doc"}:
        return _parse_docx(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return _parse_markdown(path)
    if suffix in {".png", ".jpg", ".jpeg"}:
        return _parse_image(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _normalize_positions(positions: Iterable[tuple]) -> list[dict]:
    normalized: list[dict] = []
    for pages, left, right, top, bottom in positions:
        normalized.append(
            {
                "pages": [int(p) + 1 for p in pages],
                "left": float(left),
                "right": float(right),
                "top": float(top),
                "bottom": float(bottom),
            }
        )
    return normalized


def _split_sections(
    sections: list[dict],
    chunk_token_size: int,
    overlap_tokens: int,
    delimiter: str,
    image_dir: Path,
) -> list[dict]:
    if not sections:
        return []
    pairs = [(sec.get("text") or "", sec.get("position_tag") or "") for sec in sections]
    images = [sec.get("image") for sec in sections]
    overlapped_percent = 0.0
    if chunk_token_size > 0 and overlap_tokens > 0:
        overlapped_percent = min(100.0, overlap_tokens / float(chunk_token_size) * 100.0)
    chunks, chunk_images = naive_merge_with_images(
        pairs,
        images,
        chunk_token_size,
        delimiter,
        overlapped_percent,
    )
    output: list[dict] = []
    for idx, (chunk_text, image) in enumerate(zip(chunks, chunk_images)):
        positions = RAGFlowPdfParser.extract_positions(chunk_text)
        clean_text = RAGFlowPdfParser.remove_tag(chunk_text).strip()
        metadata: dict = {"chunk_index": idx}
        if positions:
            metadata["positions"] = _normalize_positions(positions)
        if image is not None:
            image_path = _save_image(image, image_dir, f"chunk_{idx}")
            metadata["image_path"] = image_path
            metadata["img_id"] = image_path
        if "<table" in clean_text.lower():
            metadata["doc_type"] = "table"
        elif metadata.get("image_path"):
            metadata["doc_type"] = "image"
        else:
            metadata["doc_type"] = "text"
        if not clean_text and metadata.get("image_path"):
            clean_text = "[IMAGE]"
        if clean_text:
            output.append({"text": clean_text, "metadata": metadata})
    return output


def parse_and_split_document(
    path: Path,
    chunk_token_size: int,
    overlap_tokens: int,
    delimiter: str = "\n",
    image_dir: Path | None = None,
) -> list[dict]:
    sections, doc_metadata = _parse_sections(path)
    image_dir = image_dir or DEFAULT_IMAGE_DIR
    chunks = _split_sections(
        sections=sections,
        chunk_token_size=chunk_token_size,
        overlap_tokens=overlap_tokens,
        delimiter=delimiter,
        image_dir=image_dir,
    )
    for chunk in chunks:
        chunk["metadata"]["source"] = str(path)
        chunk["metadata"]["doc_metadata"] = doc_metadata
        chunk["metadata"]["file_type"] = path.suffix.lower().lstrip(".")
    return chunks
