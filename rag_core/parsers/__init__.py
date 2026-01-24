"""Document parsing entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .docx_parser import parse_docx_to_markdown
from .md_parser import parse_markdown
from .pdf_parser import parse_pdf_to_markdown


def parse_document(path: Path) -> Tuple[str, dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf_to_markdown(path)
    if suffix in {".docx", ".doc"}:
        return parse_docx_to_markdown(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return parse_markdown(path)
    raise ValueError(f"Unsupported file type: {suffix}")
