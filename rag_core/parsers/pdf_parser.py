"""PDF to markdown parser."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pdfplumber


def parse_pdf_to_markdown(path: Path) -> Tuple[str, dict]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            pages.append(f"# Page {index}\n{text}")
    markdown = "\n\n".join(pages)
    metadata = {"page_count": len(pages)}
    return markdown, metadata
