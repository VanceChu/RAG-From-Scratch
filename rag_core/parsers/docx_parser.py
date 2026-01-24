"""DOCX to markdown parser."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from docx import Document


def _heading_level(style_name: str) -> int | None:
    if not style_name:
        return None
    if not style_name.startswith("Heading"):
        return None
    parts = style_name.split()
    if parts and parts[-1].isdigit():
        return int(parts[-1])
    return None


def _is_list_paragraph(style_name: str) -> bool:
    if not style_name:
        return False
    lowered = style_name.lower()
    return "list" in lowered


def parse_docx_to_markdown(path: Path) -> Tuple[str, dict]:
    doc = Document(path)
    lines: list[str] = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = paragraph.style.name if paragraph.style else ""
        level = _heading_level(style_name)
        if level:
            level = max(1, min(level, 6))
            lines.append("#" * level + " " + text)
            continue
        if _is_list_paragraph(style_name):
            lines.append(f"- {text}")
            continue
        lines.append(text)

    markdown = "\n\n".join(lines)
    metadata = {"paragraph_count": len(lines)}
    return markdown, metadata
