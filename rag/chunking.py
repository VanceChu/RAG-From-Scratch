"""Chunk markdown content into overlapping sections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
PAGE_RE = re.compile(r"^page\s+(\d+)$", re.IGNORECASE)


@dataclass
class Chunk:
    text: str
    metadata: dict


def _split_markdown_sections(markdown: str) -> list[dict]:
    sections: list[dict] = []
    current = {"heading": "Document", "level": 0, "lines": []}

    for line in markdown.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            if current["lines"]:
                sections.append(current)
            current = {
                "heading": match.group(2).strip(),
                "level": len(match.group(1)),
                "lines": [],
            }
            continue
        current["lines"].append(line)

    if current["lines"]:
        sections.append(current)

    return sections


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if chunk_size <= 0:
        return [text.strip()]
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            break_at = text.rfind("\n\n", start + chunk_size // 2, end)
            if break_at > start:
                end = break_at
        if end <= start:
            end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def chunk_markdown(
    markdown: str,
    source: Path,
    chunk_size: int,
    overlap: int,
) -> Iterable[Chunk]:
    sections = _split_markdown_sections(markdown)

    for section_index, section in enumerate(sections):
        content = "\n".join(section["lines"]).strip()
        for chunk_index, chunk_text in enumerate(
            _chunk_text(content, chunk_size, overlap)
        ):
            heading = section["heading"]
            page = None
            page_match = PAGE_RE.match(heading.strip())
            if page_match:
                page = int(page_match.group(1))
            metadata = {
                "source": str(source),
                "section": heading,
                "section_index": section_index,
                "chunk_index": chunk_index,
                "page": page if page is not None else -1,
            }
            yield Chunk(text=chunk_text, metadata=metadata)
