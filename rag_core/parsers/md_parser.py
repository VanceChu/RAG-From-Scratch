"""Markdown parser (pass-through)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


def parse_markdown(path: Path) -> Tuple[str, dict]:
    text = path.read_text(encoding="utf-8")
    return text, {"line_count": text.count("\n") + 1}
