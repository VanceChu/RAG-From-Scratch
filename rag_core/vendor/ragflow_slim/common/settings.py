"""Minimal settings stub for ragflow_slim."""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


# Keep the surface area intentionally small. The parsers only rely on this.
PARALLEL_DEVICES: int = _env_int("RAGFLOW_PARALLEL_DEVICES", 1)
DOC_ENGINE_INFINITY: bool = os.getenv("DOC_ENGINE", "").lower() == "infinity"

