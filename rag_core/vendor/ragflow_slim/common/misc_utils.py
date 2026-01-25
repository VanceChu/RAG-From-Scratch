"""Minimal misc utilities for ragflow_slim.

These stubs avoid network installs (e.g., torch) while keeping the async
thread-pool helper used by the parsers.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_MAX_WORKERS = max(1, int(os.getenv("RAGFLOW_THREAD_POOL_WORKERS", "4")))
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS)


def pip_install_torch() -> bool:
    """No-op torch installer.

    The upstream implementation tries to install torch at runtime, which is
    undesirable in this slim vendor context.
    """
    return False


def hash_str2int(value: str, modulo: int = 2**31 - 1) -> int:
    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest, 16) % modulo


def convert_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:0.2f}{unit}"
        size /= 1024.0
    return f"{size:0.2f}TB"


async def thread_pool_exec(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXECUTOR, lambda: func(*args, **kwargs))

