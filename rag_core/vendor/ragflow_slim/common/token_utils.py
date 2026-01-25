"""Network-free token utilities for ragflow_slim."""

from __future__ import annotations

import re
from typing import Iterable

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text or "")


class _SimpleEncoder:
    def encode(self, text: str) -> list[int]:
        return list(range(len(_tokens(text))))

    def decode(self, token_ids: Iterable[int]) -> str:
        # Best-effort: return a placeholder with the same token count.
        count = len(list(token_ids))
        return " ".join(["tok"] * count)


encoder = _SimpleEncoder()


def num_tokens_from_string(string: str) -> int:
    return len(_tokens(string))


def truncate_by_tokens(string: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    toks = _tokens(string)
    return " ".join(toks[:max_len])

