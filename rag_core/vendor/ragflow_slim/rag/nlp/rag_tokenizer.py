"""Lightweight tokenizer stub for ragflow_slim.

The upstream tokenizer depends on infinity-sdk. This fallback keeps the
interface but uses simple regex-based tokenization.
"""

from __future__ import annotations

import re
from collections import Counter


def _basic_tokens(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)


class RagTokenizer:
    def tokenize(self, line: str) -> str:
        return " ".join(_basic_tokens(line))

    def fine_grained_tokenize(self, tokens: str) -> str:
        return tokens if tokens else ""

    def tag(self, token: str) -> str:
        # Minimal heuristic; upstream returns POS tags.
        return "nr" if token.istitle() else ""

    def freq(self, token: str) -> int:
        return Counter(_basic_tokens(token)).get(token, 1)

    def naive_qie(self, text: str) -> str:
        return self.tokenize(text)

    def tradi2simp(self, text: str) -> str:
        return text

    def strQ2B(self, text: str) -> str:
        return text


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
naive_qie = tokenizer.naive_qie

# Preserve attribute names referenced upstream.
_tradi2simp = tokenizer.tradi2simp
_strQ2B = tokenizer.strQ2B

