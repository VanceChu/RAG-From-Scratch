"""BM25 index for lexical retrieval."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rag_core.config import DEFAULT_BM25_DIR
from rag_core.vector_store import SearchResult

logger = logging.getLogger(__name__)

_INDEX_VERSION = 1
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _safe_collection_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    return safe or "collection"


@dataclass
class BM25Entry:
    chunk_id: int
    text: str
    metadata: dict
    tokens: list[str]


class BM25Index:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[BM25Entry] = []
        self._stats_cache: tuple[Counter, float, int] | None = None

    @classmethod
    def load(cls, collection: str, base_dir: Path | None = None) -> "BM25Index":
        root = base_dir or DEFAULT_BM25_DIR
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{_safe_collection_name(collection)}.json"
        index = cls(path)
        if not path.exists():
            return index
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("BM25 index file is corrupted: %s", path)
            return index
        if payload.get("version") != _INDEX_VERSION:
            logger.warning("BM25 index version mismatch: %s", path)
            return index
        for item in payload.get("entries", []):
            try:
                chunk_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            index.entries.append(
                BM25Entry(
                    chunk_id=chunk_id,
                    text=str(item.get("text") or ""),
                    metadata=item.get("metadata") or {},
                    tokens=list(item.get("tokens") or []),
                )
            )
        return index

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _INDEX_VERSION,
            "entries": [
                {
                    "id": entry.chunk_id,
                    "text": entry.text,
                    "metadata": entry.metadata,
                    "tokens": entry.tokens,
                }
                for entry in self.entries
            ],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.entries = []
        self._stats_cache = None
        if self.path.exists():
            self.path.unlink()

    def add(
        self,
        ids: Iterable[int],
        texts: Iterable[str],
        metadatas: Iterable[dict],
    ) -> None:
        added = False
        for chunk_id, text, metadata in zip(ids, texts, metadatas):
            tokens = _tokenize(text)
            if not tokens:
                continue
            self.entries.append(
                BM25Entry(
                    chunk_id=int(chunk_id),
                    text=text,
                    metadata=metadata,
                    tokens=tokens,
                )
            )
            added = True
        if added:
            self._stats_cache = None

    def remove(self, ids: Iterable[int]) -> None:
        remove_set = {int(value) for value in ids}
        if not remove_set:
            return
        self.entries = [
            entry for entry in self.entries if entry.chunk_id not in remove_set
        ]
        self._stats_cache = None

    def _ensure_stats(self) -> tuple[Counter, float, int]:
        if self._stats_cache is not None:
            return self._stats_cache
        df: Counter[str] = Counter()
        total_len = 0
        for entry in self.entries:
            tokens = entry.tokens
            total_len += len(tokens)
            for term in set(tokens):
                df[term] += 1
        doc_count = len(self.entries)
        avgdl = (total_len / doc_count) if doc_count else 0.0
        self._stats_cache = (df, avgdl, doc_count)
        return self._stats_cache

    @staticmethod
    def _bm25_score(
        query_terms: list[str],
        entry: BM25Entry,
        df: Counter,
        avgdl: float,
        doc_count: int,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> float:
        if not query_terms or not entry.tokens or doc_count == 0:
            return 0.0
        term_counts = Counter(entry.tokens)
        doc_len = len(entry.tokens)
        score = 0.0
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            df_t = df.get(term, 0)
            idf = math.log(1.0 + (doc_count - df_t + 0.5) / (df_t + 0.5))
            denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl if avgdl else 0.0))
            score += idf * ((tf * (k1 + 1.0)) / (denom if denom else 1.0))
        return score

    def search(self, query: str, limit: int) -> list[SearchResult]:
        if limit <= 0:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        if not self.entries:
            return []
        df, avgdl, doc_count = self._ensure_stats()
        query_terms = list(dict.fromkeys(tokens))
        scored = []
        for index, entry in enumerate(self.entries):
            score = self._bm25_score(query_terms, entry, df, avgdl, doc_count)
            scored.append((index, score))
        top_indices = sorted(scored, key=lambda item: float(item[1]), reverse=True)[:limit]
        results: list[SearchResult] = []
        for index, score in top_indices:
            entry = self.entries[index]
            results.append(
                SearchResult(
                    text=entry.text,
                    metadata=entry.metadata,
                    score=float(score),
                    id=entry.chunk_id,
                )
            )
        return results
