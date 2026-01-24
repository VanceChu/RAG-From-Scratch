"""Embedding model wrapper."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

from openai import OpenAI

_OPENAI_EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _normalize(embedding: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in embedding))
    if norm == 0:
        return embedding
    return [value / norm for value in embedding]


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    if batch_size <= 0:
        yield items
        return
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        provider: str = "sentence-transformers",
        api_key: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.provider = provider.strip().lower()
        self._dimension: int

        if self.provider in {"openai", "openai-embeddings"}:
            self.client = OpenAI(api_key=api_key)
            resolved_dim = embedding_dim or _OPENAI_EMBEDDING_DIMS.get(model_name)
            if not resolved_dim:
                raise ValueError(
                    "Unknown embedding dimension for OpenAI model "
                    f"'{model_name}'. Set RAG_EMBEDDING_DIM or pass --embedding-dim."
                )
            self._dimension = int(resolved_dim)
            self._backend = "openai"
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
            self._backend = "sentence-transformers"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Iterable[str], batch_size: int = 64) -> List[list[float]]:
        items = list(texts)
        if not items:
            return []
        if self._backend == "openai":
            return self._embed_openai(items, batch_size)

        embeddings = self.model.encode(
            items,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], batch_size=1)[0]

    def _embed_openai(self, texts: list[str], batch_size: int) -> List[list[float]]:
        embeddings: list[list[float]] = []
        for batch in _batched(texts, batch_size):
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            for item in sorted(response.data, key=lambda entry: entry.index):
                embeddings.append(_normalize(item.embedding))
        return embeddings
