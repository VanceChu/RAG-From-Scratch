"""Query embedding + vector search."""

from __future__ import annotations

from typing import Iterable

from rag_core.embeddings import EmbeddingModel
from rag_core.vector_store import SearchResult, VectorStore


def retrieve(
    query: str,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    top_k: int,
    search_k: int | None = None,
) -> list[SearchResult]:
    if not query.strip():
        return []
    embedding = embedding_model.embed_query(query)
    limit = search_k if search_k is not None else top_k
    return vector_store.search(embedding, limit=limit)
