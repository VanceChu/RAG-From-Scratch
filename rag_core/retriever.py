"""Query embedding + vector search."""

from __future__ import annotations

import logging
import time

from rag_core.embeddings import EmbeddingModel
from rag_core.bm25_index import BM25Index
from rag_core.sparse_embedding import SparseEmbeddingModel
from rag_core.vector_store import SearchResult, VectorStore


logger = logging.getLogger(__name__)


def _normalize_scores(
    results: list[SearchResult],
    higher_is_better: bool,
) -> dict[int, float]:
    if not results:
        return {}
    scores = [result.score for result in results]
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return {
            result.id: 1.0 for result in results if result.id is not None
        }
    normalized: dict[int, float] = {}
    for result in results:
        if result.id is None:
            continue
        if higher_is_better:
            normalized[result.id] = (result.score - min_score) / (max_score - min_score)
        else:
            normalized[result.id] = (max_score - result.score) / (max_score - min_score)
    return normalized


def _merge_dense_bm25(
    dense_results: list[SearchResult],
    bm25_results: list[SearchResult],
    dense_weight: float,
    metric_type: str,
    limit: int,
) -> list[SearchResult]:
    if not bm25_results:
        return dense_results[:limit]
    if not dense_results:
        return bm25_results[:limit]

    higher_is_better = metric_type.upper() != "L2"
    dense_scores = _normalize_scores(dense_results, higher_is_better=higher_is_better)
    bm25_scores = _normalize_scores(bm25_results, higher_is_better=True)

    results_by_id: dict[int, SearchResult] = {}
    for result in dense_results:
        if result.id is None:
            continue
        results_by_id[result.id] = result
    for result in bm25_results:
        if result.id is None:
            continue
        results_by_id.setdefault(result.id, result)

    merged: list[SearchResult] = []
    for chunk_id, result in results_by_id.items():
        dense_score = dense_scores.get(chunk_id, 0.0)
        bm25_score = bm25_scores.get(chunk_id, 0.0)
        score = dense_weight * dense_score + (1.0 - dense_weight) * bm25_score
        merged.append(
            SearchResult(
                text=result.text,
                metadata=result.metadata,
                score=score,
                id=chunk_id,
            )
        )
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:limit]


def retrieve(
    query: str,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    top_k: int,
    search_k: int | None = None,
    sparse_embedding_model: SparseEmbeddingModel | None = None,
    bm25_index: BM25Index | None = None,
    hybrid_alpha: float = 0.5,
    timing: dict | None = None,
) -> list[SearchResult]:
    if not query.strip():
        return []
    if timing is not None:
        timing.clear()

    total_start = time.perf_counter()
    embed_start = time.perf_counter()
    embedding = embedding_model.embed_query(query)
    embed_elapsed = time.perf_counter() - embed_start
    if timing is not None:
        timing["embed_query_s"] = embed_elapsed

    sparse_vector = None
    if sparse_embedding_model and not bm25_index:
        # APISparseEmbeddingModel takes a list of texts
        sparse_vectors = sparse_embedding_model.embed_sparse([query])
        if sparse_vectors and sparse_vectors[0]:
            sparse_vector = sparse_vectors[0]
    elif sparse_embedding_model and bm25_index:
        logger.warning("BM25 index enabled; ignoring sparse embedding model.")

    limit = search_k if search_k is not None else top_k
    dense_start = time.perf_counter()
    dense_results = vector_store.search(
        embedding,
        limit=limit,
        sparse_vector=sparse_vector,
        hybrid_alpha=hybrid_alpha,
    )
    dense_elapsed = time.perf_counter() - dense_start
    if timing is not None:
        timing["dense_search_s"] = dense_elapsed
        timing["dense_results"] = len(dense_results)

    if bm25_index:
        bm25_start = time.perf_counter()
        bm25_results = bm25_index.search(query, limit=limit)
        bm25_elapsed = time.perf_counter() - bm25_start
        if timing is not None:
            timing["bm25_search_s"] = bm25_elapsed
            timing["bm25_results"] = len(bm25_results)
        merge_start = time.perf_counter()
        merged = _merge_dense_bm25(
            dense_results=dense_results,
            bm25_results=bm25_results,
            dense_weight=hybrid_alpha,
            metric_type=vector_store.metric_type,
            limit=limit,
        )
        merge_elapsed = time.perf_counter() - merge_start
        if timing is not None:
            timing["merge_s"] = merge_elapsed
            timing["total_s"] = time.perf_counter() - total_start
            timing["merged_results"] = len(merged)
        return merged

    if timing is not None:
        timing["total_s"] = time.perf_counter() - total_start
    return dense_results
