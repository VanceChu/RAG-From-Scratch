"""Query embedding + vector search with query rewriting support."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Sequence

from rag_core.bm25_index import BM25Index
from rag_core.config import (
    DEFAULT_ENABLE_HYDE,
    DEFAULT_ENABLE_QUERY_DECOMPOSITION,
    DEFAULT_ENABLE_QUERY_EXPANSION,
    DEFAULT_ENABLE_STEP_BACK,
)
from rag_core.embeddings import EmbeddingModel
from rag_core.llm import OpenAIChatLLM
from rag_core.query_rewriter import (
    ConversationTurn,
    QueryRewriterPipeline,
    RewriteResult,
    RewriteStrategy,
)
from rag_core.observability import TraceContext, span
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


def _merge_rrf(
    dense_results: list[SearchResult],
    bm25_results: list[SearchResult],
    limit: int,
    k: int = 60,
) -> list[SearchResult]:
    if not bm25_results:
        return dense_results[:limit]
    if not dense_results:
        return bm25_results[:limit]

    k = max(1, int(k))
    scores: dict[int, float] = defaultdict(float)
    results_by_id: dict[int, SearchResult] = {}

    def add_results(results: list[SearchResult]) -> None:
        for rank, result in enumerate(results, start=1):
            if result.id is None:
                continue
            scores[result.id] += 1.0 / (k + rank)
            if result.id not in results_by_id:
                results_by_id[result.id] = result

    add_results(dense_results)
    add_results(bm25_results)

    merged: list[SearchResult] = []
    for chunk_id, score in scores.items():
        result = results_by_id[chunk_id]
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
    fusion: str = "weighted",
    rrf_k: int = 60,
    timing: dict | None = None,
    trace_ctx: TraceContext | None = None,
) -> list[SearchResult]:
    if not query.strip():
        return []
    if timing is not None:
        timing.clear()

    with span(trace_ctx, "retrieve", metadata={"top_k": top_k, "search_k": search_k}) as retrieve_ctx:
        total_start = time.perf_counter()
        with span(retrieve_ctx, "embed_query"):
            embed_start = time.perf_counter()
            embedding = embedding_model.embed_query(query)
            embed_elapsed = time.perf_counter() - embed_start
            if timing is not None:
                timing["embed_query_s"] = embed_elapsed

        sparse_vector = None
        if sparse_embedding_model and not bm25_index:
            with span(retrieve_ctx, "sparse_embed"):
                # APISparseEmbeddingModel takes a list of texts
                sparse_vectors = sparse_embedding_model.embed_sparse([query])
                if sparse_vectors and sparse_vectors[0]:
                    sparse_vector = sparse_vectors[0]
        elif sparse_embedding_model and bm25_index:
            logger.warning("BM25 index enabled; ignoring sparse embedding model.")

        limit = search_k if search_k is not None else top_k
        with span(retrieve_ctx, "dense_search", metadata={"limit": limit}):
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

        fusion_mode = (fusion or "weighted").strip().lower()
        if fusion_mode not in {"weighted", "rrf", "dense"}:
            raise ValueError(f"Unsupported fusion mode: {fusion}")

        if bm25_index and fusion_mode != "dense":
            with span(retrieve_ctx, "bm25_search", metadata={"limit": limit}):
                bm25_start = time.perf_counter()
                bm25_results = bm25_index.search(query, limit=limit)
                bm25_elapsed = time.perf_counter() - bm25_start
                if timing is not None:
                    timing["bm25_search_s"] = bm25_elapsed
                    timing["bm25_results"] = len(bm25_results)
            with span(retrieve_ctx, "fusion", metadata={"mode": fusion_mode}):
                merge_start = time.perf_counter()
                if fusion_mode == "rrf":
                    merged = _merge_rrf(
                        dense_results=dense_results,
                        bm25_results=bm25_results,
                        limit=limit,
                        k=rrf_k,
                    )
                else:
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


def retrieve_with_rewrite(
    query: str,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    top_k: int,
    llm: OpenAIChatLLM | None = None,
    conversation_history: Sequence[ConversationTurn] | None = None,
    enable_rewrite: bool = True,
    rewrite_strategies: list[str] | None = None,
    search_k: int | None = None,
    sparse_embedding_model: SparseEmbeddingModel | None = None,
    bm25_index: BM25Index | None = None,
    hybrid_alpha: float = 0.5,
    fusion: str = "weighted",
    rrf_k: int = 60,
    timing: dict | None = None,
    trace_ctx: TraceContext | None = None,
) -> tuple[list[SearchResult], RewriteResult | None]:
    """
    Retrieve with optional query rewriting.

    For single-query strategies (contextual, hyde), uses the rewritten query directly.
    For multi-query strategies (expansion, decomposition, step_back), performs
    multi-query retrieval with RRF fusion.

    Args:
        query: The user's query.
        embedding_model: The embedding model.
        vector_store: The vector store.
        top_k: Number of results to return.
        llm: LLM for query rewriting (required if enable_rewrite is True).
        conversation_history: Previous conversation turns for context.
        enable_rewrite: Whether to perform query rewriting.
        rewrite_strategies: List of strategy names to enable.
        search_k: Number of candidates to retrieve before filtering.
        sparse_embedding_model: Optional sparse embedding model.
        bm25_index: Optional BM25 index for hybrid retrieval.
        hybrid_alpha: Weight for dense vs sparse/BM25 fusion.
        fusion: Fusion strategy ("weighted", "rrf", or "dense").
        rrf_k: RRF k parameter.
        timing: Optional dict to store timing information.

    Returns:
        Tuple of (search results, rewrite result or None).
    """
    rewrite_result: RewriteResult | None = None

    if enable_rewrite and llm:
        with span(trace_ctx, "query_rewrite", metadata={"strategies": rewrite_strategies}) as rewrite_ctx:
            rewrite_start = time.perf_counter()

            strategies = None
            if rewrite_strategies:
                strategies = [
                    RewriteStrategy(s.strip())
                    for s in rewrite_strategies
                    if s.strip() in [e.value for e in RewriteStrategy]
                ]

            pipeline = QueryRewriterPipeline(
                llm=llm,
                strategies=strategies,
                enable_contextual=not strategies,
                enable_expansion=DEFAULT_ENABLE_QUERY_EXPANSION if not strategies else False,
                enable_hyde=DEFAULT_ENABLE_HYDE if not strategies else False,
                enable_decomposition=DEFAULT_ENABLE_QUERY_DECOMPOSITION if not strategies else False,
                enable_step_back=DEFAULT_ENABLE_STEP_BACK if not strategies else False,
            )

            previous_ctx = getattr(llm, "trace_ctx", None)
            try:
                llm.trace_ctx = rewrite_ctx
                rewrite_result = pipeline.rewrite(query, conversation_history)
            finally:
                llm.trace_ctx = previous_ctx

            if timing is not None:
                timing["rewrite_s"] = time.perf_counter() - rewrite_start
                timing["rewrite_strategy"] = rewrite_result.strategy.value
                timing["original_query"] = query
                timing["rewritten_queries"] = rewrite_result.rewritten_queries

            rewritten_queries = rewrite_result.rewritten_queries

            if len(rewritten_queries) > 1:
                logger.info(
                    f"Multi-query rewrite: '{query}' -> {len(rewritten_queries)} queries"
                )
                results = retrieve_multi_query(
                    queries=rewritten_queries,
                    embedding_model=embedding_model,
                    vector_store=vector_store,
                    top_k=top_k,
                    search_k=search_k,
                    sparse_embedding_model=sparse_embedding_model,
                    bm25_index=bm25_index,
                    hybrid_alpha=hybrid_alpha,
                    fusion=fusion,
                    rrf_k=rrf_k,
                    timing=timing,
                    trace_ctx=rewrite_ctx,
                )
                return results, rewrite_result
            else:
                effective_query = rewrite_result.primary_query
                logger.info(f"Query rewrite: '{query}' -> '{effective_query}'")
                if timing is not None:
                    timing["rewritten_query"] = effective_query
    else:
        effective_query = query

    results = retrieve(
        query=effective_query,
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=top_k,
        search_k=search_k,
        sparse_embedding_model=sparse_embedding_model,
        bm25_index=bm25_index,
        hybrid_alpha=hybrid_alpha,
        fusion=fusion,
        rrf_k=rrf_k,
        timing=timing,
        trace_ctx=trace_ctx,
    )

    return results, rewrite_result


def retrieve_multi_query(
    queries: list[str],
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    top_k: int,
    search_k: int | None = None,
    sparse_embedding_model: SparseEmbeddingModel | None = None,
    bm25_index: BM25Index | None = None,
    hybrid_alpha: float = 0.5,
    fusion: str = "weighted",
    rrf_k: int = 60,
    timing: dict | None = None,
    trace_ctx: TraceContext | None = None,
) -> list[SearchResult]:
    """
    Retrieve using multiple query variants and merge results via RRF.

    Used with query expansion or decomposition strategies.

    Args:
        queries: List of query variants.
        embedding_model: The embedding model.
        vector_store: The vector store.
        top_k: Number of final results to return.
        search_k: Number of candidates per query.
        sparse_embedding_model: Optional sparse embedding model.
        bm25_index: Optional BM25 index.
        hybrid_alpha: Weight for dense vs sparse/BM25 fusion.
        fusion: Fusion strategy ("weighted", "rrf", or "dense").
        rrf_k: RRF k parameter for final multi-query fusion.
        timing: Optional dict to store timing information.

    Returns:
        Merged and deduplicated search results.
    """
    if not queries:
        return []

    with span(trace_ctx, "retrieve_multi_query", metadata={"query_count": len(queries)}):
        total_start = time.perf_counter()
        all_results: list[list[SearchResult]] = []
        per_query_timing: list[dict] = []

        limit = search_k if search_k is not None else top_k

        for q in queries:
            q_timing: dict = {}
            results = retrieve(
                query=q,
                embedding_model=embedding_model,
                vector_store=vector_store,
                top_k=limit,
                search_k=search_k,
                sparse_embedding_model=sparse_embedding_model,
                bm25_index=bm25_index,
                hybrid_alpha=hybrid_alpha,
                fusion=fusion,
                rrf_k=rrf_k,
                timing=q_timing,
                trace_ctx=trace_ctx,
            )
            all_results.append(results)
            per_query_timing.append(q_timing)

        merged = _merge_multi_query_results(all_results, top_k, rrf_k)

        if timing is not None:
            timing["multi_query_count"] = len(queries)
            timing["per_query_timing"] = per_query_timing
            timing["total_s"] = time.perf_counter() - total_start

        return merged


def _merge_multi_query_results(
    all_results: list[list[SearchResult]],
    limit: int,
    k: int = 60,
) -> list[SearchResult]:
    """Merge results from multiple queries using RRF."""
    scores: dict[int, float] = defaultdict(float)
    results_by_id: dict[int, SearchResult] = {}

    for results in all_results:
        for rank, result in enumerate(results, start=1):
            if result.id is None:
                continue
            scores[result.id] += 1.0 / (k + rank)
            if result.id not in results_by_id:
                results_by_id[result.id] = result

    merged = []
    for chunk_id, score in scores.items():
        result = results_by_id[chunk_id]
        merged.append(
            SearchResult(
                text=result.text,
                metadata=result.metadata,
                score=score,
                id=chunk_id,
            )
        )

    merged.sort(key=lambda x: x.score, reverse=True)
    return merged[:limit]
