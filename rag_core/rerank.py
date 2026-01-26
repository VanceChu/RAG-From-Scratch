"""Cross-encoder reranking."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag_core.vector_store import SearchResult


class Reranker:
    def __init__(self, model_name: str) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, result.text) for result in results]
        scores = self.model.predict(pairs)
        reranked = sorted(
            zip(results, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        trimmed = reranked[:top_k] if top_k > 0 else reranked
        return [
            SearchResult(
                text=result.text,
                metadata=result.metadata,
                score=float(score),
                id=result.id,
            )
            for result, score in trimmed
        ]
