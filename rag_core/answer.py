"""Answer generation using retrieved context."""

from __future__ import annotations

from typing import Iterable

from rag_core.llm import OpenAIChatLLM
from rag_core.vector_store import SearchResult


def _context_block(index: int, result: SearchResult) -> str:
    metadata = result.metadata or {}
    source = metadata.get("source", "unknown")
    section = metadata.get("section", "unknown")
    page = metadata.get("page", -1)
    page_label = str(page) if isinstance(page, int) and page >= 0 else "n/a"
    return (
        f"[{index}] source: {source}\n"
        f"section: {section}\n"
        f"page: {page_label}\n"
        "content:\n"
        f"{result.text}"
    )


def build_context(results: list[SearchResult]) -> str:
    blocks = [_context_block(index, result) for index, result in enumerate(results, 1)]
    return "\n\n".join(blocks)


def answer_question(
    query: str,
    results: list[SearchResult],
    llm: OpenAIChatLLM,
) -> str:
    context = build_context(results)
    system_prompt = (
        "You answer questions using only the provided context. "
        "If the context is insufficient, say you do not know. "
        "Cite sources like [1], [2]."
    )
    user_prompt = (
        "Question:\n"
        f"{query}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Answer in the same language as the question."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return llm.complete(messages)
