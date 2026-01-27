"""Answer generation using retrieved context."""

from __future__ import annotations

from typing import Callable, Sequence

from rag_core.llm import OpenAIChatLLM
from rag_core.query_rewriter import ConversationTurn
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


def _history_block(
    conversation_history: Sequence[ConversationTurn] | None,
    max_turns: int = 3,
) -> str:
    if not conversation_history:
        return ""
    recent_turns = list(conversation_history[-max_turns:])
    lines = ["Conversation history (most recent last):"]
    for index, turn in enumerate(recent_turns, start=1):
        lines.append(f"Q{index}: {turn.query}")
        lines.append(f"A{index}: {turn.response}")
    return "\n".join(lines)


def _build_messages(
    query: str,
    results: list[SearchResult],
    conversation_history: Sequence[ConversationTurn] | None,
    history_turns: int,
) -> list[dict]:
    context = build_context(results)
    history = _history_block(conversation_history, max_turns=history_turns)
    history_section = f"{history}\n\n" if history else ""
    system_prompt = (
        "You answer questions using only the provided context. "
        "If the context is insufficient, say you do not know. "
        "Cite sources like [1], [2]."
    )
    user_prompt = (
        f"{history_section}"
        "Question:\n"
        f"{query}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Answer in the same language as the question."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def answer_question(
    query: str,
    results: list[SearchResult],
    llm: OpenAIChatLLM,
    conversation_history: Sequence[ConversationTurn] | None = None,
    history_turns: int = 3,
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> str:
    messages = _build_messages(
        query=query,
        results=results,
        conversation_history=conversation_history,
        history_turns=history_turns,
    )
    if not stream:
        return llm.complete(messages)

    parts: list[str] = []
    for token in llm.complete_stream(messages):
        parts.append(token)
        if on_token:
            on_token(token)
    return "".join(parts).strip()
