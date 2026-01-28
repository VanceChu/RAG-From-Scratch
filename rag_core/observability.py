"""LangFuse observability utilities with graceful fallback."""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from rag_core.config import (
    DEFAULT_LANGFUSE_ENABLED,
    DEFAULT_LANGFUSE_HOST,
    DEFAULT_LANGFUSE_PUBLIC_KEY,
    DEFAULT_LANGFUSE_SECRET_KEY,
)

_langfuse_client: Any | None = None


@dataclass
class TraceContext:
    """Context for a single trace."""

    trace_id: str | None = None
    client: Any | None = None
    trace: Any | None = None
    span: Any | None = None
    span_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_span(self, span: Any | None, span_id: str | None = None) -> "TraceContext":
        return TraceContext(
            trace_id=self.trace_id,
            client=self.client,
            trace=self.trace,
            span=span,
            span_id=span_id,
            metadata=self.metadata,
        )

    def as_trace_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {}
        if self.trace_id:
            context["trace_id"] = self.trace_id
        if self.span_id:
            context["parent_span_id"] = self.span_id
        return context


def _load_langfuse() -> Any | None:
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    if not DEFAULT_LANGFUSE_ENABLED:
        return None
    if not (DEFAULT_LANGFUSE_PUBLIC_KEY and DEFAULT_LANGFUSE_SECRET_KEY):
        return None
    try:
        from langfuse import Langfuse  # type: ignore
    except Exception:
        return None
    try:
        _langfuse_client = Langfuse(
            public_key=DEFAULT_LANGFUSE_PUBLIC_KEY,
            secret_key=DEFAULT_LANGFUSE_SECRET_KEY,
            host=DEFAULT_LANGFUSE_HOST,
        )
    except Exception:
        _langfuse_client = None
    return _langfuse_client


def is_enabled() -> bool:
    return _load_langfuse() is not None


def _safe_call(target: Any, method: str, **kwargs: Any) -> Any | None:
    if target is None:
        return None
    fn = getattr(target, method, None)
    if not callable(fn):
        return None
    try:
        return fn(**kwargs)
    except Exception:
        return None


@contextmanager
def trace(
    name: str,
    user_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> Generator[TraceContext, None, None]:
    client = _load_langfuse()
    if not client:
        yield TraceContext(trace_id=None)
        return

    if not trace_id:
        trace_id = _safe_call(client, "create_trace_id")
    if not trace_id:
        trace_id = uuid.uuid4().hex

    trace_obj = _safe_call(
        client,
        "trace",
        name=name,
        id=trace_id,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata,
    )
    ctx = TraceContext(
        trace_id=trace_id,
        client=client,
        trace=trace_obj,
        metadata=metadata or {},
    )
    try:
        yield ctx
    finally:
        _safe_call(trace_obj, "end")
        _safe_call(trace_obj, "flush")


@contextmanager
def span(
    ctx: TraceContext | None,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Generator[TraceContext, None, None]:
    if ctx is None or not is_enabled():
        yield ctx if ctx is not None else TraceContext(trace_id=None)
        return

    parent = ctx.span or ctx.trace
    span_obj = None
    if parent is not None:
        span_obj = _safe_call(parent, "span", name=name, metadata=metadata)
        if span_obj is None:
            span_obj = _safe_call(parent, "start_span", name=name, metadata=metadata)
    if span_obj is None:
        span_obj = _safe_call(
            ctx.client,
            "span",
            trace_id=ctx.trace_id,
            name=name,
            metadata=metadata,
        )

    span_id = getattr(span_obj, "id", None) or getattr(span_obj, "span_id", None)
    next_ctx = ctx.with_span(span_obj, span_id=span_id)
    try:
        yield next_ctx
    finally:
        _safe_call(span_obj, "end")
        _safe_call(span_obj, "finish")


def log_generation(
    ctx: TraceContext | None,
    name: str,
    model: str,
    input_messages: list[dict],
    output: str,
    usage: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if ctx is None or not is_enabled():
        return
    client = ctx.client or _load_langfuse()
    if not client:
        return
    generation = _safe_call(
        client,
        "start_generation",
        trace_context=ctx.as_trace_context(),
        name=name,
        model=model,
        input={"messages": input_messages},
        metadata=metadata,
    )
    if generation is None:
        generation = _safe_call(
            ctx.trace,
            "generation",
            name=name,
            model=model,
            input={"messages": input_messages},
            metadata=metadata,
        )
    if generation is None:
        return
    update_payload = {"output": output}
    if usage:
        _safe_call(generation, "update", output=update_payload, usage_details=usage)
    else:
        _safe_call(generation, "update", output=update_payload)
    _safe_call(generation, "end")


def log_score(
    ctx: TraceContext | None,
    name: str,
    value: float,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if ctx is None or not is_enabled():
        return
    client = ctx.client or _load_langfuse()
    if not client:
        return
    _safe_call(
        client,
        "score",
        trace_id=ctx.trace_id,
        name=name,
        value=value,
        comment=comment,
        metadata=metadata,
    )


def update_trace_metadata(ctx: TraceContext | None, metadata: dict[str, Any]) -> None:
    if ctx is None or not is_enabled() or not metadata:
        return
    target = ctx.trace
    if target is None:
        return
    _safe_call(target, "update", metadata=metadata)


def log_event(
    ctx: TraceContext | None,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if ctx is None or not is_enabled():
        return
    client = ctx.client or _load_langfuse()
    if not client:
        return
    _safe_call(
        client,
        "event",
        trace_id=ctx.trace_id,
        name=name,
        metadata=metadata,
    )
