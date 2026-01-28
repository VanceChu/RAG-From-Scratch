"""LLM client wrapper."""

from __future__ import annotations

import time
from typing import Iterator, Optional

from openai import OpenAI

from rag_core.observability import TraceContext, log_generation


class OpenAIChatLLM:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        trace_ctx: TraceContext | None = None,
    ) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.trace_ctx = trace_ctx

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.2,
    ) -> str:
        start_time = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        output = content.strip() if content else ""
        usage = None
        try:
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
        except Exception:
            usage = None
        log_generation(
            ctx=self.trace_ctx,
            name="llm_complete",
            model=self.model,
            input_messages=messages,
            output=output,
            usage=usage,
            metadata={"latency_s": time.perf_counter() - start_time},
        )
        return output

    def complete_stream(
        self,
        messages: list[dict],
        temperature: float = 0.2,
    ) -> Iterator[str]:
        start_time = time.perf_counter()
        parts: list[str] = []
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if not content:
                    continue
                if isinstance(content, str):
                    parts.append(content)
                    yield content
                    continue
                # Fallback for structured deltas.
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text") or ""
                            if text:
                                parts.append(str(text))
                                yield str(text)
        finally:
            output = "".join(parts).strip()
            log_generation(
                ctx=self.trace_ctx,
                name="llm_stream",
                model=self.model,
                input_messages=messages,
                output=output,
                usage=None,
                metadata={"latency_s": time.perf_counter() - start_time},
            )
