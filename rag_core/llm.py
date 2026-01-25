"""LLM client wrapper."""

from __future__ import annotations

from typing import Iterator, Optional

from openai import OpenAI


class OpenAIChatLLM:
    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.2,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def complete_stream(
        self,
        messages: list[dict],
        temperature: float = 0.2,
    ) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if not content:
                continue
            if isinstance(content, str):
                yield content
                continue
            # Fallback for structured deltas.
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text") or ""
                        if text:
                            yield str(text)
