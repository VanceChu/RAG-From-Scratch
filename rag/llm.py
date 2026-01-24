"""LLM client wrapper."""

from __future__ import annotations

from typing import Iterable, Optional

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
