"""Minimal prompt helpers required by the slim PDF parser."""

from __future__ import annotations


def vision_llm_describe_prompt(page: int | None = None) -> str:
    page_hint = f" (page {page})" if page else ""
    return (
        "Describe the document page content clearly with key sections, tables, "
        f"and figures{page_hint}."
    )

