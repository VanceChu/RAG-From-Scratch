"""Embedding model wrapper."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

from openai import OpenAI
import requests

_OPENAI_EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_VOLC_EMBEDDING_DIMS = {
    "doubao-embedding": 2048,
    "doubao-embedding-and-m3": 2048,
    "doubao-embedding-large": 4096,
    "doubao-embedding-large-and-m3": 4096,
    "doubao-embedding-large-text-250515": 4096,
    "doubao-embedding-large-text-250715": 4096,
    "bge-m3": 1024,
    "bge-large-zh-and-m3": 1024,
}

_VOLC_DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def _normalize(embedding: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in embedding))
    if norm == 0:
        return embedding
    return [value / norm for value in embedding]


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    if batch_size <= 0:
        yield items
        return
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        provider: str = "sentence-transformers",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        endpoint_path: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.provider = provider.strip().lower()
        self.endpoint_path = (endpoint_path or "").strip().lstrip("/")
        self._dimension: int

        if self.provider in {"openai", "openai-embeddings", "openai-compatible", "volcengine", "ark"}:
            resolved_base_url = base_url
            if self.provider in {"volcengine", "ark"} and not resolved_base_url:
                resolved_base_url = _VOLC_DEFAULT_BASE_URL

            if self.provider in {"volcengine", "ark"} and self.endpoint_path:
                self._backend = "volcengine_multimodal"
                self.api_key = api_key
                self.base_url = resolved_base_url or _VOLC_DEFAULT_BASE_URL
            else:
                if resolved_base_url:
                    self.client = OpenAI(api_key=api_key, base_url=resolved_base_url)
                else:
                    self.client = OpenAI(api_key=api_key)
                self._backend = "openai"

            resolved_dim = embedding_dim or _OPENAI_EMBEDDING_DIMS.get(model_name)
            if not resolved_dim and self.provider in {"volcengine", "ark"}:
                resolved_dim = _VOLC_EMBEDDING_DIMS.get(model_name)
            if not resolved_dim:
                raise ValueError(
                    "Unknown embedding dimension for embedding model "
                    f"'{model_name}'. Set RAG_EMBEDDING_DIM or pass --embedding-dim."
                )
            self._dimension = int(resolved_dim)
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
            self._backend = "sentence-transformers"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Iterable[str], batch_size: int = 64) -> List[list[float]]:
        items = list(texts)
        if not items:
            return []
        if self._backend == "openai":
            return self._embed_openai(items, batch_size)
        if self._backend == "volcengine_multimodal":
            return self._embed_volcengine_multimodal(items, batch_size)

        embeddings = self.model.encode(
            items,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], batch_size=1)[0]

    def _embed_openai(self, texts: list[str], batch_size: int) -> List[list[float]]:
        embeddings: list[list[float]] = []
        for batch in _batched(texts, batch_size):
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            for item in sorted(response.data, key=lambda entry: entry.index):
                embeddings.append(_normalize(item.embedding))
        return embeddings

    def _embed_volcengine_multimodal(
        self, texts: list[str], batch_size: int
    ) -> List[list[float]]:
        embeddings: list[list[float]] = []
        base_url = (self.base_url or _VOLC_DEFAULT_BASE_URL).rstrip("/")
        endpoint = self.endpoint_path or "embeddings/multimodal"
        url = f"{base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for text in texts:
            payload = {
                "model": self.model_name,
                "input": [{"type": "text", "text": text}],
            }
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json().get("data")
            if isinstance(data, dict):
                embeddings.append(_normalize(data.get("embedding") or []))
                continue
            if isinstance(data, list):
                for item in data:
                    embeddings.append(_normalize(item.get("embedding") or []))
                continue
            raise ValueError("Unexpected response format from volcengine multimodal embeddings.")
        return embeddings
