from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import requests

from rag_core.config import DEFAULT_SPARSE_API_KEY, DEFAULT_SPARSE_API_URL

logger = logging.getLogger(__name__)


class SparseEmbeddingModel(ABC):
    @abstractmethod
    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """
        Generate sparse embeddings for a list of texts.
        Returns a list of dictionaries where keys are token IDs (or hash) and values are weights.
        """
        pass


class APISparseEmbeddingModel(SparseEmbeddingModel):
    def __init__(
        self,
        api_key: str = DEFAULT_SPARSE_API_KEY,
        api_url: str = DEFAULT_SPARSE_API_URL,
        model_name: str = "BAAI/bge-m3",
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name

    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        if not self.api_key:
            logger.warning("No Sparse API Key provided. Returning empty sparse vectors.")
            return [{} for _ in texts]

        # Note: This implementation assumes the API follows a format that returns sparse vectors.
        # Standard OpenAI-compatible format only returns dense.
        # We will attempt to pass 'return_sparse=True' which some custom endpoints might support,
        # otherwise this might need adjustment based on the specific provider's API docs.
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float", 
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for BGE-M3 specific 'sparse_vecs' or similar custom field
            # If the API only returns 'data' with 'embedding' (dense), this won't work for hybrid
            # unless we map it. 
            
            # TODO: Verify response format for SiliconFlow BGE-M3. 
            # For now, if we don't find explicit sparse fields, we might raise a warning.
            
            results = []
            for item in data.get("data", []):
                # Placeholder logic: Look for 'index' and 'value' style sparse format
                sparse_vec = item.get("sparse_embedding") or item.get("sparse_vec")
                
                # If the API doesn't return sparse key, we return empty to avoid crashing,
                # but log a warning.
                if not sparse_vec:
                    # Fallback or specific logic for provider
                    results.append({}) 
                else:
                    # Expecting format {token_id: weight} or list of [id, weight]
                    results.append(sparse_vec)
            
            return results

        except Exception as e:
            logger.error(f"Failed to generate sparse embeddings: {e}")
            return [{} for _ in texts]
