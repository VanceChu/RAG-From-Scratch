"""Milvus vector store wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


@dataclass
class SearchResult:
    text: str
    metadata: dict
    score: float


class VectorStore:
    def __init__(self, uri: str, collection_name: str, embedding_dim: int) -> None:
        self.uri = uri
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._collection: Optional[Collection] = None
        self._connect()
        self._ensure_collection()

    def _connect(self) -> None:
        connections.connect(alias="default", uri=self.uri)

    def _ensure_collection(self) -> None:
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            schema = collection.schema
            vector_field = next(
                field for field in schema.fields if field.name == "embedding"
            )
            if vector_field.params.get("dim") != self.embedding_dim:
                raise ValueError(
                    "Embedding dimension mismatch for existing collection: "
                    f"{vector_field.params.get('dim')} != {self.embedding_dim}"
                )
            collection.load()
            self._collection = collection
            return

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]
        schema = CollectionSchema(fields, description="RAG chunks")
        collection = Collection(self.collection_name, schema)
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 8, "efConstruction": 64},
            },
        )
        collection.load()
        self._collection = collection

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            raise RuntimeError("Collection is not initialized")
        return self._collection

    def insert(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        if not (len(embeddings) == len(texts) == len(metadatas)):
            raise ValueError("Embeddings, texts, and metadatas must be the same length")
        if not embeddings:
            return
        self.collection.insert([embeddings, texts, metadatas])
        self.collection.flush()

    def search(
        self,
        embedding: list[float],
        limit: int,
        expr: Optional[str] = None,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []
        search_params = {"metric_type": "IP", "params": {"ef": max(64, limit * 4)}}
        hits = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=["text", "metadata"],
        )
        results: list[SearchResult] = []
        for hit in hits[0]:
            entity = getattr(hit, "entity", None)
            text = (
                entity.get("text")
                if entity is not None
                else hit.get("text")  # type: ignore[call-arg]
            )
            metadata = (
                entity.get("metadata")
                if entity is not None
                else hit.get("metadata")  # type: ignore[call-arg]
            )
            results.append(
                SearchResult(text=text, metadata=metadata, score=float(hit.distance))
            )
        return results
