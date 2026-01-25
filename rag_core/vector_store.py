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
    def __init__(
        self,
        uri: str,
        collection_name: str,
        embedding_dim: int,
        index_type: str = "HNSW",
        index_params: Optional[dict] = None,
        metric_type: str = "IP",
    ) -> None:
        self.uri = uri
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.index_type = self._normalize_index_type(index_type)
        self.index_params = index_params or {}
        self.metric_type = metric_type.upper()
        self._collection: Optional[Collection] = None
        self._connect()
        self._ensure_collection()

    @staticmethod
    def _normalize_index_type(index_type: str) -> str:
        normalized = index_type.strip().upper()
        if normalized == "AUTO":
            normalized = "AUTOINDEX"
        supported = {"HNSW", "IVF_FLAT", "FLAT", "AUTOINDEX"}
        if normalized not in supported:
            raise ValueError(f"Unsupported index type: {index_type}")
        return normalized

    def _build_index_params(self) -> dict:
        params: dict = {}
        if self.index_type == "HNSW":
            params = {
                "M": int(self.index_params.get("M", 8)),
                "efConstruction": int(self.index_params.get("efConstruction", 64)),
            }
        elif self.index_type == "IVF_FLAT":
            params = {"nlist": int(self.index_params.get("nlist", 128))}
        elif self.index_type in {"FLAT", "AUTOINDEX"}:
            params = {}
        return {
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "params": params,
        }

    def _build_search_params(self, limit: int) -> dict:
        params: dict = {}
        if self.index_type == "HNSW":
            params["ef"] = max(64, limit * 4)
        elif self.index_type == "IVF_FLAT":
            params["nprobe"] = max(8, limit * 2)
        return {"metric_type": self.metric_type, "params": params}

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
            if collection.indexes:
                index_params = collection.indexes[0].params or {}
                detected_type = (
                    index_params.get("index_type")
                    or index_params.get("indexType")
                    or getattr(collection.indexes[0], "index_type", None)
                )
                if detected_type:
                    self.index_type = self._normalize_index_type(str(detected_type))
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
            index_params=self._build_index_params(),
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
    ) -> list[int]:
        if not (len(embeddings) == len(texts) == len(metadatas)):
            raise ValueError("Embeddings, texts, and metadatas must be the same length")
        if not embeddings:
            return []
        mutation = self.collection.insert([embeddings, texts, metadatas])
        self.collection.flush()
        primary_keys = getattr(mutation, "primary_keys", None) or []
        return [int(key) for key in primary_keys]

    def delete_by_ids(self, ids: Iterable[int], batch_size: int = 512) -> int:
        id_list = [int(value) for value in ids]
        if not id_list:
            return 0
        total_deleted = 0
        for start in range(0, len(id_list), batch_size):
            batch = id_list[start : start + batch_size]
            expr = f"id in [{','.join(str(value) for value in batch)}]"
            result = self.collection.delete(expr)
            total_deleted += int(getattr(result, "delete_count", 0) or 0)
        self.collection.flush()
        return total_deleted

    def ids_exist(self, ids: Iterable[int]) -> bool:
        id_list = [int(value) for value in ids]
        if not id_list:
            return False
        expr = f"id in [{','.join(str(value) for value in id_list)}]"
        try:
            rows = self.collection.query(expr=expr, output_fields=["id"], limit=1)
        except TypeError:
            rows = self.collection.query(expr=expr, output_fields=["id"])
        return bool(rows)

    def search(
        self,
        embedding: list[float],
        limit: int,
        expr: Optional[str] = None,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []
        search_params = self._build_search_params(limit)
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
