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
    id: int | None = None


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
        self._has_sparse_vector: bool = False
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
            # Check for sparse vector field existence
            sparse_field = next(
                (field for field in schema.fields if field.name == "sparse_vector"), None
            )
            self._has_sparse_vector = sparse_field is not None

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
                name="sparse_vector",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
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
        
        # Dense Index
        collection.create_index(
            field_name="embedding",
            index_params=self._build_index_params(),
        )
        
        # Sparse Index
        collection.create_index(
            field_name="sparse_vector",
            index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
        )

        collection.load()
        self._collection = collection
        self._has_sparse_vector = True

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
        sparse_vectors: list[dict[int, float]] | None = None,
    ) -> list[int]:
        if not (len(embeddings) == len(texts) == len(metadatas)):
            raise ValueError("Embeddings, texts, and metadatas must be the same length")

        if not embeddings:
            return []

        if self._has_sparse_vector:
            if sparse_vectors is None:
                # Sparse vectors cannot be null; empty dicts are allowed.
                sparse_vectors = [{} for _ in texts]
            if len(sparse_vectors) != len(texts):
                raise ValueError("sparse_vectors length must match texts length")
            mutation = self.collection.insert([embeddings, sparse_vectors, texts, metadatas])
        else:
            if sparse_vectors is not None:
                raise ValueError(
                    "sparse_vectors provided but collection has no sparse_vector field. "
                    "Recreate the collection with sparse support or omit sparse embeddings."
                )
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
        sparse_vector: dict[int, float] | None = None,
        hybrid_alpha: float = 0.5,
    ) -> list[SearchResult]:
        """
        Perform search. If sparse_vector is provided, performs Hybrid Search (Dense + Sparse).
        """
        if limit <= 0:
            return []

        # Pure Dense Search
        if not sparse_vector:
            search_params = self._build_search_params(limit)
            hits = self.collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=["text", "metadata"],
            )[0]
            
        else:
            if not self._has_sparse_vector:
                raise ValueError(
                    "Hybrid search requested but collection has no sparse_vector field. "
                    "Recreate the collection with sparse support."
                )
            # Hybrid Search
            from pymilvus import AnnSearchRequest, WeightedRanker
            
            # 1. Dense Request
            search_param_dense = self._build_search_params(limit)
            req_dense = AnnSearchRequest(
                data=[embedding],
                anns_field="embedding",
                param=search_param_dense,
                limit=limit,
                expr=expr
            )
            
            # 2. Sparse Request
            req_sparse = AnnSearchRequest(
                data=[sparse_vector],
                anns_field="sparse_vector",
                param={"metric_type": "IP", "params": {}}, # Sparse typically uses IP
                limit=limit,
                expr=expr
            )
            
            # 3. Hybrid Search
            # Adjust weights based on hybrid_alpha (0.0 = pure sparse, 1.0 = pure dense)
            # Actually WeightedRanker takes simple weights.
            # Let's say weight_dense = alpha, weight_sparse = 1 - alpha
            w_dense = hybrid_alpha
            w_sparse = 1.0 - hybrid_alpha
            ranker = WeightedRanker(w_dense, w_sparse)
            
            hits = self.collection.hybrid_search(
                reqs=[req_dense, req_sparse],
                rerank=ranker,
                limit=limit,
                output_fields=["text", "metadata"]
            )[0]

        results: list[SearchResult] = []
        for hit in hits:
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
            hit_id = getattr(hit, "id", None)
            if hit_id is None:
                hit_id = getattr(hit, "pk", None)
            if hit_id is None and entity is not None:
                hit_id = entity.get("id")
            try:
                hit_id = int(hit_id) if hit_id is not None else None
            except (TypeError, ValueError):
                hit_id = None
            results.append(
                SearchResult(
                    text=text,
                    metadata=metadata,
                    score=float(hit.distance),
                    id=hit_id,
                )
            )
        return results
