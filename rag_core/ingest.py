"""Document ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import time

from rag_core.config import DEFAULT_IMAGE_DIR
from rag_core.embeddings import EmbeddingModel
from rag_core.ingest_state import IngestState
from rag_core.ragflow_pipeline import (
    SUPPORTED_EXTENSIONS,
    parse_and_split_document,
)
from rag_core.vector_store import VectorStore
from rag_core.bm25_index import BM25Index
from rag_core.sparse_embedding import SparseEmbeddingModel


def _iter_documents(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield file_path
        else:
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path


def _normalize_source(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _safe_collection_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    return safe or "collection"


@dataclass
class IngestSummary:
    inserted_chunks: int = 0
    deleted_chunks: int = 0
    skipped_documents: int = 0
    refreshed_documents: int = 0
    new_documents: int = 0
def ingest_documents(
    paths: Iterable[Path],
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    state: IngestState,
    sparse_embedding_model: SparseEmbeddingModel | None = None,
    bm25_index: BM25Index | None = None,
    timing: dict | None = None,
) -> IngestSummary:
    summary = IngestSummary()
    collection_image_dir = DEFAULT_IMAGE_DIR / _safe_collection_name(
        vector_store.collection_name
    )
    if timing is not None:
        timing.clear()
        timing["docs"] = []
        timing["totals"] = {
            "parse_split_s": 0.0,
            "embed_s": 0.0,
            "insert_s": 0.0,
            "bm25_s": 0.0,
            "total_s": 0.0,
        }

    for doc_path in _iter_documents(paths):
        doc_start = time.perf_counter()
        source = _normalize_source(doc_path)
        doc_hash = state.compute_hash(doc_path)
        doc_timing = {
            "source": source,
            "chunks": 0,
            "inserted_chunks": 0,
            "skipped": False,
            "parse_split_s": 0.0,
            "embed_s": 0.0,
            "insert_s": 0.0,
            "bm25_s": 0.0,
            "total_s": 0.0,
        }
        if state.is_unchanged(source, doc_hash, vector_store):
            summary.skipped_documents += 1
            doc_timing["skipped"] = True
            doc_timing["total_s"] = time.perf_counter() - doc_start
            if timing is not None:
                timing["docs"].append(doc_timing)
                timing["totals"]["total_s"] += doc_timing["total_s"]
            continue

        parse_start = time.perf_counter()
        chunks = parse_and_split_document(
            path=doc_path,
            chunk_token_size=chunk_size,
            overlap_tokens=overlap,
            image_dir=collection_image_dir,
        )
        parse_elapsed = time.perf_counter() - parse_start
        doc_timing["parse_split_s"] = parse_elapsed
        doc_timing["chunks"] = len(chunks)
        if timing is not None:
            timing["totals"]["parse_split_s"] += parse_elapsed

        existing = state.get(source)
        if existing is None:
            summary.new_documents += 1
        else:
            summary.refreshed_documents += 1
            summary.deleted_chunks += vector_store.delete_by_ids(existing.ids)
            if bm25_index:
                bm25_index.remove(existing.ids)

        if not chunks:
            state.mark_ingested(source, doc_hash, [])
            doc_timing["total_s"] = time.perf_counter() - doc_start
            if timing is not None:
                timing["docs"].append(doc_timing)
                timing["totals"]["total_s"] += doc_timing["total_s"]
            continue

        doc_ids: list[int] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk["text"] for chunk in batch]
            metadatas = []
            for chunk in batch:
                metadata = dict(chunk["metadata"])
                metadata["source"] = source
                metadata["doc_hash"] = doc_hash
                metadatas.append(metadata)

            embed_start = time.perf_counter()
            embeddings = embedding_model.embed(texts, batch_size=batch_size)
            embed_elapsed = time.perf_counter() - embed_start
            doc_timing["embed_s"] += embed_elapsed
            if timing is not None:
                timing["totals"]["embed_s"] += embed_elapsed
            
            sparse_vectors = None
            if sparse_embedding_model:
                sparse_vectors = sparse_embedding_model.embed_sparse(texts)

            insert_start = time.perf_counter()
            inserted_ids = vector_store.insert(embeddings, texts, metadatas, sparse_vectors=sparse_vectors)
            insert_elapsed = time.perf_counter() - insert_start
            doc_timing["insert_s"] += insert_elapsed
            if timing is not None:
                timing["totals"]["insert_s"] += insert_elapsed
            doc_ids.extend(inserted_ids)
            summary.inserted_chunks += len(inserted_ids)
            if bm25_index:
                bm25_start = time.perf_counter()
                bm25_index.add(inserted_ids, texts, metadatas)
                bm25_elapsed = time.perf_counter() - bm25_start
                doc_timing["bm25_s"] += bm25_elapsed
                if timing is not None:
                    timing["totals"]["bm25_s"] += bm25_elapsed

        state.mark_ingested(source, doc_hash, doc_ids)
        doc_timing["inserted_chunks"] = len(doc_ids)
        doc_timing["total_s"] = time.perf_counter() - doc_start
        if timing is not None:
            timing["docs"].append(doc_timing)
            timing["totals"]["total_s"] += doc_timing["total_s"]

    state.save()
    if bm25_index:
        bm25_index.save()
    return summary
