"""Document ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rag_core.config import DEFAULT_IMAGE_DIR
from rag_core.embeddings import EmbeddingModel
from rag_core.ingest_state import IngestState
from rag_core.ragflow_pipeline import (
    SUPPORTED_EXTENSIONS,
    parse_and_split_document,
)
from rag_core.vector_store import VectorStore


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
) -> IngestSummary:
    summary = IngestSummary()
    collection_image_dir = DEFAULT_IMAGE_DIR / _safe_collection_name(
        vector_store.collection_name
    )

    for doc_path in _iter_documents(paths):
        source = _normalize_source(doc_path)
        doc_hash = state.compute_hash(doc_path)
        if state.is_unchanged(source, doc_hash, vector_store):
            summary.skipped_documents += 1
            continue

        chunks = parse_and_split_document(
            path=doc_path,
            chunk_token_size=chunk_size,
            overlap_tokens=overlap,
            image_dir=collection_image_dir,
        )

        existing = state.get(source)
        if existing is None:
            summary.new_documents += 1
        else:
            summary.refreshed_documents += 1
            summary.deleted_chunks += vector_store.delete_by_ids(existing.ids)

        if not chunks:
            state.mark_ingested(source, doc_hash, [])
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
            embeddings = embedding_model.embed(texts, batch_size=batch_size)
            inserted_ids = vector_store.insert(embeddings, texts, metadatas)
            doc_ids.extend(inserted_ids)
            summary.inserted_chunks += len(inserted_ids)

        state.mark_ingested(source, doc_hash, doc_ids)

    state.save()
    return summary
