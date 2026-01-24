"""Document ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rag.chunking import chunk_markdown
from rag.embeddings import EmbeddingModel
from rag.parsers import parse_document
from rag.vector_store import VectorStore


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}


def _iter_documents(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield file_path
        else:
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path


def ingest_documents(
    paths: Iterable[Path],
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    chunk_size: int,
    overlap: int,
    batch_size: int,
) -> int:
    total_chunks = 0
    texts: list[str] = []
    metadatas: list[dict] = []

    for doc_path in _iter_documents(paths):
        markdown, doc_metadata = parse_document(doc_path)
        for chunk in chunk_markdown(
            markdown=markdown,
            source=doc_path,
            chunk_size=chunk_size,
            overlap=overlap,
        ):
            metadata = {
                **chunk.metadata,
                "doc_metadata": doc_metadata,
            }
            texts.append(chunk.text)
            metadatas.append(metadata)
            if len(texts) >= batch_size:
                embeddings = embedding_model.embed(texts, batch_size=batch_size)
                vector_store.insert(embeddings, texts, metadatas)
                total_chunks += len(texts)
                texts.clear()
                metadatas.clear()

    if texts:
        embeddings = embedding_model.embed(texts, batch_size=batch_size)
        vector_store.insert(embeddings, texts, metadatas)
        total_chunks += len(texts)

    return total_chunks
