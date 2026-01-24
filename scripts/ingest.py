"""CLI entrypoint for document ingestion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pymilvus import connections, utility

# Ensure project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MILVUS_URI,
)
from rag.embeddings import EmbeddingModel
from rag.ingest import ingest_documents
from rag.vector_store import VectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into Milvus.")
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Files or directories to ingest.",
    )
    parser.add_argument(
        "--milvus-uri",
        default=DEFAULT_MILVUS_URI,
        help="Milvus URI, e.g. data/milvus.db or http://localhost:19530",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Milvus collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size in characters.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop collection before ingesting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(path).expanduser() for path in args.paths]

    connections.connect(alias="default", uri=args.milvus_uri)
    if args.reset and utility.has_collection(args.collection):
        utility.drop_collection(args.collection)

    embedding_model = EmbeddingModel(args.embedding_model)
    vector_store = VectorStore(
        uri=args.milvus_uri,
        collection_name=args.collection,
        embedding_dim=embedding_model.dimension,
    )

    total_chunks = ingest_documents(
        paths=paths,
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    print(f"Ingested {total_chunks} chunks into {args.collection}.")


if __name__ == "__main__":
    main()
