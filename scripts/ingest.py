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

from rag_core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_INDEX_EF_CONSTRUCTION,
    DEFAULT_INDEX_M,
    DEFAULT_INDEX_NLIST,
    DEFAULT_INDEX_TYPE,
    DEFAULT_MILVUS_URI,
    DEFAULT_STATE_DIR,
    resolve_collection_name,
)
from rag_core.embeddings import EmbeddingModel
from rag_core.ingest import ingest_documents
from rag_core.ingest_state import IngestState
from rag_core.vector_store import VectorStore


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
        "--collection-raw",
        action="store_true",
        help="Use the collection name as-is (disable model-based suffix).",
    )
    parser.add_argument(
        "--index-type",
        default=DEFAULT_INDEX_TYPE,
        help="Milvus index type: HNSW, IVF_FLAT, FLAT, AUTOINDEX.",
    )
    parser.add_argument(
        "--index-nlist",
        type=int,
        default=DEFAULT_INDEX_NLIST,
        help="IVF_FLAT nlist parameter.",
    )
    parser.add_argument(
        "--index-m",
        type=int,
        default=DEFAULT_INDEX_M,
        help="HNSW M parameter.",
    )
    parser.add_argument(
        "--index-ef-construction",
        type=int,
        default=DEFAULT_INDEX_EF_CONSTRUCTION,
        help="HNSW efConstruction parameter.",
    )
    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Embedding provider: sentence-transformers or openai.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name (SentenceTransformers or OpenAI).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="Embedding dimension (required for unknown OpenAI models).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size in tokens.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap in tokens.",
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
    collection_name = resolve_collection_name(
        base_collection=args.collection,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        raw=args.collection_raw,
    )

    connections.connect(alias="default", uri=args.milvus_uri)
    if args.reset and utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    embedding_dim = args.embedding_dim if args.embedding_dim and args.embedding_dim > 0 else None
    embedding_model = EmbeddingModel(
        args.embedding_model,
        provider=args.embedding_provider,
        embedding_dim=embedding_dim,
    )
    index_params = {
        "nlist": args.index_nlist,
        "M": args.index_m,
        "efConstruction": args.index_ef_construction,
    }
    vector_store = VectorStore(
        uri=args.milvus_uri,
        collection_name=collection_name,
        embedding_dim=embedding_model.dimension,
        index_type=args.index_type,
        index_params=index_params,
    )

    state = IngestState.load(
        uri=args.milvus_uri,
        collection=collection_name,
        state_dir=DEFAULT_STATE_DIR,
    )
    if args.reset:
        state.clear()

    summary = ingest_documents(
        paths=paths,
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        state=state,
    )
    print(f"Collection: {collection_name}")
    print(
        "Ingest summary: "
        f"new_docs={summary.new_documents} "
        f"refreshed_docs={summary.refreshed_documents} "
        f"skipped_docs={summary.skipped_documents} "
        f"inserted_chunks={summary.inserted_chunks} "
        f"deleted_chunks={summary.deleted_chunks}"
    )
    print(f"State file: {state.path}")


if __name__ == "__main__":
    main()
