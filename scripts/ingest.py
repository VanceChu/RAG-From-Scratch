"""CLI entrypoint for document ingestion."""

from __future__ import annotations

import argparse
import shlex
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
    DEFAULT_COLLECTION_RAW,
    DEFAULT_IMAGE_DIR,
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_ENDPOINT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_VOLC_API_BASE_URL,
    DEFAULT_VOLC_API_KEY,
    DEFAULT_INDEX_EF_CONSTRUCTION,
    DEFAULT_INDEX_M,
    DEFAULT_INDEX_NLIST,
    DEFAULT_INDEX_TYPE,
    DEFAULT_ENABLE_BM25,
    DEFAULT_ENABLE_SPARSE,
    DEFAULT_BM25_DIR,
    DEFAULT_MILVUS_URI,
    DEFAULT_RESET,
    DEFAULT_STATE_DIR,
    resolve_collection_name,
)
from rag_core.embeddings import EmbeddingModel
from rag_core.ingest import ingest_documents
from rag_core.ingest_state import IngestState
from rag_core.ragflow_pipeline import SUPPORTED_EXTENSIONS
from rag_core.vector_store import VectorStore


def _ensure_local_milvus_parent(uri: str) -> None:
    if "://" in uri:
        return
    path = Path(uri).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_collection_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    return safe or "collection"


def _collect_documents(paths: list[Path]) -> list[Path]:
    documents: list[Path] = []
    for path in paths:
        if path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    documents.append(file_path)
        else:
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                documents.append(path)
    return documents


def _milvus_target(uri: str) -> str:
    if "://" in uri:
        return uri
    return str(Path(uri).expanduser().resolve())


def _print_ingest_overview(
    collection_name: str,
    milvus_uri: str,
    embedding_provider: str,
    embedding_model_name: str,
    embedding_dim: int,
    embedding_api_key_set: bool,
    embedding_base_url: str | None,
    embedding_endpoint: str | None,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    index_type: str,
    index_params: dict,
    enable_sparse: bool,
    enable_bm25: bool,
    input_paths: list[Path],
    documents: list[Path],
    state_path: Path,
    chunk_images_dir: Path,
    bm25_path: Path | None,
) -> None:
    print("Ingest configuration:")
    print(f"- collection: {collection_name}")
    print(f"- milvus_uri: {_milvus_target(milvus_uri)}")
    print(f"- embedding_provider: {embedding_provider}")
    print(f"- embedding_model: {embedding_model_name}")
    print(f"- embedding_dim: {embedding_dim}")
    print(f"- embedding_api_key: {'set' if embedding_api_key_set else 'not set'}")
    if embedding_base_url:
        print(f"- embedding_base_url: {embedding_base_url}")
    if embedding_endpoint:
        print(f"- embedding_endpoint: {embedding_endpoint}")
    print(f"- index_type: {index_type}")
    print(f"- index_params: {index_params}")
    print(f"- chunk_size: {chunk_size}")
    print(f"- overlap: {overlap}")
    print(f"- batch_size: {batch_size}")
    print(f"- enable_sparse: {enable_sparse}")
    print(f"- enable_bm25: {enable_bm25}")

    print("\nInputs:")
    print(f"- paths: {', '.join(str(path) for path in input_paths)}")
    print(f"- resolved_documents: {len(documents)}")
    preview_limit = 10
    for doc_path in documents[:preview_limit]:
        print(f"  {doc_path}")
    if len(documents) > preview_limit:
        print(f"  ... +{len(documents) - preview_limit} more")

    print("\nOutputs:")
    print(f"- milvus_collection: {collection_name}")
    print(f"- milvus_storage: {_milvus_target(milvus_uri)}")
    print(f"- ingest_state_file: {state_path}")
    print(f"- chunk_images_dir: {chunk_images_dir}")
    if bm25_path:
        print(f"- bm25_index_file: {bm25_path}")
    print()


def _print_ingest_timing(timing: dict) -> None:
    if not timing:
        return
    totals = timing.get("totals", {})
    docs = timing.get("docs", [])
    print("Ingest timing:")
    print(f"parse_split_s: {totals.get('parse_split_s', 0.0):.3f}")
    print(f"embed_s: {totals.get('embed_s', 0.0):.3f}")
    print(f"insert_s: {totals.get('insert_s', 0.0):.3f}")
    print(f"bm25_s: {totals.get('bm25_s', 0.0):.3f}")
    print(f"total_s: {totals.get('total_s', 0.0):.3f}")
    if docs:
        print("\nPer-document timing:")
        for doc in docs:
            print(f"doc: {doc.get('source')}")
            print(f"skipped: {doc.get('skipped')}")
            print(f"chunks: {doc.get('chunks')}")
            print(f"inserted_chunks: {doc.get('inserted_chunks')}")
            print(f"parse_split_s: {doc.get('parse_split_s', 0.0):.3f}")
            print(f"embed_s: {doc.get('embed_s', 0.0):.3f}")
            print(f"insert_s: {doc.get('insert_s', 0.0):.3f}")
            print(f"bm25_s: {doc.get('bm25_s', 0.0):.3f}")
            print(f"total_s: {doc.get('total_s', 0.0):.3f}")
    print()


def _prompt_text(label: str, default: str | None = None, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default not in (None, ""):
            return str(default)
        if not required:
            return ""
        print("A value is required.")


def _prompt_bool(label: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        value = input(f"{label} ({hint}): ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _prompt_int(label: str, default: int) -> int:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            print("Please enter a valid integer.")


def _prompt_choice(label: str, choices: list[str], default: str | None) -> str:
    normalized = {choice.lower(): choice for choice in choices}
    default_value = (default or "").strip()
    if default_value:
        default_key = default_value.lower()
        if default_key == "auto":
            default_key = "autoindex"
        default_value = normalized.get(default_key, default_value)
    choices_label = "/".join(choices)
    while True:
        suffix = f" [{default_value}]" if default_value else ""
        value = input(f"{label} ({choices_label}){suffix}: ").strip()
        if not value and default_value:
            return default_value
        if not value:
            print("Please choose a value.")
            continue
        key = value.lower()
        if key == "auto":
            key = "autoindex"
        choice = normalized.get(key)
        if choice:
            return choice
        print(f"Please choose from: {choices_label}")


def _prompt_paths(label: str, default_paths: list[str] | None) -> list[Path]:
    default_value = " ".join(str(path) for path in default_paths) if default_paths else ""
    while True:
        prompt = f"{label} [{default_value}]: " if default_value else f"{label}: "
        raw = input(prompt).strip()
        if not raw:
            if default_paths:
                return [Path(path).expanduser() for path in default_paths]
            print("Please provide at least one path.")
            continue
        raw = raw.replace(",", " ")
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            print(f"Invalid path list: {exc}")
            continue
        if not parts:
            print("Please provide at least one path.")
            continue
        return [Path(path).expanduser() for path in parts]


def _run_wizard(args: argparse.Namespace) -> argparse.Namespace:
    print("Ingest wizard (press Enter to accept defaults).")
    args.paths = [str(path) for path in _prompt_paths("Paths to ingest", args.paths)]
    args.reset = _prompt_bool("Reset collection before ingest", args.reset)
    args.enable_bm25 = _prompt_bool("Enable BM25 index", args.enable_bm25)
    args.enable_sparse = _prompt_bool("Enable sparse embeddings", args.enable_sparse)
    if args.enable_bm25 and args.enable_sparse:
        print("Note: BM25 takes precedence over sparse embeddings during retrieval.")

    args.collection = _prompt_text("Collection name", args.collection)
    args.collection_raw = _prompt_bool(
        "Use collection name as-is (no suffix)",
        args.collection_raw,
    )
    args.milvus_uri = _prompt_text("Milvus URI", args.milvus_uri)

    provider_choices = [
        "volcengine",
        "openai",
        "openai-compatible",
        "openai-embeddings",
        "sentence-transformers",
        "ark",
    ]
    args.embedding_provider = _prompt_choice(
        "Embedding provider",
        provider_choices,
        args.embedding_provider,
    )
    args.embedding_model = _prompt_text("Embedding model", args.embedding_model)
    if args.embedding_provider in {
        "openai",
        "openai-compatible",
        "openai-embeddings",
        "volcengine",
        "ark",
    }:
        args.embedding_base_url = _prompt_text(
            "Embedding base URL (blank for default)",
            args.embedding_base_url,
        )
        args.embedding_endpoint = _prompt_text(
            "Embedding endpoint (blank for default)",
            args.embedding_endpoint,
        )
        args.embedding_dim = _prompt_int("Embedding dimension (0=auto)", args.embedding_dim)

    if _prompt_bool("Configure advanced settings", False):
        args.chunk_size = _prompt_int("Chunk size (tokens)", args.chunk_size)
        args.overlap = _prompt_int("Chunk overlap (tokens)", args.overlap)
        args.batch_size = _prompt_int("Embedding batch size", args.batch_size)
        args.index_type = _prompt_choice(
            "Index type",
            ["HNSW", "IVF_FLAT", "FLAT", "AUTOINDEX"],
            args.index_type,
        )
        if args.index_type == "IVF_FLAT":
            args.index_nlist = _prompt_int("IVF nlist", args.index_nlist)
        elif args.index_type == "HNSW":
            args.index_m = _prompt_int("HNSW M", args.index_m)
            args.index_ef_construction = _prompt_int(
                "HNSW efConstruction",
                args.index_ef_construction,
            )
    print()
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into Milvus.")
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Run an interactive setup wizard.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
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
        default=DEFAULT_COLLECTION_RAW,
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
        help="Embedding provider: sentence-transformers, openai, or volcengine.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name (SentenceTransformers or OpenAI).",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=DEFAULT_EMBEDDING_BASE_URL,
        help="Embedding API base URL for OpenAI-compatible providers.",
    )
    parser.add_argument(
        "--embedding-endpoint",
        default=DEFAULT_EMBEDDING_ENDPOINT,
        help="Embedding endpoint path (e.g. embeddings/multimodal).",
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
        "--enable-sparse",
        action="store_true",
        default=DEFAULT_ENABLE_SPARSE,
        help="Enable sparse vector generation using API.",
    )
    parser.add_argument(
        "--enable-bm25",
        action="store_true",
        default=DEFAULT_ENABLE_BM25,
        help="Enable BM25 lexical index for hybrid retrieval.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=DEFAULT_RESET,
        help="Drop collection before ingesting.",
    )
    args = parser.parse_args()
    if not args.wizard and not args.paths:
        parser.error("--paths is required unless --wizard is used.")
    return args


def main() -> None:
    args = parse_args()
    if args.wizard:
        args = _run_wizard(args)
    paths = [Path(path).expanduser() for path in args.paths]
    documents = _collect_documents(paths)
    collection_name = resolve_collection_name(
        base_collection=args.collection,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        raw=args.collection_raw,
    )

    _ensure_local_milvus_parent(args.milvus_uri)
    connections.connect(alias="default", uri=args.milvus_uri)
    if args.reset and utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    embedding_dim = args.embedding_dim if args.embedding_dim and args.embedding_dim > 0 else None
    embedding_api_key = DEFAULT_EMBEDDING_API_KEY or None
    embedding_base_url = args.embedding_base_url or None
    embedding_endpoint = args.embedding_endpoint or None
    if args.embedding_provider in {"volcengine", "ark"}:
        if not embedding_api_key:
            embedding_api_key = DEFAULT_VOLC_API_KEY or None
        if not embedding_base_url:
            embedding_base_url = DEFAULT_VOLC_API_BASE_URL or None
    embedding_model = EmbeddingModel(
        args.embedding_model,
        provider=args.embedding_provider,
        api_key=embedding_api_key,
        base_url=embedding_base_url,
        endpoint_path=embedding_endpoint,
        embedding_dim=embedding_dim,
    )
    
    sparse_model = None
    if args.enable_sparse:
        from rag_core.sparse_embedding import APISparseEmbeddingModel
        # Rely on env vars for key/url defaults
        sparse_model = APISparseEmbeddingModel()
    
    bm25_index = None
    if args.enable_bm25:
        from rag_core.bm25_index import BM25Index
        bm25_index = BM25Index.load(collection=collection_name, base_dir=DEFAULT_BM25_DIR)
        if args.reset:
            bm25_index.clear()

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

    chunk_images_dir = DEFAULT_IMAGE_DIR / _safe_collection_name(collection_name)
    bm25_path = None
    if args.enable_bm25:
        bm25_path = DEFAULT_BM25_DIR / f"{_safe_collection_name(collection_name)}.json"

    _print_ingest_overview(
        collection_name=collection_name,
        milvus_uri=args.milvus_uri,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
        embedding_dim=embedding_model.dimension,
        embedding_api_key_set=bool(embedding_api_key),
        embedding_base_url=embedding_base_url,
        embedding_endpoint=embedding_endpoint,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        index_type=args.index_type,
        index_params=index_params,
        enable_sparse=args.enable_sparse,
        enable_bm25=args.enable_bm25,
        input_paths=paths,
        documents=documents,
        state_path=state.path,
        chunk_images_dir=chunk_images_dir,
        bm25_path=bm25_path,
    )

    timing: dict = {}
    summary = ingest_documents(
        paths=paths,
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        state=state,
        sparse_embedding_model=sparse_model,
        bm25_index=bm25_index,
        timing=timing,
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
    _print_ingest_timing(timing)


if __name__ == "__main__":
    main()
