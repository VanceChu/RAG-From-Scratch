"""Configuration defaults and environment overrides."""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip("\"").strip("'")
        os.environ[key] = value


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def _env_path(name: str, default: str) -> Path:
    raw_value = os.getenv(name, default)
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


DEFAULT_COLLECTION = _env_str("RAG_COLLECTION", "rag_chunks")
DEFAULT_MILVUS_URI = _env_str("MILVUS_URI", "data/milvus.db")
DEFAULT_INDEX_TYPE = _env_str("RAG_INDEX_TYPE", "HNSW")
DEFAULT_INDEX_NLIST = _env_int("RAG_INDEX_NLIST", 128)
DEFAULT_INDEX_M = _env_int("RAG_INDEX_M", 8)
DEFAULT_INDEX_EF_CONSTRUCTION = _env_int("RAG_INDEX_EF_CONSTRUCTION", 64)
DEFAULT_EMBEDDING_PROVIDER = _env_str(
    "RAG_EMBEDDING_PROVIDER", "sentence-transformers"
)
DEFAULT_EMBEDDING_MODEL = _env_str(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_EMBEDDING_DIM = _env_int("RAG_EMBEDDING_DIM", 0)
DEFAULT_RERANK_MODEL = _env_str(
    "RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
DEFAULT_OPENAI_MODEL = _env_str("RAG_OPENAI_MODEL", "gpt-5.1")

DEFAULT_CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", 800)
DEFAULT_CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", 120)
DEFAULT_TOP_K = _env_int("RAG_TOP_K", 5)
DEFAULT_SEARCH_K = _env_int("RAG_SEARCH_K", 20)
DEFAULT_RERANK_TOP_K = _env_int("RAG_RERANK_TOP_K", 5)
DEFAULT_BATCH_SIZE = _env_int("RAG_BATCH_SIZE", 64)
DEFAULT_IMAGE_DIR = _env_path("RAG_IMAGE_DIR", "data/chunk_images")
DEFAULT_STATE_DIR = _env_path("RAG_STATE_DIR", "data/ingest_state")
