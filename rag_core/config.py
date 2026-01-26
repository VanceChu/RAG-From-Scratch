"""Configuration defaults and environment overrides."""

from __future__ import annotations

import hashlib
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


def _safe_identifier(value: str, max_length: int | None = None) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in value.strip())
    safe = safe.strip("_") or "default"
    if max_length and len(safe) > max_length:
        safe = safe[:max_length]
    return safe


def resolve_collection_name(
    base_collection: str,
    embedding_provider: str,
    embedding_model: str,
    raw: bool = False,
) -> str:
    if raw:
        return base_collection
    provider = (embedding_provider or "").strip().lower()
    model = (embedding_model or "").strip()
    default_provider = DEFAULT_EMBEDDING_PROVIDER.strip().lower()
    default_model = DEFAULT_EMBEDDING_MODEL.strip()
    if provider == default_provider and model == default_model:
        return base_collection
    provider_tag = _safe_identifier(provider or "provider", max_length=24)
    model_tag = _safe_identifier(model or "model", max_length=32)
    fingerprint = hashlib.sha1(
        f"{provider}::{model}".encode("utf-8", errors="ignore")
    ).hexdigest()[:8]
    base_tag = _safe_identifier(base_collection, max_length=96)
    return f"{base_tag}__{provider_tag}__{model_tag}_{fingerprint}"


DEFAULT_COLLECTION = _env_str("RAG_COLLECTION", "rag_chunks")
DEFAULT_MILVUS_URI = _env_str("MILVUS_URI", "data/index/milvus.db")
DEFAULT_INDEX_TYPE = _env_str("RAG_INDEX_TYPE", "HNSW")
DEFAULT_INDEX_NLIST = _env_int("RAG_INDEX_NLIST", 128)
DEFAULT_INDEX_M = _env_int("RAG_INDEX_M", 8)
DEFAULT_INDEX_EF_CONSTRUCTION = _env_int("RAG_INDEX_EF_CONSTRUCTION", 64)
DEFAULT_EMBEDDING_PROVIDER = _env_str("RAG_EMBEDDING_PROVIDER", "volcengine")
DEFAULT_EMBEDDING_MODEL = _env_str(
    "RAG_EMBEDDING_MODEL", "doubao-embedding-large-text-250515"
)
DEFAULT_EMBEDDING_API_KEY = _env_str("RAG_EMBEDDING_API_KEY", "")
DEFAULT_EMBEDDING_BASE_URL = _env_str("RAG_EMBEDDING_BASE_URL", "")
DEFAULT_EMBEDDING_ENDPOINT = _env_str("RAG_EMBEDDING_ENDPOINT", "")
DEFAULT_VOLC_API_KEY = _env_str("VOLC_API_KEY", "")
DEFAULT_VOLC_API_BASE_URL = _env_str(
    "VOLC_API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
)
DEFAULT_EMBEDDING_DIM = _env_int("RAG_EMBEDDING_DIM", 0)
DEFAULT_RERANK_MODEL = _env_str(
    "RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
DEFAULT_OPENAI_MODEL = _env_str("RAG_OPENAI_MODEL", "gpt-5.1")

DEFAULT_SPARSE_PROVIDER = _env_str("RAG_SPARSE_PROVIDER", "api")
DEFAULT_SPARSE_API_URL = _env_str("RAG_SPARSE_API_URL", "https://api.siliconflow.cn/v1/embeddings")
DEFAULT_SPARSE_API_KEY = _env_str("RAG_SPARSE_API_KEY", "")

DEFAULT_CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", 800)
DEFAULT_CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", 120)
DEFAULT_TOP_K = _env_int("RAG_TOP_K", 5)
DEFAULT_SEARCH_K = _env_int("RAG_SEARCH_K", 20)
DEFAULT_RERANK_TOP_K = _env_int("RAG_RERANK_TOP_K", 5)
DEFAULT_BATCH_SIZE = _env_int("RAG_BATCH_SIZE", 64)
DEFAULT_IMAGE_DIR = _env_path("RAG_IMAGE_DIR", "data/index/chunk_images")
DEFAULT_STATE_DIR = _env_path("RAG_STATE_DIR", "data/index/ingest_state")
DEFAULT_BM25_DIR = _env_path("RAG_BM25_DIR", "data/index/bm25")
