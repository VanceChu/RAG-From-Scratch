"""Persistent ingestion state for incremental updates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from rag_core.config import DEFAULT_STATE_DIR


_STATE_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024


def _uri_fingerprint(uri: str) -> str:
    digest = hashlib.sha1(uri.encode("utf-8", errors="ignore")).hexdigest()
    return digest[:12]


def _state_path(uri: str, collection: str, state_dir: Path) -> Path:
    safe_collection = "".join(ch if ch.isalnum() else "_" for ch in collection)
    filename = f"{safe_collection}__{_uri_fingerprint(uri)}.json"
    return state_dir / filename


def _compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(_HASH_CHUNK_SIZE)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


@dataclass
class DocumentState:
    hash: str
    ids: list[int]


class IngestState:
    def __init__(self, uri: str, collection: str, state_dir: Path) -> None:
        self.uri = uri
        self.collection = collection
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = _state_path(uri, collection, state_dir)
        self.documents: Dict[str, DocumentState] = {}

    @classmethod
    def load(cls, uri: str, collection: str, state_dir: Path | None = None) -> "IngestState":
        base_dir = state_dir or DEFAULT_STATE_DIR
        state = cls(uri=uri, collection=collection, state_dir=base_dir)
        if not state.path.exists():
            return state
        payload = json.loads(state.path.read_text(encoding="utf-8"))
        if payload.get("version") != _STATE_VERSION:
            return state
        raw_documents = payload.get("documents") or {}
        for source, entry in raw_documents.items():
            doc_hash = str(entry.get("hash") or "")
            ids = [int(value) for value in (entry.get("ids") or [])]
            if not doc_hash or not ids:
                continue
            state.documents[source] = DocumentState(hash=doc_hash, ids=ids)
        return state

    def save(self) -> None:
        payload = {
            "version": _STATE_VERSION,
            "uri": self.uri,
            "collection": self.collection,
            "documents": {
                source: {"hash": entry.hash, "ids": entry.ids}
                for source, entry in self.documents.items()
            },
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.documents.clear()
        if self.path.exists():
            self.path.unlink()

    def get(self, source: str) -> Optional[DocumentState]:
        return self.documents.get(source)

    def compute_hash(self, path: Path) -> str:
        return _compute_file_hash(path)

    def is_unchanged(self, source: str, source_hash: str, vector_store: "VectorStore") -> bool:
        entry = self.get(source)
        if entry is None or entry.hash != source_hash:
            return False
        # A missing ID indicates the state file is stale relative to the collection.
        return vector_store.ids_exist(entry.ids[:5])

    def mark_ingested(self, source: str, source_hash: str, ids: Iterable[int]) -> None:
        self.documents[source] = DocumentState(hash=source_hash, ids=list(ids))

    def remove(self, source: str) -> None:
        if source in self.documents:
            del self.documents[source]


# Avoid a runtime import cycle in type checking only.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from rag_core.vector_store import VectorStore
