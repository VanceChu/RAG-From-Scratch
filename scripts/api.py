"""FastAPI server for lightweight RAG frontend."""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymilvus import connections, utility

# Ensure project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.answer import answer_question
from rag_core.bm25_index import BM25Index
from rag_core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BM25_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION,
    DEFAULT_COLLECTION_RAW,
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_ENDPOINT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_ENABLE_BM25,
    DEFAULT_ENABLE_QUERY_REWRITE,
    DEFAULT_ENABLE_SPARSE,
    DEFAULT_FUSION,
    DEFAULT_HISTORY_TURNS,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_INDEX_EF_CONSTRUCTION,
    DEFAULT_INDEX_M,
    DEFAULT_INDEX_NLIST,
    DEFAULT_INDEX_TYPE,
    DEFAULT_INTERACTIVE,
    DEFAULT_MILVUS_URI,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_RERANK_ENABLED,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_REWRITE_STRATEGIES,
    DEFAULT_RRF_K,
    DEFAULT_SEARCH_K,
    DEFAULT_STATE_DIR,
    DEFAULT_STREAM,
    DEFAULT_TOP_K,
    DEFAULT_VOLC_API_BASE_URL,
    DEFAULT_VOLC_API_KEY,
    resolve_collection_name,
)
from rag_core.embeddings import EmbeddingModel
from rag_core.ingest import ingest_documents
from rag_core.ingest_state import IngestState
from rag_core.llm import OpenAIChatLLM
from rag_core.query_rewriter import ConversationTurn, RewriteResult, RewriteStrategy
from rag_core.rerank import Reranker
from rag_core.retriever import retrieve, retrieve_with_rewrite
from rag_core.sparse_embedding import APISparseEmbeddingModel
from rag_core.vector_store import SearchResult, VectorStore

UPLOAD_ROOT = PROJECT_ROOT / "data" / "uploads"


class ConversationTurnModel(BaseModel):
    """Pydantic model for conversation turn."""

    query: str
    response: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str | None = None
    collection_raw: bool = DEFAULT_COLLECTION_RAW
    milvus_uri: str = DEFAULT_MILVUS_URI
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_base_url: str | None = DEFAULT_EMBEDDING_BASE_URL or None
    embedding_endpoint: str | None = DEFAULT_EMBEDDING_ENDPOINT or None
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    index_type: str = DEFAULT_INDEX_TYPE
    index_nlist: int = DEFAULT_INDEX_NLIST
    index_m: int = DEFAULT_INDEX_M
    index_ef_construction: int = DEFAULT_INDEX_EF_CONSTRUCTION
    search_k: int = DEFAULT_SEARCH_K
    top_k: int = DEFAULT_TOP_K
    rerank: bool = DEFAULT_RERANK_ENABLED
    rerank_model: str = DEFAULT_RERANK_MODEL
    rerank_top_k: int = DEFAULT_RERANK_TOP_K
    history_turns: int = DEFAULT_HISTORY_TURNS
    enable_sparse: bool = DEFAULT_ENABLE_SPARSE
    enable_bm25: bool = DEFAULT_ENABLE_BM25
    fusion: str = DEFAULT_FUSION
    rrf_k: int = DEFAULT_RRF_K
    hybrid_alpha: float = DEFAULT_HYBRID_ALPHA
    stream: bool = DEFAULT_STREAM
    interactive: bool = DEFAULT_INTERACTIVE
    openai_model: str = DEFAULT_OPENAI_MODEL
    enable_rewrite: bool = DEFAULT_ENABLE_QUERY_REWRITE
    rewrite_strategies: list[str] = Field(
        default_factory=lambda: DEFAULT_REWRITE_STRATEGIES.split(",")
    )
    conversation_history: list[ConversationTurnModel] | None = None


class RewriteInfo(BaseModel):
    """Information about query rewriting."""

    original_query: str
    rewritten_query: str
    rewritten_queries: list[str] | None = None
    strategy: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    evidence: list[dict[str, Any]]
    timing: dict[str, Any]
    rewrite_info: RewriteInfo | None = None


def _ensure_local_milvus_parent(uri: str) -> None:
    if "://" in uri:
        return
    path = Path(uri).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_collection_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    return safe or "collection"


def _build_embedding_model(
    provider: str,
    model: str,
    base_url: str | None,
    endpoint: str | None,
    embedding_dim: int,
) -> EmbeddingModel:
    embedding_api_key = DEFAULT_EMBEDDING_API_KEY or None
    embedding_base_url = base_url or DEFAULT_EMBEDDING_BASE_URL or None
    embedding_endpoint = endpoint or DEFAULT_EMBEDDING_ENDPOINT or None
    normalized_provider = provider.strip().lower()
    if normalized_provider in {"volcengine", "ark"}:
        if not embedding_api_key:
            embedding_api_key = DEFAULT_VOLC_API_KEY or None
        if not embedding_base_url:
            embedding_base_url = DEFAULT_VOLC_API_BASE_URL or None
    resolved_dim = embedding_dim if embedding_dim and embedding_dim > 0 else None
    return EmbeddingModel(
        model,
        provider=provider,
        api_key=embedding_api_key,
        base_url=embedding_base_url,
        endpoint_path=embedding_endpoint,
        embedding_dim=resolved_dim,
    )


def _build_vector_store(
    uri: str,
    collection_name: str,
    embedding_dim: int,
    index_type: str,
    index_nlist: int,
    index_m: int,
    index_ef_construction: int,
) -> VectorStore:
    index_params = {
        "nlist": index_nlist,
        "M": index_m,
        "efConstruction": index_ef_construction,
    }
    return VectorStore(
        uri=uri,
        collection_name=collection_name,
        embedding_dim=embedding_dim,
        index_type=index_type,
        index_params=index_params,
    )


def _build_citations(results: Iterable[SearchResult]) -> list[str]:
    citations: list[str] = []
    for result in results:
        metadata = result.metadata or {}
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "n/a")
        try:
            source = Path(str(source)).name
        except (TypeError, ValueError):
            source = str(source)
        page_label = str(page) if isinstance(page, int) and page >= 0 else str(page)
        citations.append(f"{source}:{page_label}")
    return citations


app = FastAPI(title="RAG Light API", version="0.1.0")

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is required.")

    collection_name = resolve_collection_name(
        base_collection=request.collection or DEFAULT_COLLECTION,
        embedding_provider=request.embedding_provider,
        embedding_model=request.embedding_model,
        raw=request.collection_raw,
    )

    _ensure_local_milvus_parent(request.milvus_uri)
    connections.connect(alias="default", uri=request.milvus_uri)

    embedding_model = _build_embedding_model(
        provider=request.embedding_provider,
        model=request.embedding_model,
        base_url=request.embedding_base_url,
        endpoint=request.embedding_endpoint,
        embedding_dim=request.embedding_dim,
    )

    vector_store = _build_vector_store(
        uri=request.milvus_uri,
        collection_name=collection_name,
        embedding_dim=embedding_model.dimension,
        index_type=request.index_type,
        index_nlist=request.index_nlist,
        index_m=request.index_m,
        index_ef_construction=request.index_ef_construction,
    )

    if vector_store.collection.num_entities == 0:
        raise HTTPException(status_code=404, detail="Collection is empty. Run ingest first.")

    sparse_model = None
    if request.enable_sparse:
        sparse_model = APISparseEmbeddingModel()

    bm25_index = None
    if request.enable_bm25:
        bm25_index = BM25Index.load(collection=collection_name, base_dir=DEFAULT_BM25_DIR)

    llm = OpenAIChatLLM(model=request.openai_model)

    conversation_history: list[ConversationTurn] | None = None
    if request.conversation_history:
        conversation_history = [
            ConversationTurn(query=turn.query, response=turn.response)
            for turn in request.conversation_history
        ]

    timing: dict[str, Any] = {}
    retrieve_timing: dict[str, Any] = {}
    total_start = time.perf_counter()

    rewrite_info_response: RewriteInfo | None = None
    rewrite_result: RewriteResult | None = None

    if request.enable_rewrite:
        results, rewrite_result = retrieve_with_rewrite(
            query=request.query,
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=request.top_k,
            llm=llm,
            conversation_history=conversation_history,
            enable_rewrite=True,
            rewrite_strategies=request.rewrite_strategies,
            search_k=request.search_k,
            sparse_embedding_model=sparse_model,
            bm25_index=bm25_index,
            hybrid_alpha=request.hybrid_alpha,
            fusion=request.fusion,
            rrf_k=request.rrf_k,
            timing=retrieve_timing,
        )
        if rewrite_result:
            rewrite_info_response = RewriteInfo(
                original_query=rewrite_result.original_query,
                rewritten_query=rewrite_result.primary_query,
                rewritten_queries=rewrite_result.rewritten_queries,
                strategy=rewrite_result.strategy.value,
                metadata=rewrite_result.metadata,
            )
    else:
        results = retrieve(
            query=request.query,
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=request.top_k,
            search_k=request.search_k,
            sparse_embedding_model=sparse_model,
            bm25_index=bm25_index,
            hybrid_alpha=request.hybrid_alpha,
            fusion=request.fusion,
            rrf_k=request.rrf_k,
            timing=retrieve_timing,
        )

    timing["retrieve"] = retrieve_timing

    # Use the rewritten query for generation only when it is a safe
    # contextual disambiguation of the user's intent.
    generation_query = request.query
    if rewrite_result and rewrite_result.strategy == RewriteStrategy.CONTEXTUAL:
        generation_query = rewrite_result.primary_query

    if request.rerank and results:
        reranker = Reranker(request.rerank_model)
        rerank_start = time.perf_counter()
        results = reranker.rerank(generation_query, results, top_k=request.rerank_top_k)
        timing["rerank_s"] = time.perf_counter() - rerank_start
    else:
        results = results[: request.top_k]
        timing["rerank_s"] = 0.0

    if not results:
        timing["total_s"] = time.perf_counter() - total_start
        return QueryResponse(
            answer="No relevant context found.",
            citations=[],
            evidence=[],
            timing=timing,
            rewrite_info=rewrite_info_response,
        )

    llm_start = time.perf_counter()
    answer = answer_question(
        generation_query,
        results,
        llm,
        conversation_history=conversation_history,
        history_turns=request.history_turns,
        stream=False,
    )
    timing["llm_s"] = time.perf_counter() - llm_start
    timing["total_s"] = time.perf_counter() - total_start

    evidence = []
    for result in results:
        metadata = result.metadata or {}
        evidence.append(
            {
                "text": result.text,
                "score": result.score,
                "source": metadata.get("source", "unknown"),
                "section": metadata.get("section", "unknown"),
                "page": metadata.get("page", "n/a"),
            }
        )
    citations = _build_citations(results)
    return QueryResponse(
        answer=answer,
        citations=citations,
        evidence=evidence,
        timing=timing,
        rewrite_info=rewrite_info_response,
    )


@app.post("/ingest")
async def ingest(
    files: list[UploadFile] = File(...),
    collection: str = Form(DEFAULT_COLLECTION),
    collection_raw: bool = Form(DEFAULT_COLLECTION_RAW),
    milvus_uri: str = Form(DEFAULT_MILVUS_URI),
    embedding_provider: str = Form(DEFAULT_EMBEDDING_PROVIDER),
    embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL),
    embedding_base_url: str | None = Form(DEFAULT_EMBEDDING_BASE_URL or None),
    embedding_endpoint: str | None = Form(DEFAULT_EMBEDDING_ENDPOINT or None),
    embedding_dim: int = Form(DEFAULT_EMBEDDING_DIM),
    index_type: str = Form(DEFAULT_INDEX_TYPE),
    index_nlist: int = Form(DEFAULT_INDEX_NLIST),
    index_m: int = Form(DEFAULT_INDEX_M),
    index_ef_construction: int = Form(DEFAULT_INDEX_EF_CONSTRUCTION),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
    batch_size: int = Form(DEFAULT_BATCH_SIZE),
    enable_sparse: bool = Form(DEFAULT_ENABLE_SPARSE),
    enable_bm25: bool = Form(DEFAULT_ENABLE_BM25),
    reset: bool = Form(False),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    target_collection = resolve_collection_name(
        base_collection=collection,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        raw=collection_raw,
    )

    upload_dir = UPLOAD_ROOT / _safe_collection_name(target_collection) / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for file in files:
        if not file.filename:
            continue
        target_path = upload_dir / file.filename
        contents = await file.read()
        target_path.write_bytes(contents)
        saved_paths.append(target_path)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid files provided.")

    _ensure_local_milvus_parent(milvus_uri)
    connections.connect(alias="default", uri=milvus_uri)
    if reset and utility.has_collection(target_collection):
        utility.drop_collection(target_collection)

    embedding_model_obj = _build_embedding_model(
        provider=embedding_provider,
        model=embedding_model,
        base_url=embedding_base_url,
        endpoint=embedding_endpoint,
        embedding_dim=embedding_dim,
    )

    sparse_model = APISparseEmbeddingModel() if enable_sparse else None

    bm25_index = None
    if enable_bm25:
        bm25_index = BM25Index.load(collection=target_collection, base_dir=DEFAULT_BM25_DIR)
        if reset:
            bm25_index.clear()

    vector_store = _build_vector_store(
        uri=milvus_uri,
        collection_name=target_collection,
        embedding_dim=embedding_model_obj.dimension,
        index_type=index_type,
        index_nlist=index_nlist,
        index_m=index_m,
        index_ef_construction=index_ef_construction,
    )

    state = IngestState.load(
        uri=milvus_uri,
        collection=target_collection,
        state_dir=DEFAULT_STATE_DIR,
    )
    if reset:
        state.clear()

    timing: dict[str, Any] = {}
    summary = ingest_documents(
        paths=saved_paths,
        embedding_model=embedding_model_obj,
        vector_store=vector_store,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        state=state,
        sparse_embedding_model=sparse_model,
        bm25_index=bm25_index,
        timing=timing,
    )

    return {
        "collection": target_collection,
        "summary": {
            "new_docs": summary.new_documents,
            "refreshed_docs": summary.refreshed_documents,
            "skipped_docs": summary.skipped_documents,
            "inserted_chunks": summary.inserted_chunks,
            "deleted_chunks": summary.deleted_chunks,
        },
        "timing": timing,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("scripts.api:app", host="0.0.0.0", port=8000, reload=True)
