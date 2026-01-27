"""CLI entrypoint for RAG Q&A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import time
# Ensure project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.answer import answer_question
from rag_core.config import (
    DEFAULT_BM25_DIR,
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
    DEFAULT_STREAM,
    DEFAULT_TOP_K,
    DEFAULT_VOLC_API_BASE_URL,
    DEFAULT_VOLC_API_KEY,
    resolve_collection_name,
)
from rag_core.embeddings import EmbeddingModel
from rag_core.llm import OpenAIChatLLM
from rag_core.query_rewriter import ConversationTurn, RewriteResult, RewriteStrategy
from rag_core.rerank import Reranker
from rag_core.retriever import retrieve, retrieve_with_rewrite
from rag_core.vector_store import SearchResult, VectorStore


def _ensure_local_milvus_parent(uri: str) -> None:
    if "://" in uri:
        return
    path = Path(uri).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_collection_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    return safe or "collection"


def _milvus_target(uri: str) -> str:
    if "://" in uri:
        return uri
    return str(Path(uri).expanduser().resolve())


def _print_ask_overview(
    collection_name: str,
    milvus_uri: str,
    embedding_provider: str,
    embedding_model_name: str,
    embedding_dim: int,
    embedding_api_key_set: bool,
    embedding_base_url: str | None,
    embedding_endpoint: str | None,
    search_k: int,
    top_k: int,
    rerank: bool,
    rerank_model: str,
    rerank_top_k: int,
    fusion: str,
    rrf_k: int,
    stream: bool,
    interactive: bool,
    hybrid_alpha: float,
    enable_sparse: bool,
    enable_bm25: bool,
    bm25_path: Path | None,
    enable_rewrite: bool = True,
    rewrite_strategies: str = "contextual",
) -> None:
    print("Ask configuration:")
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
    print(f"- search_k: {search_k}")
    print(f"- top_k: {top_k}")
    print(f"- rerank: {rerank}")
    if rerank:
        print(f"- rerank_model: {rerank_model}")
        print(f"- rerank_top_k: {rerank_top_k}")
    print(f"- fusion: {fusion}")
    if fusion == "rrf":
        print(f"- rrf_k: {rrf_k}")
    print(f"- stream: {stream}")
    print(f"- interactive: {interactive}")
    if fusion == "weighted" or (enable_sparse and not enable_bm25):
        print(f"- hybrid_alpha: {hybrid_alpha}")
    print(f"- enable_sparse: {enable_sparse}")
    print(f"- enable_bm25: {enable_bm25}")
    print(f"- enable_rewrite: {enable_rewrite}")
    if enable_rewrite:
        print(f"- rewrite_strategies: {rewrite_strategies}")

    print("\nInputs:")
    print(f"- collection: {collection_name}")
    print(f"- milvus_storage: {_milvus_target(milvus_uri)}")
    if bm25_path:
        print(f"- bm25_index_file: {bm25_path}")

    print("\nOutputs:")
    print("- answer: stdout")
    print("- evidence: stdout")
    print()


def _print_ask_timing(timing: dict) -> None:
    if not timing:
        return
    retrieve_timing = timing.get("retrieve", {})
    print("\nTiming:")
    if retrieve_timing:
        print(f"embed_query_s: {retrieve_timing.get('embed_query_s', 0.0):.3f}")
        print(f"dense_search_s: {retrieve_timing.get('dense_search_s', 0.0):.3f}")
        if "bm25_search_s" in retrieve_timing:
            print(f"bm25_search_s: {retrieve_timing.get('bm25_search_s', 0.0):.3f}")
            print(f"merge_s: {retrieve_timing.get('merge_s', 0.0):.3f}")
        print(f"retrieve_total_s: {retrieve_timing.get('total_s', 0.0):.3f}")
    print(f"rerank_s: {timing.get('rerank_s', 0.0):.3f}")
    print(f"llm_s: {timing.get('llm_s', 0.0):.3f}")
    print(f"total_s: {timing.get('total_s', 0.0):.3f}")


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


def _prompt_float(label: str, default: float) -> float:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            print("Please enter a valid number.")


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


def _run_wizard(args: argparse.Namespace) -> argparse.Namespace:
    print("Ask wizard (press Enter to accept defaults).")
    query_default = args.query or ""
    query = _prompt_text("Question (blank for interactive mode)", query_default)
    if query:
        args.query = query
    else:
        args.query = None
        args.interactive = True

    args.rerank = _prompt_bool("Enable rerank", args.rerank)

    args.enable_bm25 = _prompt_bool("Enable BM25 retrieval", args.enable_bm25)
    if args.enable_bm25:
        args.fusion = _prompt_choice(
            "Fusion strategy",
            ["weighted", "rrf", "dense"],
            args.fusion,
        )
        if args.fusion == "rrf":
            args.rrf_k = _prompt_int("RRF k", args.rrf_k)
        elif args.fusion == "weighted":
            args.hybrid_alpha = _prompt_float(
                "Hybrid alpha (dense weight)",
                args.hybrid_alpha,
            )
    else:
        args.enable_sparse = _prompt_bool(
            "Enable sparse embeddings",
            args.enable_sparse,
        )
        if args.enable_sparse:
            args.hybrid_alpha = _prompt_float(
                "Hybrid alpha (dense weight)",
                args.hybrid_alpha,
            )

    if _prompt_bool("Configure advanced settings", False):
        args.search_k = _prompt_int("Search k", args.search_k)
        args.top_k = _prompt_int("Top k", args.top_k)
        if args.rerank:
            args.rerank_top_k = _prompt_int("Rerank top k", args.rerank_top_k)
        args.history_turns = _prompt_int("History turns", args.history_turns)
        args.stream = _prompt_bool("Stream answer tokens", args.stream)
        args.openai_model = _prompt_text("Chat model", args.openai_model)
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
            args.embedding_dim = _prompt_int(
                "Embedding dimension (0=auto)",
                args.embedding_dim,
            )

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
    parser = argparse.ArgumentParser(description="Ask questions over your documents.")
    parser.add_argument("--query", help="Question to ask.")
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Run an interactive setup wizard.",
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
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI chat model name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to use for answer context.",
    )
    parser.add_argument(
        "--search-k",
        type=int,
        default=DEFAULT_SEARCH_K,
        help="Number of chunks to retrieve before rerank.",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RERANK_ENABLED,
        help="Enable cross-encoder reranking (default: True).",
    )
    parser.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help="Cross-encoder model name.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=DEFAULT_RERANK_TOP_K,
        help="Number of chunks to keep after rerank.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=DEFAULT_INTERACTIVE,
        help="Start an interactive conversation loop.",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=DEFAULT_HISTORY_TURNS,
        help="Number of recent turns to include in the prompt.",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_STREAM,
        help="Stream answer tokens as they are generated (default: True).",
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
        "--fusion",
        choices=("weighted", "rrf", "dense"),
        default=DEFAULT_FUSION,
        help="Fusion strategy when BM25 is enabled: weighted, rrf, or dense.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF k parameter (used only when --fusion rrf).",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=DEFAULT_HYBRID_ALPHA,
        help="Dense weight for hybrid search (0.0=BM25/sparse, 1.0=dense).",
    )
    parser.add_argument(
        "--enable-rewrite",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_QUERY_REWRITE,
        help="Enable query rewriting for better retrieval (default: True).",
    )
    parser.add_argument(
        "--rewrite-strategies",
        default=DEFAULT_REWRITE_STRATEGIES,
        help="Comma-separated list of rewrite strategies: contextual,expansion,hyde,decomposition,step_back.",
    )
    return parser.parse_args()


def _print_evidence(results: list, max_chars: int = 240) -> None:
    print("\nEvidence:")
    if not results:
        print("- No relevant context found.")
        return
    for index, result in enumerate(results, start=1):
        metadata = result.metadata or {}
        source = metadata.get("source", "unknown")
        section = metadata.get("section", "unknown")
        page = metadata.get("page", -1)
        page_label = str(page) if isinstance(page, int) and page >= 0 else "n/a"
        snippet = result.text.replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3] + "..."
        print(f"[{index}] source={source} page={page_label} section={section}")
        print(f"    {snippet}")


def _answer_once(
    query: str,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    llm: OpenAIChatLLM,
    reranker: Reranker | None,
    top_k: int,
    search_k: int,
    rerank_top_k: int,
    history: list[ConversationTurn],
    history_turns: int,
    stream: bool,
    on_token: Callable[[str], None] | None,
    sparse_embedding_model: object | None = None,
    bm25_index: object | None = None,
    hybrid_alpha: float = 0.5,
    fusion: str = "weighted",
    rrf_k: int = 60,
    enable_rewrite: bool = True,
    rewrite_strategies: list[str] | None = None,
) -> tuple[str | None, list[SearchResult], dict, RewriteResult | None]:
    timing: dict = {}
    retrieve_timing: dict = {}
    total_start = time.perf_counter()

    rewrite_result: RewriteResult | None = None
    generation_query = query

    if enable_rewrite:
        results, rewrite_result = retrieve_with_rewrite(
            query=query,
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=top_k,
            llm=llm,
            conversation_history=history if history else None,
            enable_rewrite=True,
            rewrite_strategies=rewrite_strategies,
            search_k=search_k,
            sparse_embedding_model=sparse_embedding_model,
            bm25_index=bm25_index,
            hybrid_alpha=hybrid_alpha,
            fusion=fusion,
            rrf_k=rrf_k,
            timing=retrieve_timing,
        )
    else:
        results = retrieve(
            query=query,
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=top_k,
            search_k=search_k,
            sparse_embedding_model=sparse_embedding_model,
            bm25_index=bm25_index,
            hybrid_alpha=hybrid_alpha,
            fusion=fusion,
            rrf_k=rrf_k,
            timing=retrieve_timing,
        )

    timing["retrieve"] = retrieve_timing

    # Use the rewritten query for generation only when it is a safe
    # contextual disambiguation of the user's intent.
    if rewrite_result and rewrite_result.strategy == RewriteStrategy.CONTEXTUAL:
        generation_query = rewrite_result.primary_query

    if reranker and results:
        rerank_start = time.perf_counter()
        results = reranker.rerank(generation_query, results, top_k=rerank_top_k)
        timing["rerank_s"] = time.perf_counter() - rerank_start
    else:
        results = results[:top_k]
        timing["rerank_s"] = 0.0

    if not results:
        timing["llm_s"] = 0.0
        timing["total_s"] = time.perf_counter() - total_start
        return None, [], timing, rewrite_result

    llm_start = time.perf_counter()
    answer = answer_question(
        generation_query,
        results,
        llm,
        conversation_history=history if history else None,
        history_turns=history_turns,
        stream=stream,
        on_token=on_token,
    )
    timing["llm_s"] = time.perf_counter() - llm_start
    timing["total_s"] = time.perf_counter() - total_start
    return answer, results, timing, rewrite_result


def main() -> None:
    args = parse_args()
    if args.wizard:
        args = _run_wizard(args)
    collection_name = resolve_collection_name(
        base_collection=args.collection,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        raw=args.collection_raw,
    )

    _ensure_local_milvus_parent(args.milvus_uri)
    from pymilvus import connections
    connections.connect(alias="default", uri=args.milvus_uri)
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
        sparse_model = APISparseEmbeddingModel()

    bm25_index = None
    if args.enable_bm25:
        from rag_core.bm25_index import BM25Index
        bm25_index = BM25Index.load(collection=collection_name, base_dir=DEFAULT_BM25_DIR)
        
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

    bm25_path = None
    if args.enable_bm25:
        bm25_path = DEFAULT_BM25_DIR / f"{_safe_collection_name(collection_name)}.json"

    _print_ask_overview(
        collection_name=collection_name,
        milvus_uri=args.milvus_uri,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
        embedding_dim=embedding_model.dimension,
        embedding_api_key_set=bool(embedding_api_key),
        embedding_base_url=embedding_base_url,
        embedding_endpoint=embedding_endpoint,
        search_k=args.search_k,
        top_k=args.top_k,
        rerank=args.rerank,
        rerank_model=args.rerank_model,
        rerank_top_k=args.rerank_top_k,
        fusion=args.fusion,
        rrf_k=args.rrf_k,
        stream=args.stream,
        interactive=args.interactive or not args.query,
        hybrid_alpha=args.hybrid_alpha,
        enable_sparse=args.enable_sparse,
        enable_bm25=args.enable_bm25,
        bm25_path=bm25_path,
        enable_rewrite=args.enable_rewrite,
        rewrite_strategies=args.rewrite_strategies,
    )

    if vector_store.collection.num_entities == 0:
        print("No data found in collection. Run ingest first.")
        return

    print(f"Collection: {collection_name}")
    llm = OpenAIChatLLM(model=args.openai_model)
    reranker = Reranker(args.rerank_model) if args.rerank else None
    interactive_mode = args.interactive or not args.query
    history: list[ConversationTurn] = []

    rewrite_strategies = (
        [s.strip() for s in args.rewrite_strategies.split(",") if s.strip()]
        if isinstance(args.rewrite_strategies, str)
        else args.rewrite_strategies
    )

    def run_query(query: str) -> None:
        started_stream = False

        def on_token(token: str) -> None:
            nonlocal started_stream
            if not started_stream:
                print("Answer:")
                started_stream = True
            print(token, end="", flush=True)

        answer, results, timing, rewrite_result = _answer_once(
            query=query,
            embedding_model=embedding_model,
            vector_store=vector_store,
            llm=llm,
            reranker=reranker,
            top_k=args.top_k,
            search_k=args.search_k,
            rerank_top_k=args.rerank_top_k,
            history=history,
            history_turns=args.history_turns,
            stream=args.stream,
            on_token=on_token if args.stream else None,
            sparse_embedding_model=sparse_model,
            bm25_index=bm25_index,
            hybrid_alpha=args.hybrid_alpha,
            fusion=args.fusion,
            rrf_k=args.rrf_k,
            enable_rewrite=args.enable_rewrite,
            rewrite_strategies=rewrite_strategies,
        )

        if rewrite_result:
            rewritten_queries = rewrite_result.rewritten_queries
            print("\nQuery rewrite:")
            print(f"- strategy: {rewrite_result.strategy.value}")
            print(f"- original: {query}")
            if len(rewritten_queries) == 1:
                rewritten = rewritten_queries[0]
                if rewritten != query:
                    print(f"- rewritten: {rewritten}")
            else:
                print("- rewritten queries:")
                for index, rewritten in enumerate(rewritten_queries, start=1):
                    primary_tag = " (primary)" if index == 1 else ""
                    print(f"  {index}. {rewritten}{primary_tag}")

        if not results or answer is None:
            print("No relevant context found.")
            return
        if args.stream:
            if not started_stream:
                print("Answer:")
                print(answer)
            else:
                print()
        else:
            print("Answer:")
            print(answer)
        _print_evidence(results)
        _print_ask_timing(timing)
        history.append(ConversationTurn(query=query, response=answer))

    if not interactive_mode and args.query:
        run_query(args.query)
        return

    print("Interactive mode. Type 'exit' or press Enter on an empty line to quit.")
    if args.query:
        run_query(args.query)

    while True:
        try:
            user_query = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_query or user_query.lower() in {"exit", "quit", ":q"}:
            print("Exiting.")
            break
        run_query(user_query)


if __name__ == "__main__":
    main()
