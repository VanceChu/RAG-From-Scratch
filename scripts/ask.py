"""CLI entrypoint for RAG Q&A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

# Ensure project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.answer import answer_question
from rag_core.config import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_INDEX_EF_CONSTRUCTION,
    DEFAULT_INDEX_M,
    DEFAULT_INDEX_NLIST,
    DEFAULT_INDEX_TYPE,
    DEFAULT_MILVUS_URI,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_SEARCH_K,
    DEFAULT_TOP_K,
    resolve_collection_name,
)
from rag_core.embeddings import EmbeddingModel
from rag_core.llm import OpenAIChatLLM
from rag_core.rerank import Reranker
from rag_core.retriever import retrieve
from rag_core.vector_store import SearchResult, VectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions over your documents.")
    parser.add_argument("--query", help="Question to ask.")
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
        action="store_true",
        help="Enable cross-encoder reranking.",
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
        help="Start an interactive conversation loop.",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=3,
        help="Number of recent turns to include in the prompt.",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream answer tokens as they are generated (default: True).",
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
    history: list[tuple[str, str]],
    history_turns: int,
    stream: bool,
    on_token: Callable[[str], None] | None,
) -> tuple[str | None, list[SearchResult]]:
    results = retrieve(
        query=query,
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=top_k,
        search_k=search_k,
    )

    if reranker and results:
        results = reranker.rerank(query, results, top_k=rerank_top_k)
    else:
        results = results[:top_k]

    if not results:
        return None, []

    answer = answer_question(
        query,
        results,
        llm,
        conversation_history=history,
        history_turns=history_turns,
        stream=stream,
        on_token=on_token,
    )
    return answer, results


def main() -> None:
    args = parse_args()
    collection_name = resolve_collection_name(
        base_collection=args.collection,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        raw=args.collection_raw,
    )

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

    if vector_store.collection.num_entities == 0:
        print("No data found in collection. Run ingest first.")
        return

    print(f"Collection: {collection_name}")
    llm = OpenAIChatLLM(model=args.openai_model)
    reranker = Reranker(args.rerank_model) if args.rerank else None
    interactive_mode = args.interactive or not args.query
    history: list[tuple[str, str]] = []

    def run_query(query: str) -> None:
        started_stream = False

        def on_token(token: str) -> None:
            nonlocal started_stream
            if not started_stream:
                print("Answer:")
                started_stream = True
            print(token, end="", flush=True)

        answer, results = _answer_once(
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
        )
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
        history.append((query, answer))

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
