"""CLI entrypoint for RAG Q&A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.answer import answer_question
from rag.config import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MILVUS_URI,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_SEARCH_K,
    DEFAULT_TOP_K,
)
from rag.embeddings import EmbeddingModel
from rag.llm import OpenAIChatLLM
from rag.rerank import Reranker
from rag.retriever import retrieve
from rag.vector_store import VectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions over your documents.")
    parser.add_argument("--query", required=True, help="Question to ask.")
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


def main() -> None:
    args = parse_args()

    embedding_model = EmbeddingModel(args.embedding_model)
    vector_store = VectorStore(
        uri=args.milvus_uri,
        collection_name=args.collection,
        embedding_dim=embedding_model.dimension,
    )

    if vector_store.collection.num_entities == 0:
        print("No data found in collection. Run ingest first.")
        return

    results = retrieve(
        query=args.query,
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=args.top_k,
        search_k=args.search_k,
    )

    if args.rerank and results:
        reranker = Reranker(args.rerank_model)
        results = reranker.rerank(args.query, results, top_k=args.rerank_top_k)
    else:
        results = results[: args.top_k]

    if not results:
        print("No relevant context found.")
        return

    llm = OpenAIChatLLM(model=args.openai_model)
    answer = answer_question(args.query, results, llm)
    print("Answer:")
    print(answer)
    _print_evidence(results)


if __name__ == "__main__":
    main()
