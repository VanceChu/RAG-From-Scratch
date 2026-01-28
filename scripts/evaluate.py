"""CLI tool for RAGAS evaluation."""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
import sys
from pathlib import Path
from typing import Any

from pymilvus import connections

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_core.answer import answer_question
from rag_core.bm25_index import BM25Index
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
    DEFAULT_EVAL_OPENAI_API_KEY,
    DEFAULT_EVAL_OPENAI_BASE_URL,
    DEFAULT_EVAL_OPENAI_MODEL,
    DEFAULT_FUSION,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_INDEX_EF_CONSTRUCTION,
    DEFAULT_INDEX_M,
    DEFAULT_INDEX_NLIST,
    DEFAULT_INDEX_TYPE,
    DEFAULT_MILVUS_URI,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_RERANK_ENABLED,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_REWRITE_STRATEGIES,
    DEFAULT_RRF_K,
    DEFAULT_SEARCH_K,
    DEFAULT_TOP_K,
    DEFAULT_VOLC_API_BASE_URL,
    DEFAULT_VOLC_API_KEY,
    resolve_collection_name,
)
from rag_core.embeddings import EmbeddingModel
from rag_core.evaluation import EvaluationConfig, EvalSample, load_eval_dataset, run_ragas_evaluation
from rag_core.llm import OpenAIChatLLM
from rag_core.observability import log_score, trace, update_trace_metadata
from rag_core.rerank import Reranker
from rag_core.retriever import retrieve, retrieve_with_rewrite
from rag_core.sparse_embedding import APISparseEmbeddingModel
from rag_core.vector_store import VectorStore


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG system with RAGAS.")
    parser.add_argument("--dataset", required=True, help="Path to eval dataset (JSON/JSONL).")
    parser.add_argument("--output", help="Output JSON path for detailed results.")
    parser.add_argument("--output-csv", help="Output CSV path for per-sample results.")
    parser.add_argument("--output-html", help="Output HTML report path.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples.")
    parser.add_argument("--upload-langfuse", action="store_true", help="Upload scores to LangFuse.")

    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--collection-raw", action=argparse.BooleanOptionalAction, default=DEFAULT_COLLECTION_RAW)
    parser.add_argument("--milvus-uri", default=DEFAULT_MILVUS_URI)

    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-base-url", default=DEFAULT_EMBEDDING_BASE_URL)
    parser.add_argument("--embedding-endpoint", default=DEFAULT_EMBEDDING_ENDPOINT)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)

    parser.add_argument("--index-type", default=DEFAULT_INDEX_TYPE)
    parser.add_argument("--index-nlist", type=int, default=DEFAULT_INDEX_NLIST)
    parser.add_argument("--index-m", type=int, default=DEFAULT_INDEX_M)
    parser.add_argument("--index-ef-construction", type=int, default=DEFAULT_INDEX_EF_CONSTRUCTION)

    parser.add_argument("--search-k", type=int, default=DEFAULT_SEARCH_K)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=DEFAULT_RERANK_ENABLED)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument("--enable-sparse", action=argparse.BooleanOptionalAction, default=DEFAULT_ENABLE_SPARSE)
    parser.add_argument("--enable-bm25", action=argparse.BooleanOptionalAction, default=DEFAULT_ENABLE_BM25)
    parser.add_argument("--fusion", default=DEFAULT_FUSION)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--hybrid-alpha", type=float, default=DEFAULT_HYBRID_ALPHA)
    parser.add_argument("--enable-rewrite", action=argparse.BooleanOptionalAction, default=DEFAULT_ENABLE_QUERY_REWRITE)
    parser.add_argument("--rewrite-strategies", default=DEFAULT_REWRITE_STRATEGIES)

    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--eval-openai-model", default=DEFAULT_EVAL_OPENAI_MODEL)
    parser.add_argument("--eval-openai-api-key", default=DEFAULT_EVAL_OPENAI_API_KEY)
    parser.add_argument("--eval-openai-base-url", default=DEFAULT_EVAL_OPENAI_BASE_URL)
    return parser.parse_args()


def _stringify_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _stringify_value(record.get(key)) for key in fieldnames})


def _write_html(
    path: Path,
    scores: dict[str, float],
    records: list[dict[str, Any]],
    dataset_name: str,
) -> None:
    rows = []
    for record in records:
        cells = []
        for key in sorted(record.keys()):
            value = _stringify_value(record.get(key))
            cells.append(f"<td>{html.escape(value)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    headers = "".join(f"<th>{html.escape(key)}</th>" for key in sorted(records[0].keys())) if records else ""
    score_items = "".join(
        f"<li><strong>{html.escape(key)}</strong>: {value:.4f}</li>"
        for key, value in scores.items()
    )
    rendered = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>RAGAS Report</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 32px; color: #111; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #666; margin-bottom: 16px; }}
    ul {{ padding-left: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #e2e2e2; padding: 8px; font-size: 12px; }}
    th {{ background: #f6f6f6; text-align: left; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>RAGAS Evaluation Report</h1>
  <div class="meta">Dataset: {html.escape(dataset_name)} · Generated: {datetime.now().isoformat(timespec="seconds")}</div>
  <h2>Scores</h2>
  <ul>{score_items}</ul>
  <h2>Per-sample Details</h2>
  <table>
    <thead><tr>{headers}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>"""
    path.write_text(rendered, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    samples = load_eval_dataset(dataset_path)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No evaluation samples found.")

    collection_name = resolve_collection_name(
        base_collection=args.collection,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        raw=args.collection_raw,
    )

    connections.connect(alias="default", uri=args.milvus_uri)

    embedding_model = _build_embedding_model(
        provider=args.embedding_provider,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        endpoint=args.embedding_endpoint,
        embedding_dim=args.embedding_dim,
    )
    vector_store = _build_vector_store(
        uri=args.milvus_uri,
        collection_name=collection_name,
        embedding_dim=embedding_model.dimension,
        index_type=args.index_type,
        index_nlist=args.index_nlist,
        index_m=args.index_m,
        index_ef_construction=args.index_ef_construction,
    )

    sparse_model = APISparseEmbeddingModel() if args.enable_sparse else None
    bm25_index = None
    if args.enable_bm25:
        bm25_index = BM25Index.load(collection=collection_name, base_dir=DEFAULT_BM25_DIR)

    llm = OpenAIChatLLM(model=args.openai_model)
    reranker = Reranker(args.rerank_model) if args.rerank else None
    rewrite_strategies = [s.strip() for s in args.rewrite_strategies.split(",") if s.strip()]

    eval_samples: list[EvalSample] = []
    for sample in samples:
        if args.enable_rewrite:
            results, _ = retrieve_with_rewrite(
                query=sample.question,
                embedding_model=embedding_model,
                vector_store=vector_store,
                top_k=args.top_k,
                llm=llm,
                conversation_history=None,
                enable_rewrite=True,
                rewrite_strategies=rewrite_strategies,
                search_k=args.search_k,
                sparse_embedding_model=sparse_model,
                bm25_index=bm25_index,
                hybrid_alpha=args.hybrid_alpha,
                fusion=args.fusion,
                rrf_k=args.rrf_k,
            )
        else:
            results = retrieve(
                query=sample.question,
                embedding_model=embedding_model,
                vector_store=vector_store,
                top_k=args.top_k,
                search_k=args.search_k,
                sparse_embedding_model=sparse_model,
                bm25_index=bm25_index,
                hybrid_alpha=args.hybrid_alpha,
                fusion=args.fusion,
                rrf_k=args.rrf_k,
            )

        if reranker and results:
            results = reranker.rerank(sample.question, results, top_k=args.rerank_top_k)
        else:
            results = results[: args.top_k]

        contexts = [result.text for result in results]
        answer = answer_question(
            sample.question,
            results,
            llm,
            conversation_history=None,
            history_turns=0,
            stream=False,
        )
        eval_samples.append(
            EvalSample(
                question=sample.question,
                answer=answer,
                contexts=contexts,
                ground_truth=sample.ground_truth,
                metadata=sample.metadata,
            )
        )

    eval_config = EvaluationConfig(
        openai_api_key=args.eval_openai_api_key or None,
        openai_base_url=args.eval_openai_base_url or None,
        openai_model=args.eval_openai_model or None,
    )
    try:
        result = run_ragas_evaluation(eval_samples, config=eval_config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print("RAGAS evaluation results:")
    for key, value in result.scores.items():
        print(f"- {key}: {value:.4f}")

    output_payload: dict[str, Any] = {
        "scores": result.scores,
        "per_sample": result.per_sample,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2))
    if args.output_csv:
        _write_csv(Path(args.output_csv), result.per_sample)
    if args.output_html:
        _write_html(
            Path(args.output_html),
            result.scores,
            result.per_sample,
            dataset_name=str(dataset_path),
        )

    if args.upload_langfuse:
        with trace(
            name="ragas_batch_eval",
            metadata={"dataset": str(dataset_path), "sample_count": len(eval_samples)},
        ) as trace_ctx:
            for name, value in result.scores.items():
                log_score(trace_ctx, name=f"ragas_{name}", value=value)
            update_trace_metadata(trace_ctx, {"ragas_scores": result.scores})


if __name__ == "__main__":
    main()
