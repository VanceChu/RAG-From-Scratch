"""RAGAS evaluation helpers with optional dependency handling."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from rag_core.config import (
    DEFAULT_EVAL_OPENAI_API_KEY,
    DEFAULT_EVAL_OPENAI_BASE_URL,
    DEFAULT_EVAL_OPENAI_MODEL,
)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation LLM."""

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str | None = None

    @classmethod
    def from_defaults(cls) -> "EvaluationConfig":
        return cls(
            openai_api_key=DEFAULT_EVAL_OPENAI_API_KEY or None,
            openai_base_url=DEFAULT_EVAL_OPENAI_BASE_URL or None,
            openai_model=DEFAULT_EVAL_OPENAI_MODEL or None,
        )


@dataclass
class EvalSample:
    """Single evaluation sample."""

    question: str
    answer: str | None = None
    contexts: list[str] | None = None
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation summary and per-sample results."""

    scores: dict[str, float]
    per_sample: list[dict[str, Any]]


def _lazy_import_ragas() -> tuple[Any, list[Any], Any]:
    try:
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "RAGAS is not installed. Install with: pip install ragas datasets"
        ) from exc
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    return evaluate, metrics, Dataset


def _metric_name(metric: Any) -> str:
    return getattr(metric, "name", None) or getattr(metric, "__name__", None) or str(metric)


def _normalize_contexts(contexts: Iterable[Any] | None) -> list[str]:
    if contexts is None:
        return []
    if isinstance(contexts, str):
        return [contexts]
    normalized: list[str] = []
    for item in contexts:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


@contextmanager
def _override_env(config: EvaluationConfig) -> Iterable[None]:
    overrides: dict[str, str] = {}
    if config.openai_api_key:
        overrides["OPENAI_API_KEY"] = config.openai_api_key
    if config.openai_base_url:
        overrides["OPENAI_BASE_URL"] = config.openai_base_url
    if config.openai_model:
        overrides["OPENAI_MODEL"] = config.openai_model
    original: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            original[key] = os.getenv(key)
            os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_ragas_llm(config: EvaluationConfig) -> Any | None:
    try:
        from ragas.llms import OpenAI as RagasOpenAI  # type: ignore
    except Exception:
        return None
    kwargs: dict[str, Any] = {}
    if config.openai_api_key:
        kwargs["api_key"] = config.openai_api_key
    if config.openai_base_url:
        kwargs["base_url"] = config.openai_base_url
    if config.openai_model:
        kwargs["model"] = config.openai_model
    try:
        return RagasOpenAI(**kwargs)
    except Exception:
        return None


def load_eval_dataset(path: Path) -> list[EvalSample]:
    """Load evaluation dataset from JSON or JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    items: list[dict[str, Any]] = []
    if not text:
        return []
    if text.startswith("["):
        items = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    samples: list[EvalSample] = []
    for item in items:
        question = item.get("question") or item.get("query")
        if not question:
            continue
        samples.append(
            EvalSample(
                question=str(question),
                answer=item.get("answer"),
                contexts=_normalize_contexts(item.get("contexts")),
                ground_truth=item.get("ground_truth") or item.get("ground_truths"),
                metadata={k: v for k, v in item.items() if k not in {"question", "query", "answer", "contexts", "ground_truth", "ground_truths"}},
            )
        )
    return samples


def run_ragas_evaluation(
    samples: Sequence[EvalSample],
    config: EvaluationConfig | None = None,
) -> EvalResult:
    if not samples:
        return EvalResult(scores={}, per_sample=[])

    for sample in samples:
        if not sample.answer:
            raise ValueError("Evaluation sample is missing answer.")
        if not sample.contexts:
            raise ValueError("Evaluation sample is missing contexts.")

    evaluate, metrics, dataset_cls = _lazy_import_ragas()
    config = config or EvaluationConfig.from_defaults()

    include_recall = all(sample.ground_truth for sample in samples)
    if not include_recall:
        metrics = [m for m in metrics if _metric_name(m) != "context_recall"]

    data: dict[str, Any] = {
        "question": [sample.question for sample in samples],
        "answer": [sample.answer for sample in samples],
        "contexts": [sample.contexts for sample in samples],
    }
    if include_recall:
        data["ground_truth"] = [sample.ground_truth for sample in samples]
        data["ground_truths"] = [[sample.ground_truth] for sample in samples]

    dataset = dataset_cls.from_dict(data)
    eval_llm = _build_ragas_llm(config)

    with _override_env(config):
        kwargs: dict[str, Any] = {"metrics": metrics}
        if eval_llm is not None:
            if "llm" in evaluate.__code__.co_varnames:
                kwargs["llm"] = eval_llm
        result = evaluate(dataset, **kwargs)

    df = result.to_pandas()
    scores: dict[str, float] = {}
    for metric in metrics:
        name = _metric_name(metric)
        if name in df.columns:
            scores[name] = float(df[name].mean())

    per_sample = df.to_dict("records")
    return EvalResult(scores=scores, per_sample=per_sample)


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
    config: EvaluationConfig | None = None,
) -> EvalResult:
    sample = EvalSample(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )
    return run_ragas_evaluation([sample], config=config)
