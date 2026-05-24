"""ROUGE-based evaluation for the summarisation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable
import json

import pandas as pd

from nlp_project.config import AppConfig
from nlp_project.data import load_and_validate_dataset, split_dataset
from nlp_project.summarize import (
    SUPPORTED_METHODS,
    SummaryResult,
    available_methods,
    summarize,
)


@dataclass(frozen=True)
class MethodEvaluation:
    method: str
    rouge: dict[str, float]
    mean_summary_length: float
    mean_compression_ratio: float
    mean_inference_seconds: float
    n_examples: int


@dataclass(frozen=True)
class EvaluationResult:
    per_method: list[MethodEvaluation]
    metrics_path: Path
    qualitative_path: Path
    method_summary_path: Path


def evaluate_summarizers(
    config: AppConfig,
    methods: Iterable[str] | None = None,
    test_df: pd.DataFrame | None = None,
    max_examples: int | None = None,
) -> EvaluationResult:
    """Run every requested method on the test split, score with ROUGE, write reports."""

    if test_df is None:
        df, _ = load_and_validate_dataset(config)
        test_df = split_dataset(df, config).test

    chosen_methods = tuple(methods) if methods is not None else available_methods(config)
    if not chosen_methods:
        raise ValueError("No summarisation methods available; check the configuration.")

    cap = max_examples if max_examples is not None else config.evaluation.max_examples
    if cap > 0:
        test_df = test_df.head(cap)

    text_col = config.data.text_column
    summary_col = config.data.summary_column
    id_col = config.data.id_column if config.data.id_column in test_df.columns else None
    source_col = (
        config.data.source_column
        if config.data.source_column and config.data.source_column in test_df.columns
        else None
    )

    rouge_scorer = _make_rouge_scorer(config.evaluation.rouge_variants)

    per_row_records: list[dict[str, Any]] = []
    per_method: list[MethodEvaluation] = []

    for method in chosen_methods:
        scores_by_variant: dict[str, list[float]] = {
            variant: [] for variant in config.evaluation.rouge_variants
        }
        summary_lengths: list[int] = []
        compression_ratios: list[float] = []
        elapsed_times: list[float] = []

        for _, row in test_df.iterrows():
            article = str(row[text_col])
            reference = str(row[summary_col])
            result: SummaryResult = summarize(method, article, config)
            elapsed_times.append(result.elapsed_seconds)
            summary_lengths.append(len(result.summary.split()))
            if article.strip():
                compression_ratios.append(
                    len(result.summary.split()) / max(1, len(article.split()))
                )

            scores = rouge_scorer.score(reference, result.summary)
            for variant, score in scores.items():
                scores_by_variant[variant].append(score.fmeasure)

            record = {
                "method": method,
                "article": article,
                "reference_summary": reference,
                "generated_summary": result.summary,
                "summary_length": len(result.summary.split()),
                "elapsed_seconds": result.elapsed_seconds,
            }
            for variant, score in scores.items():
                record[f"{variant}_f1"] = score.fmeasure
                record[f"{variant}_precision"] = score.precision
                record[f"{variant}_recall"] = score.recall
            if id_col:
                record[id_col] = row[id_col]
            if source_col:
                record[source_col] = row[source_col]
            per_row_records.append(record)

        per_method.append(
            MethodEvaluation(
                method=method,
                rouge={
                    variant: float(mean(values)) if values else 0.0
                    for variant, values in scores_by_variant.items()
                },
                mean_summary_length=float(mean(summary_lengths)) if summary_lengths else 0.0,
                mean_compression_ratio=(
                    float(mean(compression_ratios)) if compression_ratios else 0.0
                ),
                mean_inference_seconds=(
                    float(mean(elapsed_times)) if elapsed_times else 0.0
                ),
                n_examples=len(test_df),
            )
        )

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.reports_dir / "rouge_metrics.json"
    qualitative_path = config.reports_dir / "qualitative_review.csv"
    method_summary_path = config.reports_dir / "method_summary.csv"

    metrics_path.write_text(
        json.dumps([m.__dict__ for m in per_method], indent=2), encoding="utf-8"
    )
    pd.DataFrame(per_row_records).to_csv(qualitative_path, index=False)

    method_rows = []
    for entry in per_method:
        row = {"method": entry.method, "n_examples": entry.n_examples}
        row.update({variant: round(value, 4) for variant, value in entry.rouge.items()})
        row["mean_summary_length"] = round(entry.mean_summary_length, 2)
        row["mean_compression_ratio"] = round(entry.mean_compression_ratio, 4)
        row["mean_inference_seconds"] = round(entry.mean_inference_seconds, 4)
        method_rows.append(row)
    pd.DataFrame(method_rows).to_csv(method_summary_path, index=False)

    return EvaluationResult(
        per_method=per_method,
        metrics_path=metrics_path,
        qualitative_path=qualitative_path,
        method_summary_path=method_summary_path,
    )


def summarize_evaluation(per_method: list[MethodEvaluation]) -> str:
    if not per_method:
        return "no methods evaluated"
    variants = list(per_method[0].rouge.keys())
    header = "method".ljust(12) + "  " + "  ".join(v.rjust(10) for v in variants)
    lines = [header]
    for entry in per_method:
        cells = "  ".join(f"{entry.rouge[v]:>10.3f}" for v in variants)
        lines.append(f"{entry.method:<12}  {cells}")
    return "\n".join(lines)


def _make_rouge_scorer(variants: tuple[str, ...]):
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(list(variants), use_stemmer=True)
