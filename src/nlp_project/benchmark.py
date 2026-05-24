"""Cross-model benchmarking utilities.

Trains each supported model type on the configured dataset and writes a single
comparison report. This is the primary artefact backing the report's "model
comparison" research question.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import json
import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from nlp_project.config import AppConfig
from nlp_project.data import load_and_validate_dataset, split_dataset
from nlp_project.model import (
    SUPPORTED_MODEL_TYPES,
    build_model,
    train_model,
)


@dataclass(frozen=True)
class ModelBenchmark:
    model_type: str
    train_seconds: float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    cv_scores: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BenchmarkReport:
    rows: list[ModelBenchmark]
    json_path: Path
    csv_path: Path


def run_benchmark(
    config: AppConfig,
    cv_folds: int = 5,
    model_types: tuple[str, ...] = SUPPORTED_MODEL_TYPES,
) -> BenchmarkReport:
    """Train every requested model on the same split and report metrics."""

    df, _ = load_and_validate_dataset(config)
    splits = split_dataset(df, config)
    text_col = config.data.text_column
    label_col = config.data.label_column

    X_train = splits.train[text_col]
    y_train = splits.train[label_col]
    X_test = splits.test[text_col]
    y_test = [str(label) for label in splits.test[label_col]]

    rows: list[ModelBenchmark] = []
    safe_cv = _safe_cv_folds(y_train.tolist(), cv_folds)

    for model_type in model_types:
        scoped_config = _config_for_model(config, model_type)

        start = time.perf_counter()
        model, _ = train_model(scoped_config, splits.train)
        train_seconds = time.perf_counter() - start

        predictions = [str(label) for label in model.predict(X_test)]
        accuracy = float(accuracy_score(y_test, predictions))
        macro_f1 = float(f1_score(y_test, predictions, average="macro", zero_division=0))
        weighted_f1 = float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        )

        if safe_cv >= 2:
            cv_pipeline = build_model(scoped_config)
            cv_scores = cross_val_score(
                cv_pipeline,
                X_train,
                y_train,
                cv=StratifiedKFold(
                    n_splits=safe_cv,
                    shuffle=True,
                    random_state=config.project.random_seed,
                ),
                scoring="f1_macro",
            )
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
            cv_tuple = tuple(float(score) for score in cv_scores)
        else:
            cv_mean = float("nan")
            cv_std = float("nan")
            cv_tuple = ()

        rows.append(
            ModelBenchmark(
                model_type=model_type,
                train_seconds=float(train_seconds),
                accuracy=accuracy,
                macro_f1=macro_f1,
                weighted_f1=weighted_f1,
                cv_macro_f1_mean=cv_mean,
                cv_macro_f1_std=cv_std,
                cv_scores=cv_tuple,
            )
        )

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = config.reports_dir / "benchmark.json"
    csv_path = config.reports_dir / "benchmark.csv"
    json_path.write_text(json.dumps([row.__dict__ for row in rows], indent=2), encoding="utf-8")
    pd.DataFrame([row.__dict__ for row in rows]).to_csv(csv_path, index=False)

    # Re-fit the configured "main" model so subsequent CLI commands (predict,
    # evaluate) keep working on the artefact the user asked for.
    train_model(config, splits.train)

    return BenchmarkReport(rows=rows, json_path=json_path, csv_path=csv_path)


def summarize_benchmark(rows: list[ModelBenchmark]) -> str:
    headers = ("model", "accuracy", "macro_f1", "weighted_f1", "cv_macro_f1", "train_s")
    width = max(len(headers[0]), *(len(row.model_type) for row in rows))
    lines = [
        f"{headers[0]:<{width}}  "
        f"{headers[1]:>8}  {headers[2]:>8}  {headers[3]:>11}  "
        f"{headers[4]:>14}  {headers[5]:>7}"
    ]
    for row in rows:
        cv = f"{row.cv_macro_f1_mean:.3f}±{row.cv_macro_f1_std:.3f}"
        lines.append(
            f"{row.model_type:<{width}}  "
            f"{row.accuracy:>8.3f}  {row.macro_f1:>8.3f}  {row.weighted_f1:>11.3f}  "
            f"{cv:>14}  {row.train_seconds:>7.2f}"
        )
    return "\n".join(lines)


def _config_for_model(config: AppConfig, model_type: str) -> AppConfig:
    """Return a copy of ``config`` whose model.type and artifact_path target model_type."""

    artifact = (
        config.model.artifact_path.parent / f"model_{model_type}.joblib"
    )
    new_model = replace(config.model, type=model_type, artifact_path=artifact)
    return replace(config, model=new_model)


def _safe_cv_folds(labels: list[Any], requested: int) -> int:
    counts: dict[Any, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return 0
    return max(2, min(requested, min(counts.values())))
