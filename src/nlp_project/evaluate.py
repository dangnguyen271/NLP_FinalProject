from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from nlp_project.config import AppConfig
from nlp_project.data import load_dataset, split_dataset
from nlp_project.model import load_model


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    metrics_path: Path
    classification_report_path: Path
    error_analysis_path: Path


def evaluate_model(
    config: AppConfig,
    model: Pipeline | None = None,
    test_df: pd.DataFrame | None = None,
) -> EvaluationResult:
    if model is None:
        model = load_model(config.model.artifact_path)
    if test_df is None:
        df = load_dataset(config)
        test_df = split_dataset(df, config).test

    text_column = config.data.text_column
    label_column = config.data.label_column
    y_true = [str(label) for label in test_df[label_column]]
    y_pred = [str(label) for label in model.predict(test_df[text_column])]
    labels = sorted(set(y_true) | set(y_pred))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(report_dict[label]["precision"]),
                "recall": float(report_dict[label]["recall"]),
                "f1": float(report_dict[label]["f1-score"]),
                "support": int(report_dict[label]["support"]),
            }
            for label in labels
            if label in report_dict
        },
        "labels": labels,
        "test_rows": len(test_df),
    }

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.reports_dir / "metrics.json"
    report_path = config.reports_dir / "classification_report.txt"
    error_path = config.reports_dir / "error_analysis.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_text = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    report_path.write_text(report_text, encoding="utf-8")

    error_analysis = pd.DataFrame(
        {
            "text": list(test_df[text_column]),
            "true_label": y_true,
            "predicted_label": y_pred,
            "correct": [true == pred for true, pred in zip(y_true, y_pred, strict=True)],
        }
    )
    if config.data.id_column and config.data.id_column in test_df.columns:
        error_analysis.insert(0, config.data.id_column, list(test_df[config.data.id_column]))
    error_analysis.to_csv(error_path, index=False)

    return EvaluationResult(
        metrics=metrics,
        metrics_path=metrics_path,
        classification_report_path=report_path,
        error_analysis_path=error_path,
    )
