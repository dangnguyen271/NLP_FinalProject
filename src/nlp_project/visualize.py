"""Plotting utilities for the final report.

All functions degrade gracefully if matplotlib is unavailable by returning ``None``
instead of raising. This keeps the headless CI path identical to the local one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from nlp_project.config import AppConfig
from nlp_project.data import load_and_validate_dataset, split_dataset
from nlp_project.model import build_model, load_model


@dataclass(frozen=True)
class VisualizationOutputs:
    confusion_matrix: Path | None
    class_distribution: Path | None
    top_features: Path | None
    learning_curve: Path | None
    benchmark_chart: Path | None


def generate_all(
    config: AppConfig,
    benchmark_rows: list | None = None,
    *,
    output_dir: Path | None = None,
) -> VisualizationOutputs:
    """Produce every plot the final report needs."""

    target_dir = output_dir or (config.reports_dir / "figures")
    target_dir.mkdir(parents=True, exist_ok=True)

    df, summary = load_and_validate_dataset(config)
    splits = split_dataset(df, config)

    cm_path = save_confusion_matrix_for_config(config, splits.test, target_dir)
    dist_path = save_class_distribution(summary.class_counts, target_dir)
    feat_path = save_top_features(config, target_dir)
    curve_path = save_learning_curve(config, splits.train, target_dir)
    bench_path = save_benchmark_chart(benchmark_rows, target_dir) if benchmark_rows else None

    return VisualizationOutputs(
        confusion_matrix=cm_path,
        class_distribution=dist_path,
        top_features=feat_path,
        learning_curve=curve_path,
        benchmark_chart=bench_path,
    )


def save_confusion_matrix_for_config(
    config: AppConfig,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    except ImportError:
        return None

    model = load_model(config.model.artifact_path)
    y_true = [str(label) for label in test_df[config.data.label_column]]
    y_pred = [str(label) for label in model.predict(test_df[config.data.text_column])]
    labels = sorted(set(y_true) | set(y_pred))

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    display.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix — {config.model.type}")
    fig.tight_layout()
    output_path = output_dir / "confusion_matrix.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_class_distribution(class_counts: dict[str, int], output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels = list(class_counts.keys())
    counts = [class_counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=["#2a9d8f", "#e76f51", "#264653", "#e9c46a"][: len(labels)])
    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Number of examples")
    ax.set_title("Class distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path = output_dir / "class_distribution.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_top_features(
    config: AppConfig,
    output_dir: Path,
    *,
    top_k: int = 15,
) -> Path | None:
    """Plot the strongest n-gram features per class for linear models."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    model = load_model(config.model.artifact_path)
    vectorizer = model.named_steps.get("tfidf") if hasattr(model, "named_steps") else None
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None
    if vectorizer is None or classifier is None or not hasattr(classifier, "coef_"):
        return None

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = classifier.coef_
    classes = list(classifier.classes_)

    if coefs.shape[0] == 1:
        # Binary linear model: positive coefficients favour classes[1].
        weights = coefs[0]
        top_pos_idx = np.argsort(weights)[-top_k:]
        top_neg_idx = np.argsort(weights)[:top_k]
        per_class = [
            (str(classes[0]), feature_names[top_neg_idx], -weights[top_neg_idx]),
            (str(classes[1]), feature_names[top_pos_idx], weights[top_pos_idx]),
        ]
    else:
        per_class = []
        for index, label in enumerate(classes):
            weights = coefs[index]
            top_idx = np.argsort(weights)[-top_k:]
            per_class.append((str(label), feature_names[top_idx], weights[top_idx]))

    fig, axes = plt.subplots(1, len(per_class), figsize=(6 * len(per_class), 5), squeeze=False)
    for ax, (label, names, weights) in zip(axes[0], per_class, strict=True):
        order = np.argsort(weights)
        ax.barh(np.array(names)[order], np.array(weights)[order], color="#2a9d8f")
        ax.set_title(f"Top features → {label}")
        ax.set_xlabel("Coefficient")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path = output_dir / "top_features.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_learning_curve(
    config: AppConfig,
    train_df: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.model_selection import learning_curve
    except ImportError:
        return None

    text_col = config.data.text_column
    label_col = config.data.label_column

    pipeline = build_model(config)
    cv_folds = _safe_cv_folds(train_df[label_col].tolist(), 5)
    # Suppress sklearn's FitFailedWarning that fires when a CV fold ends up
    # with a single class on micro datasets — the learning curve is still
    # generated from the folds that succeed.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        train_sizes, train_scores, val_scores = learning_curve(
            pipeline,
            train_df[text_col],
            train_df[label_col],
            cv=cv_folds,
            scoring="f1_macro",
            train_sizes=np.linspace(0.2, 1.0, 5),
            random_state=config.project.random_seed,
            error_score=float("nan"),
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train macro-F1")
    ax.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.15,
    )
    ax.plot(train_sizes, val_scores.mean(axis=1), marker="s", label="CV macro-F1")
    ax.fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.15,
    )
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Learning curve — {config.model.type}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path = output_dir / "learning_curve.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_benchmark_chart(rows: Iterable, output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    rows = list(rows)
    if not rows:
        return None

    model_types = [row.model_type for row in rows]
    macro_f1 = [row.macro_f1 for row in rows]
    cv_mean = [row.cv_macro_f1_mean for row in rows]
    cv_std = [row.cv_macro_f1_std for row in rows]

    x = np.arange(len(model_types))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, macro_f1, width, label="Test macro-F1", color="#2a9d8f")
    ax.bar(
        x + width / 2,
        cv_mean,
        width,
        yerr=cv_std,
        label="CV macro-F1 (±σ)",
        color="#e76f51",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_types, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("Model comparison")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path = output_dir / "benchmark.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _safe_cv_folds(labels: list, requested: int) -> int:
    counts: dict = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return 2
    return max(2, min(requested, min(counts.values())))
