"""Report figures for the NewsDigest summarisation project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nlp_project.config import AppConfig
from nlp_project.data import load_and_validate_dataset, prepare_dataset
from nlp_project.evaluate import MethodEvaluation


@dataclass(frozen=True)
class VisualizationOutputs:
    rouge_comparison: Path | None
    article_length_distribution: Path | None
    summary_length_distribution: Path | None
    compression_ratio: Path | None
    per_example_rouge: Path | None


def generate_all(
    config: AppConfig,
    per_method: list[MethodEvaluation],
    qualitative_csv: Path,
    *,
    output_dir: Path | None = None,
) -> VisualizationOutputs:
    target_dir = output_dir or (config.reports_dir / "figures")
    target_dir.mkdir(parents=True, exist_ok=True)

    df, _ = load_and_validate_dataset(config)
    prepared = prepare_dataset(df, config)

    rouge_path = save_rouge_comparison(per_method, target_dir)
    article_path = save_length_distribution(
        prepared[config.data.text_column],
        title="Article length (tokens)",
        output_path=target_dir / "article_length.png",
    )
    summary_path = save_length_distribution(
        prepared[config.data.summary_column],
        title="Reference-summary length (tokens)",
        output_path=target_dir / "summary_length.png",
    )
    compression_path = save_compression_ratio(per_method, target_dir)
    qual = pd.read_csv(qualitative_csv) if qualitative_csv.exists() else None
    per_example_path = save_per_example_rouge(qual, target_dir) if qual is not None else None

    return VisualizationOutputs(
        rouge_comparison=rouge_path,
        article_length_distribution=article_path,
        summary_length_distribution=summary_path,
        compression_ratio=compression_path,
        per_example_rouge=per_example_path,
    )


def save_rouge_comparison(
    per_method: list[MethodEvaluation], output_dir: Path
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not per_method:
        return None

    variants = list(per_method[0].rouge.keys())
    methods = [entry.method for entry in per_method]
    matrix = np.array([[entry.rouge.get(v, 0.0) for v in variants] for entry in per_method])

    x = np.arange(len(variants))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a"]
    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, matrix[index], width, label=method, color=colors[index % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("ROUGE F1")
    ax.set_ylim(0, max(0.5, matrix.max() * 1.1))
    ax.set_title("ROUGE comparison across summarisation methods")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path = output_dir / "rouge_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_length_distribution(
    series: pd.Series, *, title: str, output_path: Path
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    token_counts = series.fillna("").map(lambda value: len(str(value).split()))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(token_counts, bins=20, color="#2a9d8f", edgecolor="white")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Articles")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_compression_ratio(
    per_method: list[MethodEvaluation], output_dir: Path
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not per_method:
        return None

    methods = [entry.method for entry in per_method]
    ratios = [entry.mean_compression_ratio for entry in per_method]
    lengths = [entry.mean_summary_length for entry in per_method]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(methods, ratios, color="#264653")
    for bar, length in zip(bars, lengths, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{length:.0f} tok",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Mean compression ratio (summary / article tokens)")
    ax.set_title("Summary compression across methods")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path = output_dir / "compression_ratio.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_per_example_rouge(qualitative: pd.DataFrame, output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if qualitative is None or qualitative.empty:
        return None
    if "rouge1_f1" not in qualitative.columns:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    for method, group in qualitative.groupby("method"):
        sorted_scores = np.sort(group["rouge1_f1"].values)[::-1]
        ax.plot(np.arange(1, len(sorted_scores) + 1), sorted_scores, label=method)
    ax.set_xlabel("Test example (sorted by ROUGE-1 F1)")
    ax.set_ylabel("ROUGE-1 F1")
    ax.set_title("Per-example ROUGE-1 distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path = output_dir / "per_example_rouge.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
