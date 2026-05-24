from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def summarize_metrics(metrics: dict[str, Any]) -> str:
    accuracy = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    weighted_f1 = metrics.get("weighted_f1")
    return (
        f"Accuracy: {accuracy:.3f}; "
        f"Macro F1: {macro_f1:.3f}; "
        f"Weighted F1: {weighted_f1:.3f}"
    )


def save_confusion_matrix(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: list[str],
    output_path: Path,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    except ImportError:
        return None

    matrix = confusion_matrix(list(y_true), list(y_pred), labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, colorbar=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
