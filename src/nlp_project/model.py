from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nlp_project.config import AppConfig, ModelConfig
from nlp_project.data import DataValidationError, load_dataset, split_dataset
from nlp_project.features import build_vectorizer
from nlp_project.preprocess import TextNormalizer


DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")

SUPPORTED_MODEL_TYPES = (
    "tfidf_logistic_regression",
    "tfidf_naive_bayes",
    "tfidf_linear_svm",
)


def build_classifier(model_config: ModelConfig, random_seed: int):
    """Construct the bare classifier matching ``model_config.type``."""

    model_type = model_config.type
    if model_type == "tfidf_logistic_regression":
        return LogisticRegression(
            class_weight=model_config.class_weight,
            max_iter=1000,
            random_state=random_seed,
            solver="liblinear",
        )
    if model_type == "tfidf_naive_bayes":
        # MultinomialNB has no class_weight, so we let sample_weight handle imbalance
        # downstream if needed. alpha=1 is Laplace smoothing — a standard default.
        return MultinomialNB(alpha=1.0)
    if model_type == "tfidf_linear_svm":
        return LinearSVC(
            C=1.0,
            class_weight=model_config.class_weight,
            random_state=random_seed,
        )
    raise ValueError(
        f"Unsupported model.type={model_type!r}. "
        f"Choose one of: {', '.join(SUPPORTED_MODEL_TYPES)}."
    )


def build_model(config: AppConfig) -> Pipeline:
    classifier = build_classifier(config.model, config.project.random_seed)
    return Pipeline(
        steps=[
            ("normalize", TextNormalizer(lowercase=True, remove_punctuation=False)),
            ("tfidf", build_vectorizer(config.model)),
            ("classifier", classifier),
        ]
    )


def train_model(
    config: AppConfig,
    train_df: pd.DataFrame | None = None,
) -> tuple[Pipeline, Path]:
    if train_df is None:
        df = load_dataset(config)
        train_df = split_dataset(df, config).train

    labels = train_df[config.data.label_column]
    if labels.nunique() < 2:
        raise DataValidationError("Training split must contain at least two classes.")

    model = build_model(config)
    model.fit(train_df[config.data.text_column], labels)

    config.model.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.model.artifact_path)
    return model, config.model.artifact_path


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> Pipeline:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def predict_texts(
    texts: list[str],
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> list[str]:
    model = load_model(model_path)
    return [str(label) for label in model.predict(_ensure_text_list(texts))]


def predict_texts_with_config(config: AppConfig, texts: Iterable[str]) -> list[str]:
    model = load_model(config.model.artifact_path)
    return [str(label) for label in model.predict(_ensure_text_list(texts))]


def predict_proba_with_config(
    config: AppConfig, texts: Iterable[str]
) -> list[dict[str, float]] | None:
    """Return per-class probability dicts when the underlying model supports it.

    Linear SVMs do not provide native probabilities; we fall back to ``None`` so
    callers (e.g. the Streamlit app) can degrade gracefully.
    """

    model = load_model(config.model.artifact_path)
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None
    if classifier is None or not hasattr(classifier, "predict_proba"):
        return None

    text_list = _ensure_text_list(texts)
    probs = model.predict_proba(text_list)
    classes = [str(label) for label in classifier.classes_]
    return [
        {label: float(score) for label, score in zip(classes, row, strict=True)}
        for row in probs
    ]


def _ensure_text_list(texts: Iterable[str]) -> list[str]:
    return ["" if text is None else str(text) for text in texts]
