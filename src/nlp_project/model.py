from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from nlp_project.config import AppConfig
from nlp_project.data import DataValidationError, load_dataset, split_dataset
from nlp_project.features import build_vectorizer
from nlp_project.preprocess import TextNormalizer


DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")


def build_model(config: AppConfig) -> Pipeline:
    if config.model.type != "tfidf_logistic_regression":
        raise ValueError(
            "Only model.type='tfidf_logistic_regression' is implemented for the baseline."
        )

    classifier = LogisticRegression(
        class_weight=config.model.class_weight,
        max_iter=1000,
        random_state=config.project.random_seed,
        solver="liblinear",
    )
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


def _ensure_text_list(texts: Iterable[str]) -> list[str]:
    return ["" if text is None else str(text) for text in texts]
