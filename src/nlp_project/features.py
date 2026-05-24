from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_project.config import ModelConfig


def build_vectorizer(model_config: ModelConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=model_config.max_features,
        ngram_range=model_config.ngram_range,
        lowercase=False,
    )
