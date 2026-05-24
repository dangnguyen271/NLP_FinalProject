"""Summarisation methods: Lead-3 baseline, TextRank, and BART (abstractive)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

from nlp_project.config import AppConfig, ModelConfig
from nlp_project.preprocess import split_sentences


SUPPORTED_METHODS = ("lead_3", "textrank", "bart")


@dataclass(frozen=True)
class SummaryResult:
    method: str
    summary: str
    elapsed_seconds: float


class AbstractiveUnavailableError(RuntimeError):
    """Raised when the user asks for BART but transformers/torch are not installed."""


def summarize(method: str, article: str, config: AppConfig) -> SummaryResult:
    """Single-document summarisation dispatch."""

    import time

    article = (article or "").strip()
    if not article:
        return SummaryResult(method=method, summary="", elapsed_seconds=0.0)

    start = time.perf_counter()
    if method == "lead_3":
        summary = lead_n(article, n=config.model.num_sentences_extractive)
    elif method == "textrank":
        summary = textrank(article, n=config.model.num_sentences_extractive)
    elif method == "bart":
        summary = bart_summarize(article, config.model)
    else:
        raise ValueError(
            f"Unsupported summarisation method {method!r}. "
            f"Choose one of: {', '.join(SUPPORTED_METHODS)}."
        )
    return SummaryResult(
        method=method,
        summary=summary,
        elapsed_seconds=float(time.perf_counter() - start),
    )


def summarize_batch(
    method: str, articles: Iterable[str], config: AppConfig
) -> list[SummaryResult]:
    return [summarize(method, article, config) for article in articles]


# --------------------------------------------------------------------------- #
# Extractive baselines
# --------------------------------------------------------------------------- #


def lead_n(article: str, n: int = 3) -> str:
    """The single most reliable baseline in news summarisation: take the first n sentences."""

    sentences = split_sentences(article)
    if not sentences:
        return ""
    return " ".join(sentences[: max(1, n)])


def textrank(article: str, n: int = 3) -> str:
    """TextRank with sklearn TF-IDF cosine similarity + networkx PageRank."""

    sentences = split_sentences(article)
    if not sentences:
        return ""
    if len(sentences) <= n:
        return " ".join(sentences)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # All sentences are stop-words only — degrade to lead-n
        return lead_n(article, n=n)

    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)
    graph = nx.from_numpy_array(similarity)
    scores = nx.pagerank(graph, max_iter=200, tol=1e-4)

    ranked = sorted(range(len(sentences)), key=lambda idx: -scores[idx])
    chosen = sorted(ranked[:n])  # restore original article order
    return " ".join(sentences[idx] for idx in chosen)


# --------------------------------------------------------------------------- #
# Abstractive (BART)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _load_bart_pipeline(model_name: str):
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - exercised when extra is missing
        raise AbstractiveUnavailableError(
            "transformers is required for abstractive summarisation. "
            "Install with `pip install -e \".[transformer]\"`."
        ) from exc
    return pipeline("summarization", model=model_name)


def bart_summarize(article: str, model_config: ModelConfig) -> str:
    """Wrap Hugging Face's summarization pipeline with project config defaults."""

    pipe = _load_bart_pipeline(model_config.abstractive)
    # The HF pipeline truncates by default but takes word count for min/max length.
    output = pipe(
        article,
        min_length=model_config.min_summary_tokens,
        max_length=model_config.max_summary_tokens,
        do_sample=False,
        truncation=True,
    )
    return str(output[0]["summary_text"]).strip()


def available_methods(config: AppConfig) -> tuple[str, ...]:
    """Return the methods that can run on the current environment."""

    methods = ["lead_3", "textrank"]
    if config.model.use_abstractive:
        try:  # pragma: no cover - exercised when extra is installed
            import transformers  # noqa: F401
            methods.append("bart")
        except ImportError:
            pass
    return tuple(methods)
