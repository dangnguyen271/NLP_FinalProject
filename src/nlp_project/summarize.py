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


def _effective_n(requested: int, total_sentences: int, max_ratio: float = 0.6) -> int:
    """Cap the extractive output so every summary is shorter than its source.

    Forces at least one sentence dropped, even if the user asks for more
    sentences than the article contains, so the output is always a real summary
    rather than a verbatim copy of the input.
    """

    if total_sentences <= 1:
        return 1
    ratio_cap = max(1, int(round(total_sentences * max_ratio)))
    drop_one = max(1, total_sentences - 1)
    return max(1, min(requested, ratio_cap, drop_one))


def lead_n(article: str, n: int = 3) -> str:
    """Lead-n baseline: take the first n sentences (capped for compression)."""

    sentences = split_sentences(article)
    if not sentences:
        return ""
    effective = _effective_n(n, len(sentences))
    return " ".join(sentences[:effective])


def textrank(article: str, n: int = 3) -> str:
    """TextRank with sklearn TF-IDF cosine similarity + networkx PageRank."""

    sentences = split_sentences(article)
    if not sentences:
        return ""
    effective = _effective_n(n, len(sentences))
    if len(sentences) <= effective:
        return " ".join(sentences[:effective])

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # All sentences are stop-words only — degrade to lead-n.
        return lead_n(article, n=n)

    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)
    graph = nx.from_numpy_array(similarity)
    scores = nx.pagerank(graph, max_iter=200, tol=1e-4)

    ranked = sorted(range(len(sentences)), key=lambda idx: -scores[idx])
    chosen = sorted(ranked[:effective])  # restore original article order
    return " ".join(sentences[idx] for idx in chosen)


# --------------------------------------------------------------------------- #
# Abstractive (BART)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _load_bart_pipeline(model_name: str):
    """Load (tokenizer, model) for seq2seq summarisation.

    Uses the direct AutoTokenizer / AutoModelForSeq2SeqLM API rather than the
    high-level `pipeline()` helper because the `summarization` pipeline task
    was removed from transformers 5.x. This implementation is version-stable
    across transformers 4.x and 5.x.
    """

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised when extra is missing
        raise AbstractiveUnavailableError(
            "transformers is required for abstractive summarisation. "
            "Install with `pip install -e \".[transformer]\"`."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def bart_summarize(article: str, model_config: ModelConfig) -> str:
    """Run a BART (or distilbart) summariser on a single article."""

    tokenizer, model = _load_bart_pipeline(model_config.abstractive)

    inputs = tokenizer(
        article,
        max_length=model_config.max_input_tokens,
        truncation=True,
        return_tensors="pt",
    )

    import torch

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            min_length=model_config.min_summary_tokens,
            max_length=model_config.max_summary_tokens,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


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
