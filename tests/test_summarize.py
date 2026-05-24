from __future__ import annotations

import pytest

from nlp_project.config import load_config
from nlp_project.summarize import available_methods, lead_n, summarize, textrank


ARTICLE = (
    "Researchers at a university announced a new lightweight model on Monday. "
    "The model was trained on a large corpus of tutoring dialogues. "
    "It runs in real time on a single consumer GPU. "
    "The team plans to release the weights under a permissive licence. "
    "A future version will add support for multimodal inputs."
)


def test_lead_n_returns_first_sentences():
    summary = lead_n(ARTICLE, n=2)
    assert summary.startswith("Researchers at a university")
    # second sentence should be present
    assert "tutoring dialogues" in summary


def test_textrank_returns_valid_subset():
    summary = textrank(ARTICLE, n=2)
    sentence_count = summary.count(".")
    assert 1 <= sentence_count <= 2
    # Output must come from the source article
    for fragment in summary.split(". "):
        if fragment.strip():
            assert fragment.split()[0] in ARTICLE


@pytest.mark.parametrize("method", ["lead_3", "textrank"])
def test_summarize_dispatch(write_config, method):
    config = load_config(write_config(), warn_placeholders=False)
    result = summarize(method, ARTICLE, config)
    assert result.method == method
    assert result.summary
    assert result.elapsed_seconds >= 0.0


def test_available_methods_excludes_bart_when_disabled(write_config):
    config = load_config(
        write_config({"model": {"use_abstractive": False}}), warn_placeholders=False
    )
    methods = available_methods(config)
    assert "lead_3" in methods
    assert "textrank" in methods
    assert "bart" not in methods


def test_summarize_unknown_method_raises(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    with pytest.raises(ValueError, match="Unsupported summarisation method"):
        summarize("does_not_exist", ARTICLE, config)
