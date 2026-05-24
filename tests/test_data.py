from __future__ import annotations

import pandas as pd
import pytest

from nlp_project.config import load_config
from nlp_project.data import (
    DataValidationError,
    load_dataset,
    prepare_dataset,
    split_dataset,
    validate_dataset,
)


def test_load_and_validate_fixture(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    df = load_dataset(config)
    summary = validate_dataset(df, config)
    assert summary.rows == 8
    assert summary.article_lengths["mean"] > summary.summary_lengths["mean"]


def test_validate_rejects_missing_columns(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    bad = pd.DataFrame({"article": ["only article"]})
    with pytest.raises(DataValidationError, match="missing required column"):
        validate_dataset(bad, config)


def test_validate_rejects_summary_longer_than_article(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    df = pd.DataFrame(
        {
            "article": ["short"],
            "highlights": ["this summary is clearly much longer than the article"],
        }
    )
    with pytest.raises(DataValidationError, match="summaries longer than"):
        validate_dataset(df, config)


def test_prepare_dataset_strips_html(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    df = pd.DataFrame(
        {
            "id": [1],
            "article": ["<p>Hello <b>world</b> &mdash; nice to see you.</p>"],
            "highlights": ["Hello world."],
        }
    )
    prepared = prepare_dataset(df, config)
    assert "<" not in prepared["article"].iloc[0]
    assert "Hello world" in prepared["article"].iloc[0]


def test_split_is_deterministic(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    df = load_dataset(config)
    first = split_dataset(df, config)
    second = split_dataset(df, config)
    assert list(first.train["id"]) == list(second.train["id"])
    assert list(first.test["id"]) == list(second.test["id"])
