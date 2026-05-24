from __future__ import annotations

import pandas as pd
import pytest

from nlp_project.config import load_config
from nlp_project.data import DataValidationError, load_dataset, split_dataset, validate_dataset


def test_loads_fixture_dataset(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    df = load_dataset(config)
    summary = validate_dataset(df, config)

    assert len(df) == 12
    assert summary.class_counts == {"negative": 6, "positive": 6}
    assert config.data.text_column in summary.columns


def test_missing_text_column_raises_clear_error(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    df = pd.DataFrame({"label": ["positive", "negative"]})

    with pytest.raises(DataValidationError, match="missing required column"):
        validate_dataset(df, config)


def test_single_class_raises_clear_error(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    df = pd.DataFrame(
        {
            "text": ["one example", "another example"],
            "label": ["positive", "positive"],
        }
    )

    with pytest.raises(DataValidationError, match="at least two label classes"):
        validate_dataset(df, config)


def test_split_is_deterministic(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    df = load_dataset(config)

    first = split_dataset(df, config)
    second = split_dataset(df, config)

    assert first.train["id"].tolist() == second.train["id"].tolist()
    assert first.test["id"].tolist() == second.test["id"].tolist()
    assert set(first.train["label"]) == {"positive", "negative"}
