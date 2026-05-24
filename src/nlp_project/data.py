"""Data loading, validation, and splitting for the NewsDigest summarisation project."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from nlp_project.config import AppConfig, DataConfig
from nlp_project.preprocess import normalize_article, strip_html


class DataValidationError(ValueError):
    """Raised when the dataset cannot support the configured summarisation task."""


@dataclass(frozen=True)
class DataValidationSummary:
    rows: int
    columns: tuple[str, ...]
    sources: tuple[str, ...]
    article_lengths: dict[str, float]
    summary_lengths: dict[str, float]
    duplicate_rows: int


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    test: pd.DataFrame
    validation: pd.DataFrame | None = None


def load_dataset(config: AppConfig | DataConfig) -> pd.DataFrame:
    data_config = _data_config(config)
    if not data_config.path.exists():
        raise DataValidationError(f"Dataset file does not exist: {data_config.path}")
    if data_config.path.suffix.lower() not in {".csv", ".tsv", ".parquet", ".jsonl"}:
        raise DataValidationError(
            "Supported formats are CSV, TSV, Parquet, and JSONL."
        )
    if data_config.path.suffix.lower() == ".parquet":
        return pd.read_parquet(data_config.path)
    if data_config.path.suffix.lower() == ".jsonl":
        return pd.read_json(data_config.path, lines=True)
    sep = "\t" if data_config.path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(data_config.path, sep=sep)


def validate_dataset(df: pd.DataFrame, config: AppConfig | DataConfig) -> DataValidationSummary:
    data_config = _data_config(config)
    required = [data_config.text_column, data_config.summary_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(
            f"Dataset is missing required column(s): {', '.join(missing)}"
        )
    if df.empty:
        raise DataValidationError("Dataset is empty.")

    article_values = df[data_config.text_column].map(_clean_cell)
    empty_articles = article_values[article_values == ""]
    if not empty_articles.empty:
        first_rows = ", ".join(str(index) for index in empty_articles.index[:5])
        raise DataValidationError(
            f"Article column '{data_config.text_column}' contains empty rows at: {first_rows}"
        )

    summary_values = df[data_config.summary_column].map(_clean_cell)
    empty_summaries = summary_values[summary_values == ""]
    if not empty_summaries.empty:
        first_rows = ", ".join(str(index) for index in empty_summaries.index[:5])
        raise DataValidationError(
            f"Summary column '{data_config.summary_column}' contains empty rows at: {first_rows}"
        )

    article_lengths = article_values.map(lambda text: len(text.split()))
    summary_lengths = summary_values.map(lambda text: len(text.split()))
    if (article_lengths < summary_lengths).any():
        bad = df.index[article_lengths < summary_lengths].tolist()[:5]
        raise DataValidationError(
            f"Some rows have summaries longer than articles (rows: {bad}). "
            "This usually indicates swapped columns."
        )

    duplicate_rows = int(df.duplicated(subset=required).sum())
    if duplicate_rows:
        warnings.warn(
            f"Dataset contains {duplicate_rows} duplicated article/summary row(s).",
            UserWarning,
            stacklevel=2,
        )

    sources: tuple[str, ...]
    if data_config.source_column and data_config.source_column in df.columns:
        sources = tuple(sorted({str(s) for s in df[data_config.source_column].dropna()}))
    else:
        sources = ()

    return DataValidationSummary(
        rows=len(df),
        columns=tuple(str(column) for column in df.columns),
        sources=sources,
        article_lengths={
            "mean": float(article_lengths.mean()),
            "median": float(article_lengths.median()),
            "min": int(article_lengths.min()),
            "max": int(article_lengths.max()),
        },
        summary_lengths={
            "mean": float(summary_lengths.mean()),
            "median": float(summary_lengths.median()),
            "min": int(summary_lengths.min()),
            "max": int(summary_lengths.max()),
        },
        duplicate_rows=duplicate_rows,
    )


def load_and_validate_dataset(config: AppConfig) -> tuple[pd.DataFrame, DataValidationSummary]:
    df = load_dataset(config)
    summary = validate_dataset(df, config)
    return df, summary


def prepare_dataset(df: pd.DataFrame, config: AppConfig | DataConfig) -> pd.DataFrame:
    """Strip HTML / boilerplate from articles + normalise whitespace on both columns.

    Deliberately conservative: we do NOT lowercase or remove punctuation, since
    summarisation models rely on capitalisation and punctuation for fluency.
    """

    data_config = _data_config(config)
    validate_dataset(df, data_config)
    prepared = df.copy()
    prepared[data_config.text_column] = prepared[data_config.text_column].map(
        lambda value: normalize_article(strip_html(value))
    )
    prepared[data_config.summary_column] = prepared[data_config.summary_column].map(
        lambda value: normalize_article(strip_html(value))
    )
    return prepared


def split_dataset(
    df: pd.DataFrame, config: AppConfig, *, test_size: float = 0.25
) -> DatasetSplits:
    prepared = prepare_dataset(df, config)

    if "split" in prepared.columns:
        explicit = _split_from_column(prepared)
        if explicit is not None:
            return explicit

    n_rows = len(prepared)
    if n_rows < 4:
        # Too small to split meaningfully; reuse everything for both halves.
        return DatasetSplits(train=prepared.reset_index(drop=True),
                             test=prepared.reset_index(drop=True))

    train_df, test_df = train_test_split(
        prepared,
        test_size=test_size,
        random_state=config.project.random_seed,
        shuffle=True,
    )
    return DatasetSplits(
        train=train_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def _data_config(config: AppConfig | DataConfig) -> DataConfig:
    return config.data if isinstance(config, AppConfig) else config


def _clean_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _split_from_column(df: pd.DataFrame) -> DatasetSplits | None:
    split_values = df["split"].map(lambda value: str(value).strip().lower())
    train_df = df[split_values == "train"]
    test_df = df[split_values == "test"]
    validation_df = df[split_values.isin({"validation", "val", "dev"})]
    if train_df.empty or test_df.empty:
        return None
    return DatasetSplits(
        train=train_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        validation=validation_df.reset_index(drop=True) if not validation_df.empty else None,
    )
