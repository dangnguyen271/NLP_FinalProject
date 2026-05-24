from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from nlp_project.config import AppConfig, DataConfig
from nlp_project.preprocess import normalize_text


class DataValidationError(ValueError):
    """Raised when the dataset cannot support the configured NLP task."""


@dataclass(frozen=True)
class DataValidationSummary:
    rows: int
    columns: tuple[str, ...]
    classes: tuple[str, ...]
    class_counts: dict[str, int]
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
    if data_config.path.suffix.lower() != ".csv":
        raise DataValidationError("The baseline implementation currently supports CSV data only.")
    return pd.read_csv(data_config.path)


def validate_dataset(df: pd.DataFrame, config: AppConfig | DataConfig) -> DataValidationSummary:
    data_config = _data_config(config)
    required_columns = [data_config.text_column, data_config.label_column]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise DataValidationError(f"Dataset is missing required column(s): {joined}")
    if df.empty:
        raise DataValidationError("Dataset is empty.")

    text_values = df[data_config.text_column].map(_clean_cell)
    empty_text = text_values[text_values == ""]
    if not empty_text.empty:
        first_rows = ", ".join(str(index) for index in empty_text.index[:5])
        raise DataValidationError(
            f"Text column '{data_config.text_column}' contains empty values at row(s): "
            f"{first_rows}"
        )

    label_values = df[data_config.label_column].map(_clean_cell)
    empty_label = label_values[label_values == ""]
    if not empty_label.empty:
        first_rows = ", ".join(str(index) for index in empty_label.index[:5])
        raise DataValidationError(
            f"Label column '{data_config.label_column}' contains empty values at row(s): "
            f"{first_rows}"
        )

    class_counts = label_values.value_counts().sort_index()
    if len(class_counts) < 2:
        raise DataValidationError("Classification requires at least two label classes.")

    duplicate_rows = int(df.duplicated(subset=required_columns).sum())
    if duplicate_rows:
        warnings.warn(
            f"Dataset contains {duplicate_rows} duplicated text/label row(s).",
            UserWarning,
            stacklevel=2,
        )

    return DataValidationSummary(
        rows=len(df),
        columns=tuple(str(column) for column in df.columns),
        classes=tuple(str(label) for label in class_counts.index),
        class_counts={str(label): int(count) for label, count in class_counts.items()},
        duplicate_rows=duplicate_rows,
    )


def load_and_validate_dataset(config: AppConfig) -> tuple[pd.DataFrame, DataValidationSummary]:
    df = load_dataset(config)
    summary = validate_dataset(df, config)
    return df, summary


def prepare_dataset(df: pd.DataFrame, config: AppConfig | DataConfig) -> pd.DataFrame:
    data_config = _data_config(config)
    validate_dataset(df, data_config)
    prepared = df.copy()
    prepared[data_config.text_column] = prepared[data_config.text_column].map(normalize_text)
    prepared[data_config.label_column] = prepared[data_config.label_column].map(_clean_cell)
    return prepared


def split_dataset(df: pd.DataFrame, config: AppConfig) -> DatasetSplits:
    prepared = prepare_dataset(df, config)
    data_config = config.data

    if "split" in prepared.columns:
        explicit = _split_from_column(prepared)
        if explicit is not None:
            return explicit

    labels = prepared[data_config.label_column]
    if _can_use_stratified_split(labels, config.model.test_size):
        train_df, test_df = train_test_split(
            prepared,
            test_size=config.model.test_size,
            random_state=config.project.random_seed,
            stratify=labels,
        )
    else:
        train_df, test_df = _fallback_split(
            prepared,
            label_column=data_config.label_column,
            test_size=config.model.test_size,
            seed=config.project.random_seed,
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


def _can_use_stratified_split(labels: pd.Series, test_size: float) -> bool:
    class_counts = labels.value_counts()
    n_classes = len(class_counts)
    n_rows = len(labels)
    n_test = ceil(n_rows * test_size)
    n_train = n_rows - n_test
    return bool(class_counts.min() >= 2 and n_test >= n_classes and n_train >= n_classes)


def _fallback_split(
    df: pd.DataFrame,
    *,
    label_column: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed)
    required_train_index = shuffled.groupby(label_column, sort=False).head(1).index
    remaining = shuffled.drop(index=required_train_index)

    if remaining.empty:
        return shuffled.copy(), shuffled.copy()

    target_test_rows = max(1, round(len(shuffled) * test_size))
    target_test_rows = min(target_test_rows, len(remaining))
    test_index = remaining.head(target_test_rows).index
    train_df = shuffled.drop(index=test_index)
    test_df = shuffled.loc[test_index]
    return train_df, test_df


def dataset_path_exists(path: str | Path) -> bool:
    return Path(path).exists()
