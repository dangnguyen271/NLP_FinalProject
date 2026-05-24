from __future__ import annotations

import json
from pathlib import Path

from nlp_project.benchmark import run_benchmark, summarize_benchmark
from nlp_project.config import load_config
from nlp_project.model import SUPPORTED_MODEL_TYPES


def test_benchmark_runs_every_supported_model(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    report = run_benchmark(config, cv_folds=2)

    types = [row.model_type for row in report.rows]
    assert types == list(SUPPORTED_MODEL_TYPES)
    for row in report.rows:
        assert 0.0 <= row.accuracy <= 1.0
        assert 0.0 <= row.macro_f1 <= 1.0
        assert row.train_seconds >= 0.0

    assert report.json_path.exists()
    assert report.csv_path.exists()
    parsed = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert {row["model_type"] for row in parsed} == set(SUPPORTED_MODEL_TYPES)


def test_summarize_benchmark_emits_one_row_per_model(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    report = run_benchmark(config, cv_folds=2)
    summary = summarize_benchmark(report.rows)
    for model_type in SUPPORTED_MODEL_TYPES:
        assert model_type in summary
