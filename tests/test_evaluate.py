from __future__ import annotations

import json

import pandas as pd

from nlp_project.config import load_config
from nlp_project.evaluate import evaluate_summarizers, summarize_evaluation


def test_evaluate_writes_all_reports(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    result = evaluate_summarizers(config)

    assert result.metrics_path.exists()
    assert result.qualitative_path.exists()
    assert result.method_summary_path.exists()

    methods = {entry.method for entry in result.per_method}
    assert methods == {"lead_3", "textrank"}

    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload, "metrics JSON must not be empty"
    for entry in payload:
        for variant in ("rouge1", "rouge2", "rougeL"):
            assert variant in entry["rouge"]
            assert 0.0 <= entry["rouge"][variant] <= 1.0

    method_summary = pd.read_csv(result.method_summary_path)
    assert set(method_summary["method"]) == {"lead_3", "textrank"}

    summary_text = summarize_evaluation(result.per_method)
    assert "lead_3" in summary_text
    assert "textrank" in summary_text
