from __future__ import annotations

import pytest

from nlp_project.config import load_config
from nlp_project.evaluate import evaluate_summarizers
from nlp_project.visualize import generate_all

matplotlib = pytest.importorskip("matplotlib")


def test_generate_all_produces_every_figure(write_config, tmp_path):
    config = load_config(write_config(), warn_placeholders=False)
    result = evaluate_summarizers(config)

    outputs = generate_all(
        config,
        result.per_method,
        result.qualitative_path,
        output_dir=tmp_path / "figures",
    )

    for name, path in outputs.__dict__.items():
        assert path is not None, f"figure {name} was not produced"
        assert path.exists()
        assert path.stat().st_size > 0
