from __future__ import annotations

from nlp_project.config import load_config
from nlp_project.data import load_dataset, split_dataset
from nlp_project.evaluate import evaluate_model
from nlp_project.model import load_model, predict_texts, train_model


def test_trains_saves_loads_predicts_and_evaluates(write_config):
    config = load_config(write_config(), warn_placeholders=False)
    df = load_dataset(config)
    splits = split_dataset(df, config)

    model, artifact_path = train_model(config, splits.train)
    assert artifact_path.exists()

    loaded_model = load_model(artifact_path)
    direct_predictions = loaded_model.predict(
        ["Helpful instructions are useful.", "This is incomplete and risky."]
    )
    saved_predictions = predict_texts(
        ["Helpful instructions are useful.", "This is incomplete and risky."],
        artifact_path,
    )

    assert len(direct_predictions) == 2
    assert len(saved_predictions) == 2
    assert set(saved_predictions).issubset({"positive", "negative"})

    result = evaluate_model(config, model, splits.test)
    assert "accuracy" in result.metrics
    assert "macro_f1" in result.metrics
    assert result.metrics_path.exists()
    assert result.classification_report_path.exists()
    assert result.error_analysis_path.exists()
