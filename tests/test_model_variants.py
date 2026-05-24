from __future__ import annotations

import pytest

from nlp_project.config import load_config
from nlp_project.model import (
    SUPPORTED_MODEL_TYPES,
    predict_proba_with_config,
    predict_texts_with_config,
    train_model,
)


@pytest.mark.parametrize("model_type", SUPPORTED_MODEL_TYPES)
def test_each_supported_model_trains_and_predicts(write_config, model_type):
    config = load_config(
        write_config({"model": {"type": model_type}}),
        warn_placeholders=False,
    )
    train_model(config)
    predictions = predict_texts_with_config(config, ["A clear lecture and helpful notes."])
    assert predictions and isinstance(predictions[0], str)


def test_predict_proba_returns_none_for_svm(write_config):
    config = load_config(
        write_config({"model": {"type": "tfidf_linear_svm"}}),
        warn_placeholders=False,
    )
    train_model(config)
    assert predict_proba_with_config(config, ["A clear lecture."]) is None


def test_predict_proba_works_for_probabilistic_models(write_config):
    config = load_config(
        write_config({"model": {"type": "tfidf_logistic_regression"}}),
        warn_placeholders=False,
    )
    train_model(config)
    probs = predict_proba_with_config(config, ["A clear lecture."])
    assert probs is not None
    assert len(probs) == 1
    assert pytest.approx(sum(probs[0].values()), abs=1e-6) == 1.0


def test_unknown_model_type_raises(write_config):
    with pytest.raises(Exception):
        # ConfigError validates type via the value being a non-empty string only,
        # so the error comes from build_classifier at train time.
        config = load_config(
            write_config({"model": {"type": "tfidf_xgboost"}}),
            warn_placeholders=False,
        )
        train_model(config)
