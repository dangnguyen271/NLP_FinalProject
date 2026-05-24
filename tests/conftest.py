from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_dataset_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "tiny_text_classification.csv"


@pytest.fixture
def write_config(tmp_path: Path, tiny_dataset_path: Path):
    def _write_config(overrides: dict | None = None) -> Path:
        config = {
            "project": {
                "title": "Fixture Text Classification Project",
                "task": "text_classification",
                "domain": "education",
                "application_area": "course project support",
                "github_repo_url": "https://github.com/example/private-nlp-project",
                "random_seed": 123,
            },
            "proposal": {
                "motivation": "Fast feedback on short text helps teams inspect project notes.",
                "problem_statement": "Manual sorting of short project notes is slow.",
                "expected_product": "A small text classification tool for demonstration.",
                "process_improvement": "It reduces manual review of simple text categories.",
                "research_questions": [
                    "RQ1: How well does a TF-IDF baseline classify the selected labels?",
                    "RQ2: Which errors reveal useful data-cleaning priorities?",
                ],
            },
            "data": {
                "path": str(tiny_dataset_path),
                "text_column": "text",
                "label_column": "label",
                "id_column": "id",
                "source": "Local fixture dataset for tests.",
                "provenance": "Created for deterministic offline testing.",
                "domain": "education",
                "challenges": [
                    "Small dataset size.",
                    "Short texts may lack enough context.",
                ],
            },
            "model": {
                "type": "tfidf_logistic_regression",
                "test_size": 0.25,
                "max_features": 1000,
                "ngram_range": [1, 2],
                "class_weight": "balanced",
                "artifact_path": str(tmp_path / "artifacts" / "model.joblib"),
            },
            "reports": {
                "directory": str(tmp_path / "reports"),
            },
            "team": {
                "members": [
                    {
                        "name": "Tester One",
                        "responsibilities": [
                            "Data collection and preprocessing",
                            "Dataset documentation",
                        ],
                    },
                    {
                        "name": "Tester Two",
                        "responsibilities": [
                            "Feature engineering and modelling",
                            "Evaluation and error analysis",
                        ],
                    },
                    {
                        "name": "Tester Three",
                        "responsibilities": [
                            "Visualisation and reporting",
                            "System/demo development",
                        ],
                    },
                ],
            },
        }
        if overrides:
            _deep_update(config, overrides)
        config_path = tmp_path / "project_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return config_path

    return _write_config


def _deep_update(target: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
