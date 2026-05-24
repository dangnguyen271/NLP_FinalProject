from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_news_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "tiny_news.csv"


@pytest.fixture
def write_config(tmp_path: Path, tiny_news_path: Path):
    def _write_config(overrides: dict | None = None) -> Path:
        config = {
            "project": {
                "title": "Fixture NewsDigest Project",
                "task": "text_summarization",
                "domain": "digital news media",
                "application_area": "automatic news summarisation",
                "github_repo_url": "https://github.com/example/newsdigest",
                "random_seed": 123,
            },
            "proposal": {
                "motivation": "News volume makes manual scanning slow.",
                "problem_statement": "Generate short faithful summaries of news.",
                "expected_product": "A small prototype summariser with web demo.",
                "process_improvement": "Reduces time spent reading full articles.",
                "research_questions": [
                    "RQ1: How well can a Lead-3 baseline summarise news?",
                    "RQ2: Does TextRank materially improve ROUGE over Lead-3?",
                ],
            },
            "data": {
                "path": str(tiny_news_path),
                "text_column": "article",
                "summary_column": "highlights",
                "id_column": "id",
                "source_column": "source",
                "source": "Local fixture dataset for tests.",
                "provenance": "Created for deterministic offline testing.",
                "domain": "test",
                "challenges": [
                    "Small dataset size.",
                    "Short articles limit summarisation quality.",
                ],
            },
            "model": {
                "baseline": "lead_3",
                "extractive": "textrank",
                "abstractive": "facebook/bart-large-cnn",
                "max_input_tokens": 1024,
                "min_summary_tokens": 20,
                "max_summary_tokens": 80,
                "num_sentences_extractive": 2,
                "artifact_dir": str(tmp_path / "artifacts"),
                "use_abstractive": False,
            },
            "evaluation": {
                "rouge_variants": ["rouge1", "rouge2", "rougeL"],
                "max_examples": 8,
            },
            "scrape": {
                "user_agent": "NewsDigestTest/0.1",
                "request_timeout_seconds": 5,
            },
            "reports": {
                "directory": str(tmp_path / "reports"),
            },
            "team": {
                "members": [
                    {
                        "name": "Tester One",
                        "email": "one@example.com",
                        "responsibilities": ["Data collection", "Web scraping"],
                    },
                    {
                        "name": "Tester Two",
                        "email": "two@example.com",
                        "responsibilities": ["Preprocessing", "EDA"],
                    },
                    {
                        "name": "Tester Three",
                        "email": "three@example.com",
                        "responsibilities": ["Summarisation modelling"],
                    },
                    {
                        "name": "Tester Four",
                        "email": "four@example.com",
                        "responsibilities": ["Evaluation", "Streamlit demo"],
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
