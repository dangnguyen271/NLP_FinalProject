from __future__ import annotations

import pytest

from nlp_project.config import ConfigError, PlaceholderConfigWarning, load_config


def test_loads_example_config(repo_root):
    with pytest.warns(PlaceholderConfigWarning):
        config = load_config(repo_root / "config" / "project_config.example.yaml")

    assert config.project.title == "NLP Text Classification Project"
    assert config.project.task == "text_classification"
    assert config.data.text_column == "text"
    assert config.model.ngram_range == (1, 2)
    assert config.team.members


def test_required_top_level_sections_are_validated(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("project:\n  title: Missing sections\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Missing required top-level"):
        load_config(bad_config, warn_placeholders=False)


def test_placeholder_warnings_do_not_crash(repo_root):
    with pytest.warns(PlaceholderConfigWarning, match="placeholder"):
        config = load_config(repo_root / "config" / "project_config.yaml")

    assert "replace-with" in config.project.domain
