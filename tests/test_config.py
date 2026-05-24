from __future__ import annotations

import pytest

from nlp_project.config import ConfigError, load_config


def test_loads_example_config(repo_root):
    config = load_config(
        repo_root / "config" / "project_config.example.yaml", warn_placeholders=False
    )

    assert config.project.task == "text_summarization"
    assert config.project.title.startswith("NewsDigest")
    assert config.data.text_column == "article"
    assert config.data.summary_column == "highlights"
    assert len(config.team.members) == 4


def test_required_top_level_sections_are_validated(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("project:\n  title: Missing sections\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Missing required top-level"):
        load_config(bad_config, warn_placeholders=False)


def test_default_config_has_real_team_names(repo_root):
    config = load_config(
        repo_root / "config" / "project_config.yaml", warn_placeholders=False
    )
    names = [member.name for member in config.team.members]
    # Team names from the proposal PDF should already be filled in.
    assert "Nguyen Hoang Hieu (Ethan)" in names
    assert "Thai Ba Hung" in names
    assert "Nguyen Quoc Dang" in names
    assert "Le Nguyen Gia Binh" in names
