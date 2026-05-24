from __future__ import annotations

from nlp_project.config import load_config
from nlp_project.proposal import required_proposal_sections, write_proposal


def test_generates_required_proposal_sections(write_config, tmp_path):
    config = load_config(write_config(), warn_placeholders=False)
    proposal_path = write_proposal(config, tmp_path / "proposal.md")
    text = proposal_path.read_text(encoding="utf-8")

    for section in required_proposal_sections():
        assert f"## {section}" in text
    assert "https://github.com/example/private-nlp-project" in text
    assert "drelhaj" in text
    assert "whistle-hikhi" in text


def test_data_readme_contains_assignment_required_sections(repo_root):
    text = (repo_root / "data" / "README.md").read_text(encoding="utf-8")

    for heading in (
        "# Dataset Description",
        "## Source and provenance",
        "## Size and structure",
        "## Domain",
        "## Suitability for the NLP task",
        "## Known challenges",
        "## Replacement instructions",
    ):
        assert heading in text
