from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "nlp_project.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


def test_cli_validate_data(repo_root, write_config):
    config_path = write_config()
    result = _run(["validate-data", "--config", str(config_path)], cwd=repo_root)
    assert "Validated" in result.stdout
    assert "Article tokens" in result.stdout


def test_cli_summarize_text(repo_root, write_config):
    config_path = write_config()
    article = (
        "The national space agency announced a new earth-observation satellite. "
        "It will be used for agriculture and disaster response. "
        "The first launch took place this week."
    )
    result = _run(
        [
            "summarize",
            "--config",
            str(config_path),
            "--method",
            "lead_3",
            "--text",
            article,
        ],
        cwd=repo_root,
    )
    assert "Method: lead_3" in result.stdout
    assert "satellite" in result.stdout


def test_cli_evaluate(repo_root, write_config):
    config_path = write_config()
    result = _run(["evaluate", "--config", str(config_path)], cwd=repo_root)
    assert "lead_3" in result.stdout
    assert "textrank" in result.stdout
