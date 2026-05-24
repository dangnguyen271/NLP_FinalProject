from __future__ import annotations

import os
import subprocess
import sys


def test_cli_validate_train_evaluate_and_predict(write_config, repo_root):
    config_path = write_config()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"

    validate = _run_cli(repo_root, env, "validate-data", "--config", str(config_path))
    assert validate.returncode == 0, validate.stderr
    assert "Validated" in validate.stdout

    train = _run_cli(repo_root, env, "train", "--config", str(config_path))
    assert train.returncode == 0, train.stderr
    assert "Model written" in train.stdout

    evaluate = _run_cli(repo_root, env, "evaluate", "--config", str(config_path))
    assert evaluate.returncode == 0, evaluate.stderr
    assert "Metrics written" in evaluate.stdout

    predict = _run_cli(
        repo_root,
        env,
        "predict",
        "--config",
        str(config_path),
        "--text",
        "This example is useful and clear.",
    )
    assert predict.returncode == 0, predict.stderr
    assert "Predicted label:" in predict.stdout


def _run_cli(repo_root, env, *args):
    return subprocess.run(
        [sys.executable, "-m", "nlp_project.cli", *args],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
