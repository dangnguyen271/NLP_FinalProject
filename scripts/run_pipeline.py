from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.cli import main as cli_main  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full NLP project pipeline")
    parser.add_argument("--config", default="config/project_config.yaml")
    args = parser.parse_args()
    return cli_main(["run-all", "--config", args.config])


if __name__ == "__main__":
    raise SystemExit(main())
