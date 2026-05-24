from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.config import load_config  # noqa: E402
from nlp_project.proposal import render_proposal_pdf, write_proposal  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Render proposal.md to proposal.pdf")
    parser.add_argument("--config", default="config/project_config.yaml")
    parser.add_argument("--proposal", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    proposal_path = Path(args.proposal) if args.proposal else config.repo_root / "proposal.md"
    if not proposal_path.is_absolute():
        proposal_path = config.repo_root / proposal_path
    if not proposal_path.exists():
        proposal_path = write_proposal(config, proposal_path)

    output_path = Path(args.output) if args.output else config.repo_root / "proposal.pdf"
    if not output_path.is_absolute():
        output_path = config.repo_root / output_path

    rendered = render_proposal_pdf(proposal_path, output_path)
    if rendered is None:
        return 0
    print(f"Proposal PDF written to {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
