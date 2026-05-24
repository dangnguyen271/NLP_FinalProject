"""NewsDigest command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from nlp_project.config import ConfigError, load_config
from nlp_project.data import (
    DataValidationError,
    load_and_validate_dataset,
    split_dataset,
)
from nlp_project.evaluate import evaluate_summarizers, summarize_evaluation
from nlp_project.proposal import render_proposal_pdf, write_proposal
from nlp_project.scraper import ScrapeError, fetch_article
from nlp_project.summarize import (
    SUPPORTED_METHODS,
    AbstractiveUnavailableError,
    available_methods,
    summarize,
)
from nlp_project.visualize import generate_all as generate_visualizations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NewsDigest CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in (
        "validate-data",
        "evaluate",
        "visualize",
        "generate-proposal",
        "run-all",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="config/project_config.yaml")
        command.set_defaults(func=_COMMANDS[name])

    summarize_cmd = subparsers.add_parser("summarize")
    summarize_cmd.add_argument("--config", default="config/project_config.yaml")
    summarize_cmd.add_argument("--text")
    summarize_cmd.add_argument("--file")
    summarize_cmd.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default="lead_3",
        help="Which summarisation method to apply (default: lead_3).",
    )
    summarize_cmd.set_defaults(func=_summarize)

    scrape_cmd = subparsers.add_parser("scrape")
    scrape_cmd.add_argument("--config", default="config/project_config.yaml")
    scrape_cmd.add_argument("--url", required=True)
    scrape_cmd.add_argument("--method", choices=SUPPORTED_METHODS, default="lead_3")
    scrape_cmd.add_argument("--show-article", action="store_true",
                            help="Print the cleaned article body before the summary.")
    scrape_cmd.set_defaults(func=_scrape)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (
        ConfigError,
        DataValidationError,
        FileNotFoundError,
        ValueError,
        ScrapeError,
        AbstractiveUnavailableError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _validate_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, summary = load_and_validate_dataset(config)
    print(
        f"Validated {summary.rows} rows from {config.data.path}; "
        f"sources: {len(summary.sources)}; duplicates: {summary.duplicate_rows}"
    )
    print(
        f"Article tokens — mean {summary.article_lengths['mean']:.0f}, "
        f"median {summary.article_lengths['median']:.0f}, "
        f"range [{summary.article_lengths['min']}, {summary.article_lengths['max']}]"
    )
    print(
        f"Summary tokens — mean {summary.summary_lengths['mean']:.0f}, "
        f"median {summary.summary_lengths['median']:.0f}, "
        f"range [{summary.summary_lengths['min']}, {summary.summary_lengths['max']}]"
    )


def _summarize(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.text and args.file:
        raise ValueError("Pass --text OR --file, not both.")
    if args.file:
        article = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        article = args.text
    else:
        raise ValueError("Pass --text \"...\" or --file path/to/article.txt.")

    result = summarize(args.method, article, config)
    print(f"# Method: {result.method} ({result.elapsed_seconds:.3f}s)")
    print(result.summary)


def _scrape(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    scraped = fetch_article(args.url, config.scrape)
    if args.show_article:
        print(f"# Title: {scraped.title}")
        print(f"# Domain: {scraped.domain}")
        print(scraped.article)
        print()
    result = summarize(args.method, scraped.article, config)
    print(f"# Summary ({result.method}, {result.elapsed_seconds:.3f}s)")
    print(result.summary)


def _evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    methods = available_methods(config)
    result = evaluate_summarizers(config, methods=methods)
    print(summarize_evaluation(result.per_method))
    print(f"\nMetrics JSON:      {result.metrics_path}")
    print(f"Per-row qualitative: {result.qualitative_path}")
    print(f"Method summary CSV:  {result.method_summary_path}")


def _visualize(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    metrics_path = config.reports_dir / "rouge_metrics.json"
    qualitative_path = config.reports_dir / "qualitative_review.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run `python -m nlp_project.cli evaluate` first."
        )
    from nlp_project.evaluate import MethodEvaluation
    per_method = [
        MethodEvaluation(**entry) for entry in json.loads(metrics_path.read_text())
    ]
    outputs = generate_visualizations(config, per_method, qualitative_path)
    for name, path in outputs.__dict__.items():
        if path is None:
            print(f"{name}: skipped (matplotlib unavailable or no data)")
        else:
            print(f"{name}: {path}")


def _generate_proposal(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    proposal_path = write_proposal(config)
    print(f"Proposal written to {proposal_path}")


def _run_all(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    df, summary = load_and_validate_dataset(config)
    splits = split_dataset(df, config)

    eval_result = evaluate_summarizers(
        config,
        methods=available_methods(config),
        test_df=splits.test,
    )

    metrics_path = config.reports_dir / "rouge_metrics.json"
    qualitative_path = config.reports_dir / "qualitative_review.csv"
    vis_outputs = generate_visualizations(
        config, eval_result.per_method, qualitative_path
    )

    proposal_path = write_proposal(config)
    pdf_path = render_proposal_pdf(proposal_path, config.repo_root / "proposal.pdf")

    print(f"Validated rows: {summary.rows}")
    print(f"Train / test sizes: {len(splits.train)} / {len(splits.test)}")
    print(f"Methods evaluated: {[entry.method for entry in eval_result.per_method]}")
    print("ROUGE comparison:")
    print(summarize_evaluation(eval_result.per_method))
    print(f"Metrics JSON:        {metrics_path}")
    print(f"Qualitative CSV:     {qualitative_path}")
    print(f"Method summary CSV:  {eval_result.method_summary_path}")
    for name, path in vis_outputs.__dict__.items():
        if path is not None:
            print(f"Figure {name}: {path}")
    print(f"Proposal markdown:   {proposal_path}")
    if pdf_path is not None:
        print(f"Proposal PDF:        {pdf_path}")


_COMMANDS = {
    "validate-data": _validate_data,
    "evaluate": _evaluate,
    "visualize": _visualize,
    "generate-proposal": _generate_proposal,
    "run-all": _run_all,
}


if __name__ == "__main__":
    raise SystemExit(main())
