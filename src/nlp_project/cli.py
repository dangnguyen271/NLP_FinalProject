from __future__ import annotations

import argparse
from pathlib import Path
import sys

from nlp_project.benchmark import run_benchmark, summarize_benchmark
from nlp_project.config import ConfigError, load_config
from nlp_project.data import DataValidationError, load_and_validate_dataset, split_dataset
from nlp_project.evaluate import evaluate_model
from nlp_project.model import predict_proba_with_config, predict_texts_with_config, train_model
from nlp_project.proposal import render_proposal_pdf, write_proposal
from nlp_project.report import summarize_metrics
from nlp_project.visualize import generate_all as generate_all_visualizations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NLP project command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in (
        "validate-data",
        "train",
        "evaluate",
        "generate-proposal",
        "benchmark",
        "visualize",
        "run-all",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="config/project_config.yaml")
        command.set_defaults(func=_COMMANDS[name])

    predict = subparsers.add_parser("predict")
    predict.add_argument("--config", default="config/project_config.yaml")
    predict.add_argument("--text", required=True)
    predict.add_argument(
        "--proba",
        action="store_true",
        help="Print per-class probabilities when the model supports them.",
    )
    predict.set_defaults(func=_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ConfigError, DataValidationError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _validate_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, summary = load_and_validate_dataset(config)
    classes = ", ".join(f"{label}={count}" for label, count in summary.class_counts.items())
    print(
        f"Validated {summary.rows} rows from {config.data.path}; "
        f"classes: {classes}; duplicates: {summary.duplicate_rows}"
    )


def _train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    df, _ = load_and_validate_dataset(config)
    splits = split_dataset(df, config)
    _, artifact_path = train_model(config, splits.train)
    print(f"Model written to {artifact_path}")


def _evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = evaluate_model(config)
    print(summarize_metrics(result.metrics))
    print(f"Metrics written to {result.metrics_path}")
    print(f"Classification report written to {result.classification_report_path}")
    print(f"Error analysis written to {result.error_analysis_path}")


def _predict(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    prediction = predict_texts_with_config(config, [args.text])[0]
    print(f"Predicted label: {prediction}")
    if getattr(args, "proba", False):
        probs = predict_proba_with_config(config, [args.text])
        if probs is None:
            print("Probabilities: unavailable for this model type.")
        else:
            scores = ", ".join(f"{label}={score:.3f}" for label, score in probs[0].items())
            print(f"Probabilities: {scores}")


def _generate_proposal(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    proposal_path = write_proposal(config)
    print(f"Proposal written to {proposal_path}")


def _benchmark(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    report = run_benchmark(config)
    print(summarize_benchmark(report.rows))
    print(f"\nBenchmark JSON: {report.json_path}")
    print(f"Benchmark CSV:  {report.csv_path}")


def _visualize(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    outputs = generate_all_visualizations(config)
    for name, path in outputs.__dict__.items():
        if path is None:
            print(f"{name}: skipped (matplotlib unavailable)")
        else:
            print(f"{name}: {path}")


def _run_all(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    df, summary = load_and_validate_dataset(config)
    splits = split_dataset(df, config)
    _, artifact_path = train_model(config, splits.train)
    result = evaluate_model(config, test_df=splits.test)
    benchmark = run_benchmark(config)
    visualizations = generate_all_visualizations(config, benchmark_rows=benchmark.rows)
    proposal_path = write_proposal(config)
    pdf_path = render_proposal_pdf(proposal_path, config.repo_root / "proposal.pdf")

    print(f"Validated dataset rows: {summary.rows}")
    print(f"Model artifact: {artifact_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Classification report: {result.classification_report_path}")
    print(f"Error analysis: {result.error_analysis_path}")
    print(f"Benchmark CSV: {benchmark.csv_path}")
    print("Benchmark summary:")
    print(summarize_benchmark(benchmark.rows))
    for name, path in visualizations.__dict__.items():
        if path is not None:
            print(f"Figure {name}: {path}")
    print(f"Proposal markdown: {proposal_path}")
    if pdf_path is not None:
        print(f"Proposal PDF: {Path(pdf_path)}")
    else:
        print("Proposal PDF: skipped")


_COMMANDS = {
    "validate-data": _validate_data,
    "train": _train,
    "evaluate": _evaluate,
    "generate-proposal": _generate_proposal,
    "benchmark": _benchmark,
    "visualize": _visualize,
    "run-all": _run_all,
}


if __name__ == "__main__":
    raise SystemExit(main())
