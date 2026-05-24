"""Fetch CNN/DailyMail or XSum from the Hugging Face Datasets hub.

Usage:
    python scripts/fetch_datasets.py --dataset cnn_dailymail --split test --max-rows 500
    python scripts/fetch_datasets.py --dataset xsum --split test --max-rows 500

Writes data/<dataset>.csv with columns: id, article, highlights, source, split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _load_hf_dataset(name: str, split: str, max_rows: int | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "huggingface datasets is required. Install with `pip install -e \".[data]\"`."
        ) from exc

    if name == "cnn_dailymail":
        ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
        article_key, summary_key = "article", "highlights"
    elif name == "xsum":
        ds = load_dataset("xsum", split=split)
        article_key, summary_key = "document", "summary"
    else:
        raise SystemExit(f"Unknown dataset: {name}")

    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    return ds, article_key, summary_key


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CNN/DailyMail or XSum")
    parser.add_argument(
        "--dataset", choices=["cnn_dailymail", "xsum"], default="cnn_dailymail"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument(
        "--out", type=Path, default=None, help="Override output CSV path."
    )
    args = parser.parse_args()

    ds, article_key, summary_key = _load_hf_dataset(
        args.dataset, args.split, args.max_rows
    )

    import pandas as pd

    rows = []
    for idx, row in enumerate(ds):
        rows.append(
            {
                "id": idx,
                "source": args.dataset,
                "article": row[article_key],
                "highlights": row[summary_key],
                "split": args.split,
            }
        )
    df = pd.DataFrame(rows)

    out_path = args.out or (ROOT / "data" / f"{args.dataset}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
