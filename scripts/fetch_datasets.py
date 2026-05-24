"""Fetch CNN/DailyMail or XSum from the Hugging Face Datasets hub.

Usage:
    python scripts/fetch_datasets.py --dataset cnn_dailymail --split test --max-rows 500
    python scripts/fetch_datasets.py --dataset xsum --split test --max-rows 500

Writes data/<dataset>.csv with columns: id, article, highlights, source, split.
When --max-rows is set, the script avoids materialising the full split before
writing the CSV.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ROWS_API_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    config_name: str | None
    article_key: str
    summary_key: str


DATASET_SPECS = {
    "cnn_dailymail": (
        DatasetSpec("abisee/cnn_dailymail", "3.0.0", "article", "highlights"),
    ),
    "xsum": (
        DatasetSpec("EdinburghNLP/xsum", "default", "document", "summary"),
    ),
}


def _load_hf_dataset(name: str, split: str, max_rows: int | None):
    specs = DATASET_SPECS.get(name)
    if not specs:
        raise SystemExit(f"Unknown dataset: {name}")

    errors: list[str] = []
    for spec in specs:
        try:
            rows = _fetch_rows(spec, split, max_rows)
            return rows, spec.article_key, spec.summary_key
        except Exception as exc:  # noqa: BLE001 - preserve all loader failure context.
            errors.append(f"{spec.path}: {type(exc).__name__}: {exc}")

    joined_errors = "\n  - ".join(errors)
    raise SystemExit(
        "Could not load the requested dataset from Hugging Face.\n"
        f"Tried:\n  - {joined_errors}\n"
        "If rate limited, set HF_TOKEN for a Hugging Face access token and retry."
    )


def _fetch_rows(spec: DatasetSpec, split: str, max_rows: int | None) -> list[dict]:
    headers = _auth_headers()
    rows: list[dict] = []
    target_rows = max_rows if max_rows and max_rows > 0 else None
    offset = 0

    while target_rows is None or len(rows) < target_rows:
        length = PAGE_SIZE
        if target_rows is not None:
            length = min(PAGE_SIZE, target_rows - len(rows))
        params = {
            "dataset": spec.path,
            "config": spec.config_name,
            "split": split,
            "offset": offset,
            "length": length,
        }
        response = requests.get(ROWS_API_URL, params=params, headers=headers, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"{response.status_code} from Hugging Face rows API: {response.text[:500]}"
            )

        payload = response.json()
        page_rows = [item["row"] for item in payload.get("rows", [])]
        if not page_rows:
            break

        rows.extend(page_rows)
        offset += len(page_rows)

    return rows


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


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
                "id": row.get("id", idx),
                "source": args.dataset,
                "article": row.get(article_key, ""),
                "highlights": row.get(summary_key, ""),
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
