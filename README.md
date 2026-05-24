# NewsDigest — News Collection and Automatic Summarisation

COMP4020 final project. A reproducible NLP pipeline plus an interactive
Streamlit demo that takes an English news article (pasted text or a public URL)
and produces a concise summary using Lead-3, TextRank, or BART.

**Team:** Nguyen Hoang Hieu (Ethan) · Thai Ba Hung · Nguyen Quoc Dang · Le Nguyen Gia Binh
**GitHub:** https://github.com/thaibahung/NLP-Project

## What's inside

- **CLI pipeline** — `validate-data`, `summarize`, `scrape`, `evaluate`, `visualize`, `generate-proposal`, `run-all`.
- **Three summarisation methods behind one config switch** — Lead-3 (baseline), TextRank (extractive), BART (abstractive).
- **ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum** evaluation with per-row qualitative CSV.
- **Five auto-generated figures** — ROUGE comparison, article-length distribution, summary-length distribution, compression ratios, per-example ROUGE distribution.
- **Web scraping** — fetch and clean a public news URL with BeautifulSoup heuristics.
- **Streamlit demo (bonus deliverable)** — three tabs: paste article, fetch URL, live ROUGE recomputation.
- **Final report** at [`reports/final_report.md`](reports/final_report.md) and **presentation deck** at [`reports/presentation.md`](reports/presentation.md).
- **26 automated tests** (config, data, preprocessing, summarisation, evaluation, visualisation, proposal, CLI). All pass offline in ~4 seconds.

## Setup

```bash
python -m pip install -e ".[dev,app]"
```

Extras:

- `dev` — pytest + ruff
- `app` — Streamlit (for the bonus web demo)
- `data` — Hugging Face Datasets (only needed to fetch CNN/DailyMail or XSum)
- `transformer` — transformers + torch + sentencepiece (only needed for BART)

Python 3.11+ required.

## Run the full pipeline

```bash
python -m nlp_project.cli run-all --config config/project_config.yaml
```

Outputs (under [`reports/`](reports/)):

- `rouge_metrics.json` — per-method mean ROUGE scores.
- `method_summary.csv` — one row per method with ROUGE, length, compression, latency.
- `qualitative_review.csv` — every test-row × method with ROUGE breakdown.
- `figures/rouge_comparison.png`, `article_length.png`, `summary_length.png`, `compression_ratio.png`, `per_example_rouge.png`.
- regenerated `proposal.md` and `proposal.pdf`.

## CLI cheatsheet

| Command | Purpose |
|---|---|
| `validate-data` | Schema and length checks on the configured CSV. |
| `summarize --text "..." --method lead_3` | Summarise a single string. |
| `summarize --file path/to/article.txt --method textrank` | Summarise a file. |
| `scrape --url <URL> --method bart --show-article` | Fetch a public news URL and summarise it. |
| `evaluate` | Run every available method on the test split; write ROUGE reports. |
| `visualize` | Regenerate the five report figures from the saved metrics. |
| `generate-proposal` | Regenerate `proposal.md` from the YAML config. |
| `run-all` | Everything above, in one command. |

## Switching summarisation method

Edit `config/project_config.yaml`:

```yaml
model:
  baseline: lead_3
  extractive: textrank
  abstractive: facebook/bart-large-cnn
  num_sentences_extractive: 3
  use_abstractive: false   # set to true after `pip install -e ".[transformer]"`
```

The pipeline will run any method that's available. BART is lazy-loaded the
first time it's called.

## Bonus deliverable — Streamlit demo

```bash
python -m nlp_project.cli evaluate    # populates method_summary.csv first
streamlit run src/nlp_project/app.py
```

Three tabs:
1. **Paste article** — type or paste a body and get summary + latency + compression.
2. **Fetch URL** — give a public news URL; the app scrapes, cleans, and summarises it.
3. **Evaluation** — view the saved method summary table and recompute ROUGE on demand.

## Fetching the real benchmarks (CNN/DailyMail and XSum)

```bash
pip install -e ".[data]"
python scripts/fetch_datasets.py --dataset cnn_dailymail --split test --max-rows 500
python scripts/fetch_datasets.py --dataset xsum         --split test --max-rows 500
```

The script writes `data/cnn_dailymail.csv` and `data/xsum.csv` with the
canonical `id,article,highlights,source,split` schema. Then update
`config/project_config.yaml` → `data.path` and re-run `run-all`.

## Enabling BART

```bash
pip install -e ".[transformer]"
# config/project_config.yaml
# model.use_abstractive: true
python -m nlp_project.cli run-all
```

The first call downloads ~1.6 GB of model weights. Subsequent calls are cached.

## Tests

```bash
python -m pytest -q
```

26 tests, ~4 seconds, fully offline. Covers config, data validation, preprocessing,
summarisation, ROUGE evaluation, visualisations, the proposal generator, and CLI.

## Repository layout

```
config/                  YAML config (single source of truth)
data/                    bundled news_sample.csv + dataset README
src/nlp_project/
  app.py                   Streamlit demo (bonus deliverable)
  cli.py                   command-line entry point
  config.py                typed YAML loader
  data.py                  loading, validation, splits
  evaluate.py              ROUGE evaluation
  preprocess.py            HTML cleaning + sentence segmentation
  proposal.py              proposal markdown + 1-page PDF
  scraper.py               news URL fetcher / parser
  summarize.py             Lead-3, TextRank, BART
  visualize.py             every report figure
tests/                   pytest suite (26 tests)
scripts/                 fetch_datasets.py, run_pipeline.py, render_proposal_pdf.py
reports/                 metrics, figures, final_report.md, presentation.md
```

## GitHub submission checklist

- Keep the GitHub repository private.
- Add `drelhaj` and `whistle-hikhi` as collaborators.
- Include the final GitHub link in `proposal.md`, `proposal.pdf`, and the report.
- Submit Phase-1 PDF, Phase-2 final report (export `reports/final_report.md` to PDF), and Phase-3 slides to Canvas by their respective deadlines.
