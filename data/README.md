# Dataset Description

NewsDigest uses three sources of news article + summary pairs:

1. **Bundled sample** — `data/news_sample.csv` (24 short article + highlight pairs, redistributable inside the private course repo). Used by the offline pipeline, smoke tests, and CI.
2. **CNN/DailyMail v3.0.0** — primary benchmark dataset (Hermann et al. 2015; See et al. 2017). Long articles paired with multi-sentence reference highlights. Fetched on demand by `scripts/fetch_datasets.py`.
3. **XSum** — secondary benchmark dataset (Narayan et al. 2018). BBC articles paired with single-sentence, highly abstractive summaries. Fetched on demand by the same script.

In addition the team scrapes a small set of recent articles (≤ 20) at presentation time for the live demo via `scripts/fetch_recent.py` (uses BeautifulSoup over publicly accessible news pages).

## Schema

Every CSV consumed by the pipeline must contain at least:

- `id` (int/str): unique row identifier.
- `article` (str): the full article body. Single document per row.
- `highlights` (str): the reference / gold-standard summary.

Optional columns:

- `source` (str): outlet identifier (CNN, BBC, custom-scrape, …) used in error analysis.
- `split` (str): when present, overrides the random train/test split. Values: `train`, `test`, `validation`.

## Size and structure of the bundled sample

| Column | Type | Notes |
|---|---|---|
| `id` | int | 1–24 |
| `source` | str | always `bundled` |
| `article` | str | ~70–110 words per row |
| `highlights` | str | one sentence, ~20–30 words |

## CNN/DailyMail

- ~287k train, ~13k validation, ~11k test article/highlight pairs.
- Articles average ~780 tokens; highlights average ~56 tokens across 3-4 sentences.
- Suitability: the canonical English summarisation benchmark; matches the project's RQ1/RQ2.
- Known challenges: very long articles exceed the 1024-token cap of `bart-large-cnn`; near-duplicate articles appear across the corpus.

## XSum

- ~204k train, ~11k validation, ~11k test article/summary pairs.
- Articles average ~430 tokens; summaries are a single sentence (~23 tokens) and highly abstractive.
- Suitability: complements CNN/DailyMail with an extreme-summarisation style.
- Known challenges: single-sentence reference makes ROUGE-2 punitive; many summaries contain content not literally present in the article.

## Suitability for the NLP task

All three corpora follow the same article+summary schema, which means the same preprocessing, summarisation, and evaluation code paths work without modification across the bundled sample, CNN/DailyMail, and XSum.

## Known challenges

- **Long-form input:** CNN/DailyMail articles regularly exceed 1024 BART tokens, so the pipeline must truncate or chunk.
- **HTML/boilerplate:** scraped articles include ads, share buttons, and "read more" links that must be cleaned before summarisation.
- **Duplication:** near-duplicate stories across news aggregators bias ROUGE evaluation if not deduplicated.
- **Style heterogeneity:** CNN, BBC, and blog posts differ in average length, vocabulary, and lead structure.
- **Reference quality:** CNN/DailyMail highlights are bullet-style and often more extractive than a natural English summary.

## Replacement instructions

1. Place a new CSV (article+highlights schema) under `data/` or document a fetch script.
2. Update `config/project_config.yaml` → `data.path`, `text_column`, `summary_column`, `id_column`, `source`, `provenance`, `domain`, `challenges`.
3. Update this README.
4. Run `python -m nlp_project.cli validate-data && python -m nlp_project.cli run-all`.

## Fetching the full benchmark datasets

```bash
pip install -e ".[data]"
python scripts/fetch_datasets.py --dataset cnn_dailymail --split test --max-rows 200
python scripts/fetch_datasets.py --dataset xsum --split test --max-rows 200
```

The script writes `data/cnn_dailymail.csv` and `data/xsum.csv` with the schema above. Large fetched files are git-ignored (see `.gitignore`).
