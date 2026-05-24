# NewsDigest — Presentation Deck (Phase 3)

**Format:** ~12 content slides + title + Q&A = ~20 min + 5–7 min Q&A. Every team member presents.

---

## Slide 1 — Title
**NewsDigest: News Collection and Automatic Summarisation**
COMP4020 — NLP Final Project
Nguyen Hoang Hieu (Ethan) · Thai Ba Hung · Nguyen Quoc Dang · Le Nguyen Gia Binh
GitHub: github.com/thaibahung/NLP-Project

> Speaker: Binh. 30 sec. Open with the elevator pitch: *"NewsDigest takes a news article and gives you back a faithful, readable summary in under one second."*

---

## Slide 2 — Motivation & Problem Definition
- Digital news volume keeps growing; manual reading does not scale.
- Many articles repeat each other across outlets and bury the key facts.
- **Task:** single-document text summarisation of English news.
- **Stakeholders:** busy readers, journalists triaging the news cycle, accessibility tools.

> Speaker: Ethan. 90 sec.

---

## Slide 3 — Datasets
- **Bundled sample** — 24 article-highlight pairs for offline tests.
- **CNN/DailyMail v3.0.0** — primary benchmark (long articles, multi-sentence highlights).
- **XSum** — secondary benchmark (single-sentence extreme summaries).
- **Scraped articles** — small live set for the demo.

> Speaker: Ethan. 90 sec. Show `data/news_sample.csv` head + the article-length histogram.

---

## Slide 4 — NLP Pipeline
1. Validate CSV (schema, lengths, duplicates)
2. Clean HTML + Unicode + boilerplate (preserve case + punctuation)
3. Sentence-segment with abbreviation-aware regex
4. Summarise — Lead-3, TextRank, or BART (one config switch)
5. Score with ROUGE-1/2/L/Lsum + qualitative CSV
6. Visualise + deploy as Streamlit demo

> Speaker: Hung. 90 sec. Show the pipeline diagram or `reports/figures/article_length.png`.

---

## Slide 5 — Research Questions
- **RQ1** — How well does an NLP system summarise news?
- **RQ2** — Does BART beat extractive baselines on ROUGE?
- **RQ3** — Which article characteristics drive method differences?

> Speaker: Hung. 30 sec.

---

## Slide 6 — Methods
- **Lead-3** — first three sentences (canonical news baseline).
- **TextRank** — TF-IDF + cosine-similarity sentence graph + PageRank.
- **BART (`facebook/bart-large-cnn`)** — abstractive, via Hugging Face `pipeline("summarization")`.
- Identical preprocessing for every method; single config flag switches between them.

> Speaker: Dang. 90 sec. Show the three-line API: `summarize(method, article, config)`.

---

## Slide 7 — RQ1 Results (Lead-3 on bundled sample)

| Metric  | Value |
|---------|------:|
| ROUGE-1 |  0.359 |
| ROUGE-2 |  0.121 |
| ROUGE-L |  0.257 |

- Lead paragraphs are *engineered* to be summaries — Lead-3 is a tough baseline to beat.

> Speaker: Dang. 60 sec.

---

## Slide 8 — RQ2 Method comparison (bundled sample)

| Method   | ROUGE-1 | ROUGE-2 | ROUGE-L | Latency |
|----------|--------:|--------:|--------:|--------:|
| lead_3   |  0.359  |  0.121  |  0.257  |  ~0 ms |
| textrank |  0.367  |  0.128  |  0.269  |   7 ms |
| bart\*  |   ≈0.44 |   ≈0.21 |   ≈0.40 | 1–3 s  |

\* BART numbers shown are published `bart-large-cnn` scores on the full CNN/DailyMail dataset; reproduce with `scripts/fetch_datasets.py` then re-run.

> Speaker: Dang. 90 sec. Show `reports/figures/rouge_comparison.png`.

---

## Slide 9 — RQ3 Where the methods differ
- Short articles ⇒ Lead-3 and TextRank converge to the same output.
- Multi-subject articles ⇒ TextRank picks the globally most central sentence.
- One-sentence references (XSum) ⇒ extractive multi-sentence outputs lose ROUGE-2.
- Qualitative review in `reports/qualitative_review.csv`.

> Speaker: Binh. 90 sec. Show one failure-mode row from the qualitative CSV.

---

## Slide 10 — Live demo (mandatory)
1. `streamlit run src/nlp_project/app.py`
2. **Paste article** → summary + latency + compression.
3. **Fetch URL** → cleaned body + summary on a public news article.
4. **Recompute ROUGE** → live evaluation tab.

> Speaker: Binh. 3 min. Backup screencast saved to `reports/figures/` (record before submission).

---

## Slide 11 — Reproducibility
- One command: `python -m nlp_project.cli run-all`.
- Single YAML config drives CLI, tests, demo.
- `pytest -q` — 26 tests, ~4 s, fully offline.
- Every artefact in `reports/` regenerable from `git clone`.

> Speaker: Hung. 60 sec.

---

## Slide 12 — Limitations
- Bundled sample is small (24 rows); real numbers come from the HF benchmarks.
- BART's 1024-token cap truncates long news articles.
- ROUGE rewards lexical overlap, not faithfulness — manual qualitative review remains essential.
- Single-language (English); domain narrow to news.

> Speaker: Hung. 60 sec.

---

## Slide 13 — Conclusion & Future Work
- Reproducible 3-method news-summarisation pipeline with a Streamlit demo.
- Lead-3 and TextRank are very close on short news; BART expected to lead on the full benchmark.
- **Next:** chunked long-article decoding; fine-tune BART on XSum; multi-document summarisation; faithfulness metrics (QAGS, BERTScore).

> Speaker: Ethan. 60 sec.

---

## Slide 14 — Q&A

> All four members on stage. Pre-rehearsed answers for:
> - *"Why TF-IDF for TextRank rather than embeddings?"* — reproducibility on small datasets, no GPU needed.
> - *"How would you scale to thousands of articles?"* — batch BART on GPU, cache embeddings, deduplicate at ingest.
> - *"How do you know the summaries are faithful?"* — ROUGE + qualitative review; future work adds QAGS-style faithfulness.

---

## Demo script (Slide 10)

1. Open the Streamlit app: `streamlit run src/nlp_project/app.py --server.port 8501`.
2. Prepared **paste-article** example: any 4–6 sentence news paragraph from CNN's homepage.
3. Prepared **URL** example: a BBC article URL chosen the morning of the talk (test it 10 minutes before stage).
4. Show the **method switch** in the sidebar — Lead-3 vs. TextRank produces visibly different output on multi-subject articles.
5. Hit **Recompute ROUGE** in the evaluation tab to demonstrate live evaluation.

## Backup contingency

- Pre-recorded 90-second screencast (save to `reports/figures/demo.mp4`).
- Terminal fallback: `python -m nlp_project.cli summarize --text "<paste>" --method textrank`.
- Static fallback: show `reports/method_summary.csv` + four PNG figures full-screen.
