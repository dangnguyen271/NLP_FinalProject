# NLP Final Project — Sentiment Analysis of Course Feedback

COMP4020 / COMP5040 final project. A reproducible NLP pipeline plus an
interactive Streamlit demo that classifies short higher-education course
feedback as **positive** or **negative**, with cross-model benchmarking,
visualisations, and a complete report.

## What's inside

- **CLI pipeline** — validate, train, evaluate, predict, benchmark, visualize, run-all.
- **Three classifiers behind one config switch** — logistic regression, Multinomial Naive Bayes, Linear SVM.
- **5-fold stratified cross-validation** and a held-out test split, both reported.
- **Five auto-generated figures** — confusion matrix, class distribution, top features, learning curve, cross-model benchmark.
- **Interactive Streamlit web app (bonus deliverable)** — single-text and CSV-batch inference, confidence scores, and per-token "Why?" attribution.
- **Final report** at [`reports/final_report.md`](reports/final_report.md) and **presentation deck** at [`reports/presentation.md`](reports/presentation.md).
- **25 automated tests** covering config, data, preprocess, model, CLI, benchmark, visualisations, and proposal.

## Setup

```bash
python -m pip install -e ".[dev,app]"
```

Python 3.11+ required. The `app` extra pulls in Streamlit for the bonus demo;
the rest of the pipeline works without it.

## Run the full pipeline

```bash
python -m nlp_project.cli run-all --config config/project_config.yaml
```

Produces (under [`reports/`](reports/) and [`artifacts/`](artifacts/)):

- `metrics.json`, `classification_report.txt`, `error_analysis.csv`
- `benchmark.json`, `benchmark.csv`
- `figures/confusion_matrix.png`, `class_distribution.png`, `top_features.png`, `learning_curve.png`, `benchmark.png`
- `artifacts/model.joblib` (configured model) and `model_<type>.joblib` (each benchmarked model)
- regenerated `proposal.md` and `proposal.pdf`

## Individual commands

| Command | What it does |
|---|---|
| `validate-data` | Schema and content checks on the configured CSV. |
| `train` | Fit the model selected via `model.type`. |
| `evaluate` | Hold-out evaluation; writes metrics + classification report + error analysis. |
| `predict --text "..." [--proba]` | Predict a single string; optionally show class probabilities. |
| `benchmark` | Train all three models on the same split; write a comparison CSV/JSON. |
| `visualize` | Regenerate the five report figures. |
| `generate-proposal` | Regenerate `proposal.md` from `config/project_config.yaml`. |
| `run-all` | Everything above, in one command. |

## Bonus deliverable — Streamlit demo

```bash
streamlit run src/nlp_project/app.py
```

The web app:

1. Loads the trained artefact from `artifacts/model.joblib`.
2. Accepts a single statement and returns label, confidence, and per-token attributions.
3. Accepts a CSV upload for batch scoring (downloadable predictions).
4. Recomputes held-out metrics on demand.
5. Shows the cross-model benchmark from `reports/benchmark.csv` when available.

Train the model at least once before launching the app:

```bash
python -m nlp_project.cli train --config config/project_config.yaml
```

## Switching classifiers

Edit `config/project_config.yaml`:

```yaml
model:
  type: tfidf_logistic_regression   # or tfidf_naive_bayes, tfidf_linear_svm
```

Then `python -m nlp_project.cli train` to rebuild the artefact. No code change required.

## Tests

```bash
python -m pytest -q
```

The suite runs in ~4 seconds offline and covers every public module.

## Replacing the dataset

1. Place the new CSV under `data/`.
2. Update `config/project_config.yaml` — `data.path`, `text_column`, `label_column`, `id_column`, source, provenance, domain, challenges.
3. Update `data/README.md` with the new dataset description.
4. `python -m nlp_project.cli validate-data && python -m nlp_project.cli run-all`.

## GitHub submission checklist

- Keep the GitHub repository **private**.
- Add `drelhaj` and `whistle-hikhi` as collaborators.
- Include the final GitHub repository link in `proposal.md`, `proposal.pdf`, and the report.
- Submit `proposal.pdf` (Phase 1), `reports/final_report.md` exported to PDF (Phase 2), and slides (Phase 3) to Canvas by their respective deadlines.

## Repository layout

```
config/                  YAML config (single source of truth)
data/                    sample_dataset.csv + dataset README
src/nlp_project/         Python package
  app.py                   Streamlit demo (bonus deliverable)
  benchmark.py             Cross-model benchmark + CV
  cli.py                   Command-line entry point
  config.py                Typed config loader
  data.py                  Loading, validation, stratified split
  evaluate.py              Held-out metrics + error analysis
  features.py              TF-IDF vectoriser factory
  model.py                 Three classifier families + persistence
  preprocess.py            Deterministic text normaliser
  proposal.py              Proposal markdown + PDF rendering
  report.py                Metric summariser + confusion-matrix helper
  visualize.py             All report figures
tests/                   pytest suite (25 tests)
reports/                 metrics, figures, final_report.md, presentation.md
artifacts/               serialised model pipelines
scripts/                 helper scripts (pipeline runner, proposal PDF)
```
