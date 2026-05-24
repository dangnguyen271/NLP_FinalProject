# NLP Project Scaffold

This repository is a reproducible COMP4020 / COMP5040 NLP project scaffold. It provides
a configurable text classification baseline with data validation, preprocessing,
training, evaluation, proposal generation, and a small prediction interface.

The default configuration uses `data/sample_dataset.csv`, a tiny demo dataset for tests
and smoke runs only. Replace it with the team's selected dataset before making project
claims.

## Setup

```bash
python -m pip install -e ".[dev]"
```

Python 3.11 or newer is required.

## Run tests

```bash
python -m pytest -q
```

Optional coverage:

```bash
python -m pytest --cov=src/nlp_project --cov-report=term-missing
```

## Run the pipeline

```bash
python -m nlp_project.cli run-all --config config/project_config.yaml
```

The pipeline validates the CSV, trains a TF-IDF logistic regression baseline, writes
evaluation reports under `reports/`, saves the model under `artifacts/`, regenerates
`proposal.md`, and attempts to render `proposal.pdf`.

Prediction example:

```bash
python -m nlp_project.cli predict \
  --config config/project_config.yaml \
  --text "This example text is useful and clear."
```

## Replace the sample dataset

1. Put the final CSV dataset under `data/`.
2. Update `config/project_config.yaml`:
   - `data.path`
   - `data.text_column`
   - `data.label_column`
   - `data.id_column`, if available
   - source, provenance, domain, and known challenges
3. Update `data/README.md` with the final dataset description.
4. Run `python -m nlp_project.cli validate-data --config config/project_config.yaml`.
5. Run the full pipeline again.

## Regenerate proposal outputs

```bash
python -m nlp_project.cli generate-proposal --config config/project_config.yaml
python scripts/render_proposal_pdf.py --config config/project_config.yaml
```

The PDF renderer uses 12 pt Times-compatible text through ReportLab when available. If
the proposal becomes too long, shorten the configuration text and regenerate.

## GitHub submission checklist

- Keep the GitHub repository private.
- Add `drelhaj` and `whistle-hikhi` as collaborators.
- Include the final GitHub repository link in `proposal.md` and `proposal.pdf`.
- Submit the proposal PDF to Canvas by the deadline.
