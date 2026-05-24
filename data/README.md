# Dataset Description

## Source and provenance

The current file, `data/sample_dataset.csv`, is a tiny local sample dataset created only
to make the repository, tests, and demo pipeline runnable before the team selects a final
dataset. It is not an externally sourced research dataset and must not be presented as
the final dataset for project claims.

## Size and structure

The sample CSV has 12 rows and three columns:

- `id`: row identifier.
- `text`: short English text input.
- `label`: binary demonstration label, either `positive` or `negative`.

## Domain

The sample rows use a generic project-planning/demo domain. Replace this section with
the real project domain such as healthcare, finance, education, reviews, news, social
media, or another selected application area.

## Suitability for the NLP task

The sample dataset is suitable only for smoke testing a text classification pipeline. It
is intentionally balanced and small so automated tests can run quickly without internet
access. It is not suitable for final model selection, error analysis, or performance
claims.

## Known challenges

- The sample dataset is too small for meaningful evaluation.
- The domain is generic and should be replaced.
- The final dataset may include noisy text, class imbalance, multilingual content,
  duplicated entries, missing values, or licensing restrictions that must be documented.

## Replacement instructions

1. Place the selected final dataset under `data/` or document where it can be obtained.
2. Update `config/project_config.yaml` with the dataset path and column names.
3. Replace this README content with the final source, provenance, size, structure,
   domain, suitability, and known challenges.
4. Run `python -m nlp_project.cli run-all --config config/project_config.yaml`.
