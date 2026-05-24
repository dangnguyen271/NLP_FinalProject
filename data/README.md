# Dataset Description

## Source and provenance

`data/sample_dataset.csv` is a curated corpus of 200 short English course-feedback
statements written for this NLP final project. The statements were authored to mirror
patterns commonly observed in real student feedback (course evaluations, post-class
surveys, anonymous comment cards) while remaining redistributable inside a private
academic repository. The dataset can be regenerated deterministically by re-running the
scripted augmentation utilities under `scripts/`.

For comparison and external validity, the codebase is also capable of consuming
publicly available text-classification datasets that follow the same `id,text,label`
schema (e.g. IMDB sentiment, SST-2, Amazon polarity). Drop a compatible CSV into
`data/` and point `config/project_config.yaml` at it; no code changes are required.

## Size and structure

- Rows: 200 (100 positive, 100 negative)
- Columns:
  - `id` (int): unique row identifier.
  - `text` (str): a single short English statement (5–25 tokens typical).
  - `label` (str): one of `positive` or `negative`.

The dataset is balanced by construction; class imbalance can be simulated for
robustness testing by sub-sampling with the `scripts/run_pipeline.py` helper.

## Domain

The domain is **higher-education course feedback** (specifically NLP/ML courses).
This domain was chosen for three reasons:

1. It is directly relevant to the COMP4020 / COMP5040 audience, who routinely produce
   and consume such feedback.
2. The vocabulary is rich enough (pedagogy, NLP terminology, organisational language)
   to surface meaningful n-gram features for TF-IDF baselines.
3. Sentiment is reliably resolvable from short snippets, keeping annotation noise low.

## Suitability for the NLP task

The dataset supports the configured **binary text classification** task. With 200
balanced examples, stratified 80/20 splits leave 40 held-out test rows distributed
evenly across classes, which is sufficient to demonstrate the full NLP pipeline
(preprocessing → representation → modelling → evaluation) and to compare baselines.

For the final report's *generalisation* discussion we additionally recommend a sanity
check against an external public dataset (IMDB or SST-2) which the codebase supports
out-of-the-box.

## Known challenges

- **Domain shift:** the corpus is specific to course feedback; models trained on it
  may not transfer to other sentiment domains without fine-tuning.
- **Lexical ambiguity:** words like "challenging" or "rigorous" can be either
  positive or negative depending on framing; the model must learn context.
- **Short text:** each statement is a single sentence, limiting signal for
  bag-of-words representations and motivating n-gram features.
- **Author distribution:** statements share a single author voice; real-world
  feedback varies more widely in style and length.
- **Sarcasm / negation:** several rows include negation ("not unreasonable",
  "I did not learn much") that test the model's ability to handle polarity flips.

## Replacement instructions

1. Place the new CSV under `data/` (or document an external download script).
2. Update `config/project_config.yaml`:
   - `data.path`, `data.text_column`, `data.label_column`, optional `data.id_column`
   - source, provenance, domain, and known challenges
3. Update this README with the final dataset description.
4. Run `python -m nlp_project.cli validate-data --config config/project_config.yaml`.
5. Run the full pipeline with `python -m nlp_project.cli run-all`.
