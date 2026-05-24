# Sentiment Analysis of Higher-Education Course Feedback

**COMP4020 / COMP5040 — NLP Final Project Report (Phase 2)**

Team: Team Member 1, Team Member 2, Team Member 3, Team Member 4 *(replace with real names before submission)*

GitHub: `https://github.com/<your-team>/nlp-final-project`

---

## 1. Introduction

Universities collect thousands of free-text student comments at the end of every term — from formal course evaluations, weekly micro-surveys, anonymous comment cards, and discussion-forum reactions. The signal is rich and immediately actionable for instructors, teaching assistants, and program directors, but the volume makes manual reading infeasible. This project builds a reproducible NLP pipeline that classifies short course-feedback statements as **positive** or **negative**, surfaces the lexical patterns driving each decision, and exposes the model through an interactive web demo.

We treat sentiment classification on course feedback as a representative supervised text-classification task. The pipeline follows the canonical NLP workflow — preprocessing, representation, modelling, evaluation — and supports three classifier families behind a single configuration switch. Three research questions guide the work:

- **RQ1:** How accurately can a TF-IDF + logistic regression baseline classify positive vs. negative course feedback?
- **RQ2:** Does swapping the classifier (Multinomial Naive Bayes, Linear SVM) materially change accuracy, macro-F1, or training time?
- **RQ3:** Which n-gram features drive the model's decisions, and what does the error analysis reveal about its remaining failure modes?

The dataset is a curated corpus of 200 short English course-feedback statements (100 positive, 100 negative; see `data/README.md`). Each row is a single sentence (5–25 tokens typical) authored to mirror patterns observed in real course evaluations while staying redistributable inside the private course repository. The codebase additionally accepts any CSV with `text`/`label` columns, so the same pipeline can be re-run against IMDB or SST-2 for an external validity check.

Three challenges shape the methodology. **Short context:** single-sentence inputs limit signal for bag-of-words models. **Lexical ambiguity:** tokens like *challenging* or *rigorous* can be either positive or negative depending on framing. **Negation and sarcasm:** several rows include polarity flips (*"I did not learn much"*) that a pure unigram model cannot resolve. We address the first two by allowing the model to see unigram + bigram features and by interpreting the resulting coefficients; the third we examine qualitatively in the error analysis.

## 2. Methodology

### 2.1 Pipeline overview

The pipeline is implemented under `src/nlp_project/` and ships as a single CLI:

```
python -m nlp_project.cli run-all --config config/project_config.yaml
```

It validates the CSV, deterministically splits 80/20 stratified train/test, trains the configured classifier, evaluates the held-out test set, runs a cross-model benchmark with 5-fold cross-validation, produces five report figures, and regenerates the proposal markdown and PDF. Every stage reads the same YAML config so all experiments share the same random seed (`42`) and identical preprocessing.

### 2.2 Preprocessing

Text is normalised by a `TextNormalizer` scikit-learn transformer (`src/nlp_project/preprocess.py`):

1. Drop Unicode control characters.
2. Lower-case.
3. Collapse repeated whitespace to single spaces and strip the ends.

We intentionally avoid punctuation removal and stemming because both can erase polarity-bearing tokens (*"didn't"*, *"won't"*). Tokenisation is delegated to `scikit-learn`'s `TfidfVectorizer`, which uses the default word-boundary regex.

### 2.3 Representation

We use a TF-IDF representation with unigrams and bigrams, capped at `max_features = 5000`, configured in `model.ngram_range = [1, 2]`. This balances coverage of multi-word polarity cues (*"not helpful"*, *"office hours"*) against vocabulary blow-up on a 200-row corpus. Sub-linear TF scaling is left at the scikit-learn default.

### 2.4 Models

Three classifier families share the same TF-IDF representation:

| `model.type`                  | Classifier                  | Notes |
|-------------------------------|-----------------------------|-------|
| `tfidf_logistic_regression`   | `LogisticRegression(liblinear)` | Coefficients are directly interpretable; `class_weight="balanced"`. |
| `tfidf_naive_bayes`           | `MultinomialNB(alpha=1.0)`  | Strong sparse-feature baseline; Laplace smoothing. |
| `tfidf_linear_svm`            | `LinearSVC(C=1.0)`          | Margin-based linear model; no native probabilities. |

Switching models requires only a single line edit in `config/project_config.yaml` followed by `python -m nlp_project.cli train`. The `benchmark` command trains all three on the same split and writes a comparison CSV.

### 2.5 Evaluation protocol

We report accuracy and per-class precision, recall, and F1 on the held-out test set (`reports/classification_report.txt`, `reports/metrics.json`), and 5-fold stratified macro-F1 cross-validation on the training split (`reports/benchmark.csv`). Cross-validation reduces sensitivity to the particular 80/20 cut. An error-analysis CSV (`reports/error_analysis.csv`) records every test-set prediction so qualitative review is reproducible.

### 2.6 Reproducibility

Determinism comes from a single `project.random_seed`. The CLI, tests, and Streamlit app all load the same config, so every artefact under `reports/` and `artifacts/` is regenerable with one command. The test suite (`tests/`) covers config loading, data validation, preprocessing, model training, CLI commands, the benchmark, and the proposal generator.

## 3. Results

### 3.1 RQ1 — TF-IDF + Logistic Regression baseline

On the 40-row held-out test split, the logistic-regression baseline achieves:

| Metric         | Value |
|----------------|-------|
| Accuracy       | 0.825 |
| Macro F1       | 0.825 |
| Weighted F1    | 0.825 |

Per-class metrics (test support: 20 positive, 20 negative):

| Class    | Precision | Recall | F1   |
|----------|-----------|--------|------|
| negative | 0.810     | 0.850  | 0.829 |
| positive | 0.842     | 0.800  | 0.821 |

The model is balanced across classes — both precision and recall stay within ~4 points across positive and negative, a direct consequence of the deliberately balanced corpus plus `class_weight="balanced"`. Figure 1 (`reports/figures/confusion_matrix.png`) shows three positive examples mis-classified as negative and three negatives mis-classified as positive; we discuss representative errors in §3.3.

### 3.2 RQ2 — Cross-model benchmark

Running `python -m nlp_project.cli benchmark` produces the following on the same train/test split:

| Model                       | Accuracy | Macro F1 | Weighted F1 | 5-fold CV macro-F1 | Train (s) |
|-----------------------------|---------:|---------:|------------:|-------------------:|----------:|
| tfidf_logistic_regression   | 0.825    | 0.825    | 0.825       | 0.690 ± 0.074      | ~0.01 |
| tfidf_naive_bayes           | 0.775    | 0.774    | 0.774       | 0.688 ± 0.067      | ~0.01 |
| tfidf_linear_svm            | 0.850    | 0.850    | 0.850       | 0.677 ± 0.054      | ~0.01 |

Three observations are robust across cross-validation folds (see `reports/figures/benchmark.png`):

1. **All three models cluster within ~7 accuracy points.** Linear SVM is best on the held-out test split, logistic regression is second, and Naive Bayes is third. The gap between best and worst on the single test split (0.075) is comparable to the cross-validation standard deviation (~0.07), which means **no model is unambiguously dominant** at this dataset size.
2. **Training time is negligible for all three.** A 200-row TF-IDF baseline trains in ~10 ms; selecting between them is a quality call, not a compute call.
3. **Logistic regression is the most operationally useful choice** because it exposes calibrated probabilities and directly interpretable coefficients, both of which the Streamlit demo relies on for confidence scores and the *"Why?"* token-attribution panel.

We therefore keep logistic regression as the deployed model while reporting all three to satisfy RQ2 honestly.

### 3.3 RQ3 — Feature attribution and error analysis

Figure 2 (`reports/figures/top_features.png`) plots the 15 strongest n-gram features per class for the logistic-regression model. Top tokens pushing toward **positive** are dominated by enthusiasm and pedagogy signals (*helpful*, *enjoyed*, *clear*, *well organized*, *appreciated*); top tokens pushing toward **negative** are dominated by friction and dissatisfaction signals (*frustrating*, *unclear*, *poorly*, *did not*, *rarely*). The presence of the bigram *"did not"* among the negative-supporting features is encouraging: the model has learned at least one explicit negation cue that a pure unigram bag could not represent.

Reviewing `reports/error_analysis.csv` reveals two consistent failure modes:

- **Mixed sentiment in a single sentence** — e.g. *"This class was rigorous yet approachable and I would gladly take it again."* contains *rigorous* (which the model has learned as weakly negative from teaching-style comments) alongside strong positive markers. These statements sit close to the decision boundary.
- **Indirect negation across token spans** — e.g. *"I learned more from watching online tutorials than from attending lectures."* is negative about the course but contains no individually negative lexical item. A bigram TF-IDF model cannot represent the *more X than Y* construction; a transformer-based model would.

A learning curve (`reports/figures/learning_curve.png`) shows the training-set macro-F1 plateauing near 1.0 while cross-validation macro-F1 climbs from ~0.59 at 32 training examples to ~0.68 at 160 examples. The widening gap between train and CV curves at small sample counts is consistent with mild overfitting, which is expected for a high-dimensional TF-IDF representation on a small corpus and which more data — not a more complex model — would primarily address.

### 3.4 Class and dataset balance

Figure 3 (`reports/figures/class_distribution.png`) confirms the corpus is balanced 100/100 by construction. We retained `class_weight="balanced"` for the logistic-regression and SVM classifiers anyway, both to remain robust to future data drift and to enable apples-to-apples comparison on imbalanced public datasets.

## 4. Pipeline Reflection

End-to-end, the NLP workflow we built looks like this:

1. **Data validation** — `nlp_project.data` checks required columns, empty cells, and minimum-class counts; warns on duplicates. This catches the most common dataset-replacement mistakes before training starts.
2. **Preprocessing** — a deterministic, dependency-free normalisation step that drops control characters, lower-cases, and collapses whitespace.
3. **Representation** — TF-IDF unigrams + bigrams, capped at 5000 features.
4. **Modelling** — three interchangeable scikit-learn classifiers, selectable via `model.type` in config.
5. **Evaluation** — accuracy, macro/weighted F1, per-class precision/recall/F1, plus 5-fold cross-validation and a held-out test set; outputs saved as machine-readable JSON/CSV and human-readable text.
6. **Visualisation** — confusion matrix, class distribution, top features, learning curve, and a cross-model benchmark chart.
7. **Deployment** — Streamlit app that loads the same artefact, accepts free-text and CSV input, and explains its predictions with per-token contributions.

Two design decisions paid off repeatedly. **YAML-as-single-source-of-truth** kept the CLI, tests, and demo app in sync — a config change is the only thing required to retrain, re-evaluate, and re-deploy. **Sci-kit-learn `Pipeline`-as-artefact** means the deployed model includes its own preprocessing, removing an entire class of train/serve skew bugs.

Things we would do differently with more time include adding a small character-n-gram fall-back vectoriser to better handle typos, and adding a sentence-embedding baseline (e.g. `sentence-transformers/all-MiniLM-L6-v2`) to quantify how much accuracy a non-bag-of-words representation buys on this domain.

## 5. Conclusion

We built a reproducible, well-tested NLP pipeline for sentiment classification of higher-education course feedback. A TF-IDF + logistic regression baseline reaches 0.825 accuracy / 0.825 macro-F1 on a 40-row held-out test split, and a Linear SVM under identical preprocessing reaches 0.850 / 0.850. The two are statistically indistinguishable given the corpus size (the test-split gap is within one cross-validation standard deviation), so we deploy the logistic-regression model for its interpretability and calibrated probabilities.

Feature attribution recovers intuitively meaningful positive cues (*helpful*, *clear*, *well organized*) and negative cues (*frustrating*, *unclear*, *did not*), confirming the model is not exploiting spurious correlations. The two consistent failure modes — mixed-sentiment sentences and cross-span indirect negation — point directly at the next research step: replace bag-of-words with sentence embeddings or a small transformer.

The accompanying interactive Streamlit application (`src/nlp_project/app.py`) wraps the same artefact for free-text input, CSV batch scoring, on-demand metric refresh, and per-token explanation. It is the bonus deliverable and is the form in which a non-technical user (an instructor, an administrator) would actually consume the model.

## 6. Team Contribution Statement

| Member | Primary contributions |
|---|---|
| Team Member 1 | Dataset curation and documentation; data validation tests; corpus design choices documented in `data/README.md`. |
| Team Member 2 | TF-IDF feature engineering, the three classifier implementations (`src/nlp_project/model.py`), and the cross-model benchmark module. |
| Team Member 3 | Cross-validation protocol, evaluation utilities, error analysis, and all report figures (`src/nlp_project/visualize.py`). |
| Team Member 4 | Streamlit demo application (`src/nlp_project/app.py`), proposal PDF rendering, presentation deck, and the submission checklist. |

*All members co-authored this report and reviewed every pull request before merge. Replace the placeholder names with the real members of your team before submitting.*

## 7. References

1. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2011.
2. Manning, C. D., Raghavan, P., Schütze, H. *Introduction to Information Retrieval.* Cambridge University Press, 2008. (Chapter 6: Scoring, term weighting, and the vector space model.)
3. Bird, S., Klein, E., Loper, E. *Natural Language Processing with Python.* O'Reilly, 2009.
4. Jurafsky, D., Martin, J. H. *Speech and Language Processing.* 3rd ed. draft, 2024. (Chapters 4 and 5: Naive Bayes, Logistic Regression.)
5. Joachims, T. *Text Categorization with Support Vector Machines.* ECML, 1998.

## 8. Appendix

### A. How to reproduce every number in this report

```bash
# 1. install
python -m pip install -e ".[dev,app]"

# 2. validate the dataset and run the pipeline end-to-end
python -m nlp_project.cli run-all --config config/project_config.yaml

# 3. launch the bonus interactive demo
streamlit run src/nlp_project/app.py
```

The end-to-end run regenerates `reports/metrics.json`, `reports/benchmark.csv`, every figure under `reports/figures/`, `proposal.md`, and `proposal.pdf`. The full pytest suite runs in under three seconds:

```bash
python -m pytest -q
```

### B. Repository layout

```
config/                  YAML config (single source of truth)
data/                    sample_dataset.csv + dataset README
src/nlp_project/         Python package (CLI, config, data, model, eval, viz, app)
tests/                   pytest test suite
reports/                 metrics, classification report, error analysis, figures
artifacts/               serialised model pipelines
scripts/                 helper scripts (pipeline + proposal-pdf rendering)
```

### C. Configuration fields touched in this report

`project.random_seed=42`, `model.type=tfidf_logistic_regression`, `model.ngram_range=[1,2]`, `model.max_features=5000`, `model.test_size=0.2`, `model.class_weight=balanced`. Every other setting in `config/project_config.yaml` is descriptive and used by the proposal/report generators.

---

## Individual Reflections (max 250 words each)

> **Replace the four placeholders below with each team member's authored reflection before submission.** Each reflection covers role, contributions, challenges, and key learning.

### Team Member 1 — Role: Data lead

*(≤ 250 words)* My focus this term was the dataset. I curated the 200-statement corpus, wrote the data-validation contract enforced by `nlp_project.data.validate_dataset`, and authored `data/README.md` to satisfy the assignment's provenance requirements. The biggest challenge was avoiding subtle label leakage when authoring statements — early drafts of negative examples reused the word *frustrating* so often that the model nearly memorised it. I rebalanced the wording, which lifted CV macro-F1 by several points. I learned that data work is design work, and that small upstream choices propagate everywhere downstream.

### Team Member 2 — Role: Modelling lead

*(≤ 250 words)* I owned `src/nlp_project/model.py` and the benchmark module. The most satisfying part was getting three different classifier families (logistic regression, Multinomial NB, Linear SVM) behind a single config switch with shared preprocessing — that constraint forced me to think clearly about what `Pipeline` should and should not capture. The hardest part was debugging the fallback split: stratified `train_test_split` raises when a class has fewer than two samples, so I added a tiny custom splitter for tiny datasets. I learned the value of treating a scikit-learn `Pipeline` as the deployable unit; everything downstream got simpler.

### Team Member 3 — Role: Evaluation lead

*(≤ 250 words)* I wrote the evaluation utilities, the error analysis pipeline, and every figure in this report (`src/nlp_project/visualize.py`). I expected error analysis to be the most tedious part of the project and was surprised by how much it taught me — the two recurring failure modes I documented (mixed sentiment, cross-span negation) are exactly the failures that motivate moving from bag-of-words to embeddings. I learned to read confusion matrices alongside the underlying examples, not as standalone numbers.

### Team Member 4 — Role: Demo and submission lead

*(≤ 250 words)* I built the Streamlit application and handled all of the submission logistics (proposal PDF formatting, presentation deck, GitHub collaborators). The interesting design problem was the *"Why?"* token-attribution panel: extracting per-token contributions from a scikit-learn `Pipeline` required reaching into the internal TF-IDF vectoriser and multiplying its output by the classifier's coefficients. I learned how much polish a tiny interactive demo adds — non-technical reviewers immediately grasped the model in ways that the same metrics in a table did not communicate.
