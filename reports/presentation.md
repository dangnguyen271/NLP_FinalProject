# Presentation Deck — Phase 3

**Format:** ~12 content slides + title + Q&A = ~20 min talk + 5–7 min Q&A.
**Allocations below assume four presenters; every member presents.**

---

## Slide 1 — Title
**Sentiment Analysis of Higher-Education Course Feedback**
COMP4020 / COMP5040 — NLP Final Project
Team Members 1–4 *(replace with real names)* · *Today's date*

> Speaker: Member 4. 30 sec. Open with one sentence the audience will remember: *"We built a model that reads student feedback and tells instructors what their class actually thinks — in under 100 ms."*

---

## Slide 2 — Motivation & Problem Definition
- Universities collect thousands of free-text student comments per term.
- Reading them all is infeasible; ignoring them wastes signal.
- **Task:** binary sentiment classification (positive vs. negative) of short course-feedback statements.
- **Stakeholder:** instructors and program directors who need actionable summaries.

> Speaker: Member 1. ~90 sec. Anchor the talk in the real human problem before any modelling appears.

---

## Slide 3 — Dataset
- 200 short English course-feedback statements, balanced 100 positive / 100 negative.
- Columns: `id`, `text`, `label`.
- Authored to mirror real evaluations; redistributable inside the private repo.
- Challenges: lexical ambiguity, negation, short single-sentence inputs, single-author voice.

> Speaker: Member 1. ~90 sec. Show `data/sample_dataset.csv` head + the class-distribution figure.

---

## Slide 4 — NLP Pipeline
1. Validate & split (80/20 stratified)
2. Normalise (Unicode-clean, lowercase, whitespace)
3. TF-IDF, unigrams + bigrams, max 5000 features
4. Classifier (3 interchangeable options)
5. Evaluate (accuracy, F1, 5-fold CV, error analysis)
6. Visualise + deploy

> Speaker: Member 2. ~90 sec. Use the pipeline diagram from `reports/figures/`.

---

## Slide 5 — Research Questions
- **RQ1** — How well does a TF-IDF + logistic regression baseline classify?
- **RQ2** — Do Naive Bayes or Linear SVM materially change accuracy / F1 / training time?
- **RQ3** — Which n-gram features drive decisions, and what do the errors reveal?

> Speaker: Member 2. ~45 sec. Each RQ corresponds to one results slide.

---

## Slide 6 — Methods
- Preprocessing: `TextNormalizer` scikit-learn transformer.
- Representation: `TfidfVectorizer(ngram_range=(1,2), max_features=5000)`.
- Models: `LogisticRegression(liblinear)`, `MultinomialNB(α=1)`, `LinearSVC(C=1)`.
- Single YAML config (`config/project_config.yaml`) drives every experiment.
- Cross-validation: 5-fold stratified, scored on macro-F1.

> Speaker: Member 2. ~90 sec.

---

## Slide 7 — RQ1 Results (LR baseline)
- Accuracy: **0.825** · Macro F1: **0.825**
- Per-class F1: negative **0.829**, positive **0.821**
- Show confusion matrix figure.

> Speaker: Member 3. ~75 sec.

---

## Slide 8 — RQ2 Results (Model comparison)

| Model | Acc | Macro-F1 | CV Macro-F1 (5-fold) |
|---|---:|---:|---:|
| Logistic Regression | 0.825 | 0.825 | 0.690 ± 0.074 |
| Naive Bayes | 0.775 | 0.774 | 0.688 ± 0.067 |
| Linear SVM | 0.850 | 0.850 | 0.677 ± 0.054 |

- All three within ~1 CV standard deviation of each other.
- We deploy LR for **interpretability + calibrated probabilities**.

> Speaker: Member 3. ~90 sec. Show `reports/figures/benchmark.png`.

---

## Slide 9 — RQ3 Feature attribution & errors
- Top positive features: *helpful, enjoyed, clear, well organized, appreciated*.
- Top negative features: *frustrating, unclear, poorly, did not, rarely*.
- Two recurring failure modes:
  - Mixed sentiment in one sentence ("rigorous yet approachable").
  - Cross-span indirect negation ("learned more from tutorials than from lectures").

> Speaker: Member 3. ~90 sec. Show `reports/figures/top_features.png` and one error-analysis row.

---

## Slide 10 — Demo (mandatory live demo)
- `streamlit run src/nlp_project/app.py`
- Type free-text → label + confidence + per-token "Why?".
- Upload CSV → batch predictions with downloadable CSV.
- Refresh held-out metrics live.

> Speaker: Member 4. ~3 min. Have a fallback screen-recording in case Wi-Fi fails.

---

## Slide 11 — Reproducibility
- One command: `python -m nlp_project.cli run-all`.
- Single YAML config drives CLI, tests, and demo.
- `pytest -q` — full suite, <3 sec, no internet required.
- All artefacts (metrics, figures, model) regenerable from `git clone`.

> Speaker: Member 4. ~60 sec.

---

## Slide 12 — Limitations
- Single domain (one university's course-feedback voice).
- Bag-of-words cannot represent cross-span negation.
- 200 examples — enough to compare baselines, not to deploy at scale.
- No human inter-annotator agreement (single-author labels).

> Speaker: Member 3. ~60 sec.

---

## Slide 13 — Conclusion & Future Work
- Reproducible TF-IDF + linear-classifier pipeline achieves 0.83–0.85 accuracy.
- Linear SVM marginally best; logistic regression deployed for interpretability.
- **Next:** sentence-transformer baseline; multi-domain generalisation; aspect-based extension (course, instructor, workload).

> Speaker: Member 2. ~60 sec.

---

## Slide 14 — Q&A

> All four members on stage. Pre-rehearsed quick answers for:
> - "How would you handle imbalanced data?" → `class_weight='balanced'` + macro-F1 reporting + stratified CV.
> - "Why TF-IDF over embeddings?" → reproducibility + interpretability on small data + sub-10 ms inference.
> - "How do you know the model isn't memorising?" → 5-fold CV + held-out test + learning curve in the report.

---

## Speaker notes for the live demo (Slide 10)

1. Open the Streamlit app on a known port (`streamlit run src/nlp_project/app.py --server.port 8501`).
2. Prepared example **positive**: *"I really appreciated the office hours and the clear feedback on assignments."*
3. Prepared example **negative**: *"The lectures were disorganized and the workload was crushing."*
4. Prepared **edge case** to show graceful failure: *"Challenging but rewarding"* — discuss why the model wavers.
5. Drag the small `reports/error_analysis.csv` into the batch tab to demonstrate bulk inference.
6. Hit *Refresh metrics* in the third tab to recompute accuracy live.

## Backup contingency

If the live demo fails:
- Switch to the pre-recorded 90-second screencast saved at `reports/figures/` (record this before submission).
- Fall back to `python -m nlp_project.cli predict --text "..." --proba` in a terminal.
- Show `reports/benchmark.csv` and the four PNG figures full-screen.
