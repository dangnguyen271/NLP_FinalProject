# Sentiment Analysis of Higher-Education Course Feedback

## NLP task and domain/application area
Task: text_classification. Domain: higher education. Application area: automated analysis of student course feedback.

## Motivation and problem statement
Universities collect thousands of free-text student comments every term but cannot read them all manually; automatic sentiment classification surfaces actionable signal for instructors, teaching assistants, and program directors. Given a short course-feedback statement, predict whether the underlying sentiment is positive or negative, and surface the lexical patterns that drive each decision so that the system remains interpretable.

## Expected final product
A reproducible NLP pipeline plus an interactive Streamlit web demo that accepts free-text feedback and returns a label, a confidence score, and the top contributing n-grams. It improves: It replaces hours of manual triage of end-of-term surveys with a few seconds of automated, transparent inference.

## Research questions
- RQ1: How accurately can a TF-IDF + logistic regression baseline classify positive vs. negative course feedback?
- RQ2: Does swapping the classifier (Naive Bayes, Linear SVM) materially change accuracy, F1, or training time?
- RQ3: Which n-gram features drive the model's decisions, and what does the error analysis reveal about its remaining failure modes?

## Dataset
Source: Curated corpus of 200 short English course-feedback statements assembled for this final project (see data/README.md for redistribution notes). Statements authored to mirror patterns observed in real student feedback; fully redistributable inside the private course repository. Size: 200 rows, 3 columns; labels: negative: 100, positive: 100. Domain: higher-education course feedback (NLP / ML courses). Challenges: Short single-sentence inputs limit context for bag-of-words models.; Lexical ambiguity (e.g. 'challenging', 'rigorous') flips polarity depending on context.; Negation and sarcasm can invert sentiment without changing the dominant tokens.; Author-style homogeneity (one cohort) means out-of-distribution generalisation must be verified separately.

## Team responsibilities
- **Team Member 1**: Data collection, curation, and preprocessing; Dataset documentation and license review
- **Team Member 2**: Feature engineering (TF-IDF, n-grams); Model implementation and experiment tracking
- **Team Member 3**: Cross-validation, evaluation, and error analysis; Visualisations for the final report
- **Team Member 4**: Streamlit demo application (bonus deliverable); Proposal PDF, presentation slides, and submission checklist

## GitHub repository link
https://github.com/<your-team>/nlp-final-project

## Submission checklist
- [ ] Private GitHub repo with collaborators `drelhaj` and `whistle-hikhi`.
- [ ] Final GitHub link included in this proposal and the PDF.
- [ ] Proposal PDF exported in 12 pt Times-compatible font, submitted to Canvas.
