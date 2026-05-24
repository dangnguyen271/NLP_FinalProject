"""Streamlit demo application — the project's bonus deliverable.

Run with:

    streamlit run src/nlp_project/app.py

The app is intentionally self-contained: it discovers the project config, loads
the trained artefact, and offers a single-text inference UI plus a batch-CSV
upload mode. Both modes degrade gracefully when the model does not expose
probabilities (e.g. Linear SVM)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from nlp_project.config import AppConfig, find_repo_root, load_config
from nlp_project.evaluate import evaluate_model
from nlp_project.model import (
    SUPPORTED_MODEL_TYPES,
    load_model,
    predict_proba_with_config,
    predict_texts_with_config,
)


PAGE_TITLE = "NLP Course-Feedback Sentiment Demo"
PAGE_ICON = "🎓"


def _load_default_config() -> AppConfig:
    repo_root = find_repo_root()
    return load_config(repo_root / "config" / "project_config.yaml", warn_placeholders=False)


def _stylise(score: float) -> str:
    if score >= 0.75:
        return "very confident"
    if score >= 0.6:
        return "confident"
    if score >= 0.5:
        return "leaning"
    return "uncertain"


def _format_proba(proba: dict[str, float]) -> pd.DataFrame:
    rows = [
        {"label": label, "probability": score}
        for label, score in sorted(proba.items(), key=lambda kv: -kv[1])
    ]
    return pd.DataFrame(rows)


def _explain_top_features(model, text: str, top_k: int = 5) -> pd.DataFrame | None:
    """Best-effort linear-model token attribution for a single input."""

    if not hasattr(model, "named_steps"):
        return None
    vectorizer = model.named_steps.get("tfidf")
    classifier = model.named_steps.get("classifier")
    if vectorizer is None or classifier is None or not hasattr(classifier, "coef_"):
        return None

    normalizer = model.named_steps.get("normalize")
    normalized = normalizer.transform([text])[0] if normalizer is not None else text
    vector = vectorizer.transform([normalized])
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_

    if coefs.shape[0] == 1:
        contributions = (vector.toarray()[0] * coefs[0])
        pos_class = str(classifier.classes_[1])
        neg_class = str(classifier.classes_[0])
        order = contributions.argsort()
        top_pos = order[-top_k:][::-1]
        top_neg = order[:top_k]
        records = []
        for idx in top_pos:
            if contributions[idx] > 0:
                records.append(
                    {"token": feature_names[idx], "weight": float(contributions[idx]), "supports": pos_class}
                )
        for idx in top_neg:
            if contributions[idx] < 0:
                records.append(
                    {"token": feature_names[idx], "weight": float(contributions[idx]), "supports": neg_class}
                )
        return pd.DataFrame(records).sort_values("weight", ascending=False)

    # Multiclass: report each class's strongest supporting token.
    records = []
    for class_idx, label in enumerate(classifier.classes_):
        contributions = vector.toarray()[0] * coefs[class_idx]
        top = contributions.argsort()[-top_k:][::-1]
        for idx in top:
            if contributions[idx] > 0:
                records.append(
                    {"token": feature_names[idx], "weight": float(contributions[idx]), "supports": str(label)}
                )
    return pd.DataFrame(records)


def _ensure_model(config: AppConfig):
    if not config.model.artifact_path.exists():
        return None
    return load_model(config.model.artifact_path)


def _read_uploaded_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


def main() -> None:  # pragma: no cover - UI entrypoint
    import streamlit as st  # local import keeps streamlit optional

    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.title(f"{PAGE_ICON}  {PAGE_TITLE}")
    st.caption(
        "Bonus deliverable for the COMP4020 / COMP5040 NLP final project. "
        "Type a sentence of course feedback and the model returns its predicted "
        "sentiment, confidence, and the tokens that drove the decision."
    )

    try:
        config = _load_default_config()
    except Exception as exc:
        st.error(f"Could not load project configuration: {exc}")
        st.stop()
        return

    model = _ensure_model(config)
    if model is None:
        st.warning(
            "No trained model artefact was found. "
            f"Run `python -m nlp_project.cli train --config {config.config_path}` "
            "in a terminal first."
        )
        st.stop()
        return

    with st.sidebar:
        st.subheader("Project")
        st.write(f"**Title:** {config.project.title}")
        st.write(f"**Domain:** {config.project.domain}")
        st.write(f"**Task:** {config.project.task}")
        st.write(f"**Active model:** `{config.model.type}`")
        st.write(f"**Random seed:** {config.project.random_seed}")
        st.divider()
        st.caption(
            "Switch between models by editing `config/project_config.yaml` "
            "and re-running `python -m nlp_project.cli train`."
        )
        st.write("**Supported model types**")
        for model_type in SUPPORTED_MODEL_TYPES:
            marker = "✅" if model_type == config.model.type else "•"
            st.write(f"{marker} {model_type}")

    single_tab, batch_tab, metrics_tab = st.tabs(
        ["Single prediction", "Batch CSV", "Held-out metrics"]
    )

    with single_tab:
        st.subheader("Predict a single statement")
        example = (
            "The lectures were extremely well organized and the slides "
            "made complex topics easy to follow."
        )
        text = st.text_area("Course-feedback statement", value=example, height=120)
        if st.button("Predict", type="primary"):
            text = (text or "").strip()
            if not text:
                st.warning("Enter at least one non-empty sentence.")
            else:
                label = predict_texts_with_config(config, [text])[0]
                probs = predict_proba_with_config(config, [text])
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.metric("Predicted label", label)
                    if probs is not None:
                        top_score = max(probs[0].values())
                        st.metric(
                            "Confidence",
                            f"{top_score:.1%}",
                            help=f"Calibration descriptor: {_stylise(top_score)}",
                        )
                with col_b:
                    if probs is not None:
                        st.bar_chart(_format_proba(probs[0]).set_index("label"))
                    else:
                        st.info(
                            "Active model does not expose calibrated probabilities. "
                            "Switch to `tfidf_logistic_regression` or `tfidf_naive_bayes` "
                            "to enable confidence scores."
                        )

                explanation = _explain_top_features(model, text)
                if explanation is not None and not explanation.empty:
                    st.subheader("Why?")
                    st.dataframe(explanation, hide_index=True, use_container_width=True)
                    st.caption(
                        "Tokens with the largest positive weight push the prediction "
                        "toward each class; negative weights push it away."
                    )

    with batch_tab:
        st.subheader("Score a CSV of statements")
        st.caption(
            "Upload a CSV with a column named `text`. The output adds a "
            "`predicted_label` column (and `confidence` when available)."
        )
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            df = _read_uploaded_csv(uploaded)
            if "text" not in df.columns:
                st.error("The uploaded CSV must contain a column named `text`.")
            else:
                texts: Iterable[str] = df["text"].astype(str).tolist()
                df = df.copy()
                df["predicted_label"] = predict_texts_with_config(config, texts)
                probs = predict_proba_with_config(config, texts)
                if probs is not None:
                    df["confidence"] = [max(row.values()) for row in probs]
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download predictions",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

    with metrics_tab:
        st.subheader("Held-out evaluation")
        st.caption(
            "Recomputes metrics on the configured test split using the artefact "
            "currently on disk."
        )
        if st.button("Refresh metrics"):
            with st.spinner("Evaluating…"):
                result = evaluate_model(config)
            metrics = result.metrics
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
            st.metric("Weighted F1", f"{metrics['weighted_f1']:.3f}")
            per_class = pd.DataFrame(metrics["per_class"]).T.reset_index().rename(
                columns={"index": "label"}
            )
            st.dataframe(per_class, hide_index=True, use_container_width=True)
            st.caption(
                f"Metrics file: `{result.metrics_path}` · "
                f"Error analysis: `{result.error_analysis_path}`"
            )

        benchmark_csv = config.reports_dir / "benchmark.csv"
        if benchmark_csv.exists():
            st.divider()
            st.subheader("Cross-model benchmark")
            st.dataframe(
                pd.read_csv(benchmark_csv), hide_index=True, use_container_width=True
            )
        else:
            st.info(
                "Run `python -m nlp_project.cli benchmark` to populate the "
                "cross-model comparison shown here."
            )


if __name__ == "__main__":  # pragma: no cover
    main()
