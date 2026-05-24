"""NewsDigest Streamlit demo (bonus deliverable).

Run with:

    streamlit run src/nlp_project/app.py

Paste an article body or a public news URL and receive a Lead-3 / TextRank /
BART summary. BART is loaded lazily; if `transformers` is not installed the UI
hides the option."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nlp_project.config import AppConfig, find_repo_root, load_config
from nlp_project.evaluate import evaluate_summarizers, summarize_evaluation
from nlp_project.scraper import ScrapeError, fetch_article
from nlp_project.summarize import (
    AbstractiveUnavailableError,
    SUPPORTED_METHODS,
    summarize,
)


PAGE_TITLE = "NewsDigest — Automatic News Summarisation"
PAGE_ICON = "📰"


def _load_default_config() -> AppConfig:
    repo_root = find_repo_root()
    return load_config(repo_root / "config" / "project_config.yaml", warn_placeholders=False)


def _safe_method_list(config: AppConfig) -> list[str]:
    methods = ["lead_3", "textrank"]
    if config.model.use_abstractive:
        try:  # pragma: no cover
            import transformers  # noqa: F401
            methods.append("bart")
        except ImportError:
            pass
    return methods


def main() -> None:  # pragma: no cover - UI entrypoint
    import streamlit as st

    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.title(f"{PAGE_ICON}  {PAGE_TITLE}")
    st.caption(
        "Bonus deliverable for the COMP4020 NewsDigest project. "
        "Paste an article or a public news URL and receive a concise summary."
    )

    try:
        config = _load_default_config()
    except Exception as exc:
        st.error(f"Could not load project configuration: {exc}")
        st.stop()
        return

    with st.sidebar:
        st.subheader("Project")
        st.write(f"**Title:** {config.project.title}")
        st.write(f"**Task:** {config.project.task}")
        st.write(f"**Domain:** {config.project.domain}")
        st.write(f"**Random seed:** {config.project.random_seed}")
        st.divider()
        st.subheader("Settings")
        method = st.radio(
            "Summarisation method",
            options=_safe_method_list(config),
            help="Lead-3 returns the first three sentences. TextRank ranks "
                 "sentences by TF-IDF similarity. BART is abstractive (Hugging Face).",
        )
        num_sentences = st.slider(
            "Extractive output sentences",
            min_value=1,
            max_value=8,
            value=config.model.num_sentences_extractive,
        )
        st.divider()
        st.write("**Team**")
        for member in config.team.members:
            st.write(f"• {member.name}")

    tab_paste, tab_url, tab_eval = st.tabs(
        ["Paste article", "Fetch URL", "Evaluation"]
    )

    config_with_n = config
    if num_sentences != config.model.num_sentences_extractive:
        from dataclasses import replace
        new_model = replace(config.model, num_sentences_extractive=num_sentences)
        config_with_n = replace(config, model=new_model)

    with tab_paste:
        st.subheader("Paste an article body")
        article = st.text_area(
            "Article text",
            height=260,
            placeholder="Paste the body of a news article here…",
        )
        if st.button("Summarise", type="primary", key="summarise_paste"):
            if not (article or "").strip():
                st.warning("Enter article text first.")
            else:
                try:
                    result = summarize(method, article, config_with_n)
                except AbstractiveUnavailableError as exc:
                    st.error(str(exc))
                else:
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.subheader("Summary")
                        st.write(result.summary)
                    with col_b:
                        st.metric("Method", result.method)
                        st.metric("Latency", f"{result.elapsed_seconds:.3f} s")
                        compression = len(result.summary.split()) / max(1, len(article.split()))
                        st.metric("Compression", f"{compression:.1%}")

    with tab_url:
        st.subheader("Fetch and summarise a URL")
        url = st.text_input("Public news article URL")
        if st.button("Fetch + summarise", type="primary", key="summarise_url"):
            if not url.strip():
                st.warning("Paste a URL first.")
            else:
                try:
                    scraped = fetch_article(url, config.scrape)
                except ScrapeError as exc:
                    st.error(str(exc))
                else:
                    with st.expander("Show cleaned article", expanded=False):
                        st.write(f"**Title:** {scraped.title}")
                        st.write(f"**Domain:** {scraped.domain}")
                        st.write(scraped.article)
                    try:
                        result = summarize(method, scraped.article, config_with_n)
                    except AbstractiveUnavailableError as exc:
                        st.error(str(exc))
                    else:
                        st.subheader("Summary")
                        st.write(result.summary)
                        st.caption(
                            f"{result.method} · {result.elapsed_seconds:.3f}s · "
                            f"{len(result.summary.split())} tokens"
                        )

    with tab_eval:
        st.subheader("ROUGE on the held-out test split")
        st.caption(
            "Runs all available summarisation methods on the held-out test set "
            "and reports ROUGE-1, ROUGE-2, ROUGE-L."
        )
        method_summary = config.reports_dir / "method_summary.csv"
        if method_summary.exists():
            st.dataframe(
                pd.read_csv(method_summary), hide_index=True, use_container_width=True
            )
        if st.button("Recompute ROUGE", key="recompute_rouge"):
            with st.spinner("Evaluating…"):
                result = evaluate_summarizers(config_with_n)
            st.success("Evaluation complete.")
            st.code(summarize_evaluation(result.per_method))
            st.dataframe(
                pd.read_csv(result.method_summary_path),
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":  # pragma: no cover
    main()
