"""Page 3 - multilingual sentiment dashboard (dataset or latest bulk analysis)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from src import data_exploration as eda
from src import ui
from src import visualization as viz

ui.setup_page("Multilingual Analytics", ":material/bar_chart:")
ui.page_intro("Multilingual Sentiment Dashboard", "Explore sentiment patterns, language coverage and review characteristics across the project dataset or your latest bulk analysis.", "DATA EXPLORATION / GLOBAL COVERAGE")

st.markdown('<div class="ms-section-kicker">01 · FILTER YOUR ANALYSIS</div>', unsafe_allow_html=True)
sources = ["Project dataset (gold labels)"]
if "bulk_results" in st.session_state:
    sources.append("Latest bulk analysis (predicted labels)")
choice = st.radio("Data", sources, horizontal=True)

if choice.startswith("Project"):
    df = ui.dataset_required()
    if df is None:
        st.stop()
    splits = st.multiselect("Splits", ["train", "val", "test"], default=["train", "val", "test"])
    df = df[df["split"].isin(splits)]
    st.caption("Sentiment here is the dataset label (native label or mapped star rating), not a prediction.")
else:
    from src.bulk_analysis import prediction_columns
    res = st.session_state.bulk_results
    key = st.selectbox("Predictions from model", prediction_columns(res),
                       format_func=lambda k: settings.MODEL_DISPLAY_NAMES[k])
    df = res[res["analysis_status"] == "analysed"].copy()
    df["sentiment"] = df[f"{key}_sentiment"]
    df["length_chars"] = df["review_length_chars"]
    st.caption(f"File: {st.session_state.bulk_meta['file']}. Sentiment = model prediction.")

if df.empty:
    st.warning("No reviews to display.")
    st.stop()

langs = sorted(df["language_name"].dropna().unique())
picked = st.multiselect("Languages", langs, default=langs)
df = df[df["language_name"].isin(picked)]
if df.empty:
    st.warning("Select at least one language.")
    st.stop()

st.markdown('<div class="ms-section-kicker">02 · DATASET SNAPSHOT</div>', unsafe_allow_html=True)
counts = df["sentiment"].value_counts()
k = st.columns(6)
k[0].metric("Total reviews", f"{len(df):,}")
k[1].metric("Positive", f"{counts.get('positive', 0):,}")
k[2].metric("Negative", f"{counts.get('negative', 0):,}")
k[3].metric("Neutral", f"{counts.get('neutral', 0):,}")
k[4].metric("Languages", df["language_name"].nunique())
k[5].metric("Avg. length", f"{df['length_chars'].mean():.0f} chars")

st.markdown('<div class="ms-section-kicker">03 · SENTIMENT & LANGUAGE PATTERNS</div>', unsafe_allow_html=True)
c1, c2 = st.columns([1, 1.15], gap="large")
with c1:
    st.plotly_chart(viz.sentiment_distribution(df, title="Overall sentiment mix"), width="stretch")
with c2:
    st.plotly_chart(viz.language_distribution(df, title="Review volume by language"), width="stretch")
    capped = int((df["language_name"].value_counts() == 3000).sum())
    if capped:
        st.caption(f"Sampling note: {capped} selected language group(s) contain exactly 3,000 reviews, matching the dataset's per-language sampling cap.")

normalize = st.toggle("Show sentiment by language as percentages", value=True)
st.plotly_chart(viz.sentiment_by_language(df, normalize=normalize, title="Sentiment composition by language"),
                width="stretch")
st.plotly_chart(viz.positive_vs_negative(df, title="Positive and negative review volume by language"),
                width="stretch")

st.markdown('<div class="ms-section-kicker">04 · REVIEW CHARACTERISTICS</div>', unsafe_allow_html=True)
c1, c2 = st.columns([1.05, 1], gap="large")
with c1:
    st.plotly_chart(viz.length_distribution(df, title="Review length distribution"), width="stretch")
with c2:
    st.plotly_chart(viz.length_by_language_box(df, title="Review length by language"), width="stretch")

st.markdown('<div class="ms-section-kicker">05 · MODEL PERFORMANCE</div>', unsafe_allow_html=True)
st.subheader("Model performance comparison")
comp = ui.cached_comparison()
if comp.empty:
    st.info("Run `python -m src.evaluation` to add test-set model performance here.")
else:
    st.plotly_chart(viz.model_comparison_chart(comp), width="stretch")

if choice.startswith("Project"):
    st.divider()
    st.markdown('<div class="ms-section-kicker">06 · DATASET REFERENCE</div>', unsafe_allow_html=True)
    st.subheader("Dataset statistics")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Reviews per language", "Length statistics", "Cleaning log",
                                            "Train/val/test split", "Frequent terms"])
    with tab1:
        st.dataframe(eda.reviews_per_language(df), width="stretch")
        st.dataframe(eda.sentiment_share_by_language(df).style.format("{:.1%}"), width="stretch")
    with tab2:
        st.dataframe(eda.length_statistics(df), width="stretch")
        st.caption("Words for Chinese/Japanese are counted as characters (no spaces between words).")
    with tab3:
        summary = ui.cached_summary()
        st.markdown("Rows removed while building the dataset (from `data/processed/dataset_summary.json`). "
                    "After sampling, the processed data contains no missing texts and no duplicates:")
        st.dataframe(eda.cleaning_report(summary), hide_index=True, width="stretch")
        st.json(eda.duplicate_check(df))
        st.dataframe(eda.missing_values(df).T, width="stretch")
    with tab4:
        st.dataframe(eda.split_by_language(df), width="stretch")
    with tab5:
        code = st.selectbox("Language", sorted(df["language"].unique()), format_func=settings.language_name)
        st.caption("Stop words are removed only here, for display, and only for languages with an NLTK list. "
                   "They are not removed for modelling because lists contain negations such as 'not'/'nicht'.")
        st.dataframe(eda.top_terms_table(df, code), hide_index=True, width="stretch")
