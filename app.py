"""Multilingual Review Sentiment Analysis System - overview dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings
from src import ui

ui.setup_page("Overview", ":material/home:")

st.markdown(
    """<section class="ms-hero">
      <div class="ms-eyebrow">LANGUAGE INTELLIGENCE / SENTIMENT ANALYTICS</div>
      <h1>Every language has a voice.<br>Understand the sentiment behind it.</h1>
      <p>Explore multilingual customer reviews, compare machine-learning and transformer models,
      and uncover how opinions differ across languages — including Hindi-English code-mixed text.</p>
    </section>""", unsafe_allow_html=True
)

df = ui.dataset_required()
summary = ui.cached_summary()
avail = ui.cached_available_models()
metrics = ui.cached_metrics()

st.markdown('<div class="ms-section-kicker">WORKSPACE OVERVIEW</div>', unsafe_allow_html=True)
st.subheader("Your dataset, at a glance")
if df is not None:
    total = len(df)
    test_n = int((df["split"] == "test").sum())
    lang_n = int(df["language"].nunique())
    source_n = int(df["source"].nunique())
    trained_n = sum(avail.values())
    sentiment_n = df["sentiment"].value_counts()
    k = st.columns(5)
    k[0].metric("Total reviews", f"{total:,}", help="Processed review records in the project dataset")
    k[1].metric("Languages", f"{lang_n:02d}")
    k[2].metric("Data sources", f"{source_n:02d}")
    k[3].metric("Test reviews", f"{test_n:,}")
    k[4].metric("Available models", f"{trained_n:02d}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.05, 1.45], gap="large")
    with left:
        st.markdown('<div class="ms-section-kicker">SENTIMENT SNAPSHOT</div>', unsafe_allow_html=True)
        st.subheader("Review sentiment mix")
        sent = pd.DataFrame({"Sentiment": ["Positive", "Neutral", "Negative"],
                             "Reviews": [int(sentiment_n.get("positive", 0)), int(sentiment_n.get("neutral", 0)), int(sentiment_n.get("negative", 0))]})
        fig = px.pie(sent, names="Sentiment", values="Reviews", hole=.68,
                     color="Sentiment", color_discrete_map={"Positive":"#19a879", "Neutral":"#94a3b8", "Negative":"#e35b68"})
        fig.update_traces(textposition="outside", textinfo="percent", hole=.68,
                          marker=dict(line=dict(color="#ffffff", width=4)))
        fig.update_layout(height=360, margin=dict(t=12,b=35,l=10,r=10), showlegend=True,
                          legend=dict(orientation="h",y=-.13,x=.5,xanchor="center",yanchor="top",
                                      title_text="Sentiment"),
                          annotations=[dict(text=f"{total:,}<br><span style='font-size:11px;color:#8793a8'>REVIEWS</span>",
                                            x=.5,y=.5,showarrow=False,font=dict(size=23,color="#14213a"))])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Green = positive · Grey = neutral · Red = negative. Percentages are calculated from the selected project dataset.")
    with right:
        st.markdown('<div class="ms-section-kicker">GLOBAL COVERAGE</div>', unsafe_allow_html=True)
        st.subheader("Where the reviews come from")
        lang = (df.groupby("language").size().rename("Reviews").reset_index()
                .sort_values("Reviews", ascending=True))
        lang["Language"] = lang["language"].map(settings.language_name)
        fig = px.bar(lang, x="Reviews", y="Language", orientation="h", text="Reviews",
                     color="Reviews", color_continuous_scale=["#b9c9ff", "#416cf3"])
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        fig.update_layout(height=360, margin=dict(t=12,b=35,l=8,r=55),
                          coloraxis_showscale=False, xaxis_title="Review records", yaxis_title=None,
                          plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        max_per_language = int(lang["Reviews"].max()) if not lang.empty else 0
        if max_per_language == 3000:
            st.caption("Several language sources are capped at 3,000 sampled reviews in this prepared dataset; equal bars reflect the sampling design, not live traffic.")
        else:
            st.caption("Languages are ranked by review volume. Counts reflect the processed project dataset.")

    st.markdown('<div class="ms-section-kicker">MODEL INTELLIGENCE</div>', unsafe_allow_html=True)
    st.subheader("Model availability & measured performance")
    rows = []
    best = {k: v["overall"] for k, v in metrics.get("models", {}).items()}
    for key, name in settings.MODEL_DISPLAY_NAMES.items():
        rows.append({"Model": name, "Status": "Available" if avail.get(key) else "Not installed",
                     "Training source": "Project data" if key != settings.MODEL_XLMR_PRE else "External pretrained model",
                     "Test accuracy": round(best[key]["accuracy"], 4) if key in best else None,
                     "Macro F1": round(best[key]["f1_macro"], 4) if key in best else None})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if not metrics:
        st.info("Evaluation results will appear here after running `python -m src.evaluation`.")

    st.markdown('<div class="ms-section-kicker">EXPLORE THE PLATFORM</div>', unsafe_allow_html=True)
    st.subheader("Choose a workspace")
    cards = st.columns(3, gap="medium")
    destinations = [
        ("01", "Single review analysis", "Classify one review, detect its language and inspect model explanations.", "pages/1_Single_Review_Analysis.py"),
        ("02", "Bulk review analysis", "Upload a CSV and analyse thousands of reviews in one run.", "pages/2_Bulk_Review_Analysis.py"),
        ("03", "Multilingual analytics", "Explore sentiment patterns across languages and data sources.", "pages/3_Multilingual_Analytics.py"),
        ("04", "Model evaluation", "Inspect accuracy, macro-F1, confusion matrices and per-language results.", "pages/4_Model_Evaluation.py"),
        ("05", "Error analysis", "Find misclassified reviews and investigate likely error patterns.", "pages/5_Error_Analysis.py"),
        ("06", "Model comparison", "Compare classical TF-IDF models with multilingual XLM-RoBERTa.", "pages/6_Model_Comparison.py"),
    ]
    for i, item in enumerate(destinations):
        with cards[i % 3]:
            ui.card(f"{item[0]}  ·  {item[1]}", f'<div class="ms-muted" style="min-height:48px">{item[2]}</div>')
            st.page_link(item[3], label="Open workspace →")

    with st.expander("Dataset sources, label mapping & research notes"):
        for key, src in summary.get("sources", {}).items():
            st.markdown(f"- **{src['name']}** - [{src['url']}]({src['url']}) - {src['license']}")
        st.markdown("Star ratings (MARC, B2W) are mapped to sentiment as **1-2 → negative, 3 → neutral, 4-5 → positive**; the original rating is retained in `original_label`.")
        st.caption("Dataset labels are not the same as model predictions. Language domains and class distributions differ; consult the evaluation and limitations pages before interpreting scores.")
else:
    st.warning("The processed dataset is not available yet. Run `python -m src.data_loader` from the project root.")
