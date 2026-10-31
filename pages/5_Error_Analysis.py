"""Page 5 - error analysis on real held-out test predictions."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings
from src import error_analysis as ea
from src import ui

ui.setup_page("Error Analysis", ":material/bug_report:")
ui.page_intro(
    "Error Analysis",
    "Inspect real held-out reviews the selected model misclassified. Every example and confidence value comes from saved test predictions.",
    "MODEL DIAGNOSTICS / FAILURE ANALYSIS",
)

if not ui.evaluation_required():
    st.stop()

preds = ui.cached_predictions()
keys = ea.model_keys_in(preds)
if not keys:
    st.error("No model prediction columns were found in the saved evaluation file.")
    st.stop()

default = keys.index(settings.MODEL_XLMR_FT) if settings.MODEL_XLMR_FT in keys else 0
key = st.selectbox("Model to analyse", keys, index=default,
                   format_func=lambda k: settings.MODEL_DISPLAY_NAMES.get(k, k))
wrong = preds[f"pred_{key}"] != preds["sentiment"]
error_rows = preds.loc[wrong]

# At-a-glance metrics are calculated directly from the selected model's saved predictions.
c = st.columns(4)
c[0].metric("Test reviews", f"{len(preds):,}", help="Number of held-out reviews in the saved test prediction file.")
c[1].metric("Errors", f"{int(wrong.sum()):,}", help="Reviews where the prediction differs from the dataset sentiment label.")
c[2].metric("Error rate", f"{wrong.mean():.1%}", help="Errors divided by all held-out test reviews.")
mean_error_conf = error_rows[f"conf_{key}"].mean() if not error_rows.empty else 0.0
c[3].metric("Mean confidence on errors", f"{mean_error_conf:.1%}",
            help="Average model confidence on incorrect predictions. Confidence is not the same as correctness.")

# Compact, data-backed context makes the headline error rate easier to interpret.
if not error_rows.empty:
    confusion = ea.error_types(preds, key)
    top_confusion = confusion.iloc[0]
    language_errors = ea.errors_by_language(preds, key)
    top_language = language_errors.iloc[0]
    st.markdown(
        f'<div class="ms-panel"><div class="ms-panel-title">Quick read</div>'
        f'<div class="ms-muted">Most frequent confusion: <b>{top_confusion["true_label"]} → '
        f'{top_confusion["predicted_label"]}</b> ({int(top_confusion["count"]):,} reviews) '
        f'&nbsp; · &nbsp; Highest language error rate: <b>{top_language.name}</b> '
        f'({top_language["error_rate"]:.1%}, n={int(top_language["n"]):,})</div></div>',
        unsafe_allow_html=True,
    )


def show_examples(df: pd.DataFrame, n: int, key_name: str) -> None:
    """Show a compact table plus an explicit full-text inspector for real examples."""
    if df.empty:
        st.info("No examples for this selection.")
        return

    view = df.head(n).copy().reset_index(drop=True)
    view["cues"] = view["cues"].map(lambda values: ", ".join(values) if values else "—")
    view["confidence_pct"] = view[f"conf_{key}"] * 100
    view = view[["language_name", "text", "sentiment", "original_label",
                 f"pred_{key}", "confidence_pct", "cues"]]
    view = view.rename(columns={
        "language_name": "Language", "text": "Review text", "sentiment": "True label",
        "original_label": "Original label", f"pred_{key}": "Predicted",
        "confidence_pct": "Confidence", "cues": "Detected cues",
    })
    st.caption(f"Showing {len(view):,} of {len(df):,} matching errors · sorted by highest confidence first")
    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        height=min(470, 42 + 38 * min(len(view), 11)),
        key=key_name,
        column_config={
            "Language": st.column_config.TextColumn(width="small"),
            "Review text": st.column_config.TextColumn(width="large", help="Select a row below to inspect the complete review."),
            "True label": st.column_config.TextColumn(width="small"),
            "Original label": st.column_config.TextColumn(width="small"),
            "Predicted": st.column_config.TextColumn(width="small"),
            "Confidence": st.column_config.ProgressColumn(
                width="medium", min_value=0, max_value=100, format="%.0f%%",
                help="The model's confidence for its predicted class; not a probability of being correct.",
            ),
            "Detected cues": st.column_config.TextColumn(width="medium"),
        },
    )

    # Streamlit's grid intentionally clips long cells. This inspector exposes the complete source text.
    st.markdown("**Inspect a complete review**")
    options = list(range(len(df.head(n))))
    selected = st.selectbox(
        "Choose an example",
        options,
        key=f"{key_name}_detail_select",
        format_func=lambda i: (
            f"{df.iloc[i]['language_name']} · {df.iloc[i][f'conf_{key}']:.0%} confidence · "
            f"{str(df.iloc[i]['text']).replace(chr(10), ' ')[:82]}"
        ),
        label_visibility="collapsed",
    )
    row = df.iloc[selected]
    with st.container(border=True):
        a, b, d = st.columns(3)
        a.caption(f"True label: **{row['sentiment']}**")
        b.caption(f"Predicted: **{row[f'pred_{key}']}**")
        d.caption(f"Confidence: **{row[f'conf_{key}']:.1%}**")
        st.write(str(row["text"]))
        cues = row["cues"]
        st.caption("Detected cues: " + (", ".join(cues) if cues else "None detected"))


st.markdown('<div class="ms-section-kicker">EXPLORE THE FAILURE MODES</div>', unsafe_allow_html=True)
n_show = st.slider("Examples per table", min_value=5, max_value=100, value=20, step=5)
tabs = st.tabs([
    "Incorrect positive", "Incorrect negative", "Incorrect neutral", "By language",
    "Model disagreements", "Error patterns", "Causes & limitations",
])

for tab, lab in zip(tabs[:3], settings.LABELS[::-1]):
    with tab:
        df = ea.incorrect_predictions(preds, key, predicted=lab)
        st.markdown(
            f"The model predicted **{lab}** when the dataset label was different. "
            f"There are **{len(df):,}** such cases, ordered by confidence."
        )
        by_true = df["sentiment"].value_counts()
        if not by_true.empty:
            st.caption("Actual labels: " + " · ".join(f"{label}: {count:,}" for label, count in by_true.items()))
        show_examples(df, n_show, f"tbl_{lab}")

with tabs[3]:
    el = ea.errors_by_language(preds, key)
    fig = px.bar(
        el.reset_index(), x="language", y="error_rate",
        text=el["error_rate"].map("{:.1%}".format),
        hover_data=["n", "errors"], color_discrete_sequence=["#4169F5"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False,
                      hovertemplate="%{x}<br>Error rate: %{y:.1%}<br>Errors: %{customdata[1]} of %{customdata[0]}<extra></extra>")
    fig.update_layout(
        yaxis_tickformat=".0%", xaxis_title=None, yaxis_title="Error rate",
        height=390, margin=dict(l=15, r=20, t=35, b=15),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#263752"),
    )
    st.plotly_chart(fig, width="stretch")
    lang = st.selectbox("Filter by language", sorted(preds["language"].unique()),
                        format_func=settings.language_name, key="error_language_filter")
    df = ea.incorrect_predictions(preds, key, language=lang)
    st.markdown(f"**{len(df):,}** misclassified {settings.language_name(lang)} reviews.")
    et = ea.error_types(preds[preds["language"] == lang], key)
    if et.empty:
        st.info("No errors for this language.")
    else:
        st.dataframe(et.style.format({"share_of_errors": "{:.1%}"}), hide_index=True, width="stretch")
    show_examples(df, n_show, "tbl_lang")

with tabs[4]:
    other_keys = [k for k in keys if k != key]
    if not other_keys:
        st.info("Only one model was evaluated, so model disagreements cannot be calculated.")
    else:
        default_b = other_keys.index(settings.MODEL_LR) if settings.MODEL_LR in other_keys else 0
        key_b = st.selectbox("Compare with", other_keys, index=default_b,
                             format_func=lambda k: settings.MODEL_DISPLAY_NAMES.get(k, k))
        d = ea.disagreements(preds, key, key_b)
        st.markdown(f"The selected models disagree on **{len(d):,}** test reviews ({len(d) / len(preds):.1%}).")
        who = d["correct_model"].map({key: ui.short_name(key), key_b: ui.short_name(key_b), "neither": "Neither"})
        st.dataframe(who.value_counts().rename("Reviews").to_frame(), width="content")
        summary = ea.disagreement_summary(preds, key, key_b)
        st.dataframe(summary.style.format({"disagreement_rate": "{:.1%}"}), width="stretch")
        filt = st.radio(
            "Show disagreements where",
            ["All", f"Only {ui.short_name(key)} correct", f"Only {ui.short_name(key_b)} correct", "Neither model is correct"],
            horizontal=True,
        )
        if filt.startswith("Only"):
            chosen = key if ui.short_name(key) in filt else key_b
            d = d[d["correct_model"] == chosen]
        elif filt == "Neither model is correct":
            d = d[d["correct_model"] == "neither"]
        view = d.head(n_show)[["language_name", "text", "sentiment", f"pred_{key}", f"pred_{key_b}", "cues"]].copy()
        view["cues"] = view["cues"].map(lambda values: ", ".join(values) if values else "—")
        view.columns = ["Language", "Review text", "True label", ui.short_name(key), ui.short_name(key_b), "Detected cues"]
        st.dataframe(view, hide_index=True, width="stretch", height=min(470, 42 + 38 * min(len(view), 11)),
                     column_config={"Review text": st.column_config.TextColumn(width="large"),
                                    "Detected cues": st.column_config.TextColumn(width="medium")})
        if not d.empty:
            with st.expander("Inspect first disagreement in full"):
                r = d.iloc[0]
                st.write(str(r["text"]))
                st.caption(f"True: {r['sentiment']} · {ui.short_name(key)}: {r[f'pred_{key}']} · "
                           f"{ui.short_name(key_b)}: {r[f'pred_{key_b}']}")

with tabs[5]:
    st.markdown("**Most common confusions (true label → predicted label)**")
    st.dataframe(ea.error_types(preds, key).style.format({"share_of_errors": "{:.1%}"}), hide_index=True, width="stretch")
    st.markdown("**Error rate with vs. without linguistic cues**")
    st.caption(
        "Cues are transparent keyword and length heuristics. They suggest possible error drivers; "
        "they do not prove why an individual prediction was wrong."
    )
    cs = ea.cue_statistics(preds, key)
    if not cs.empty:
        long = cs.melt(id_vars="cue", value_vars=["error_rate_with_cue", "error_rate_without_cue"],
                       var_name="group", value_name="error_rate")
        fig = px.bar(long, x="cue", y="error_rate", color="group", barmode="group",
                     color_discrete_sequence=["#4169F5", "#A8B2C4"],
                     labels={"cue": "Linguistic cue", "error_rate": "Error rate", "group": "Review group"})
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title=None, height=390,
                          margin=dict(l=15, r=20, t=35, b=15), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(cs.style.format({"error_rate_with_cue": "{:.1%}", "error_rate_without_cue": "{:.1%}",
                                      "difference": "{:+.1%}"}), hide_index=True, width="stretch")
    cue = st.selectbox("Inspect errors containing this cue", list(ea.CUE_DESCRIPTIONS),
                       format_func=lambda value: value.replace("_", " ").title())
    st.caption(ea.CUE_DESCRIPTIONS[cue])
    cue_errors = preds[wrong & preds["cues"].map(lambda values: cue in values)].sort_values(f"conf_{key}", ascending=False)
    show_examples(cue_errors, n_show, "tbl_cue")

with tabs[6]:
    st.markdown("### Possible causes of errors")
    for title, description in ea.POSSIBLE_CAUSES.items():
        with st.expander(title):
            st.write(description)
    st.markdown("### Evaluation limitations")
    st.markdown(
        """
- **Labels are proxies.** MARC and B2W labels come from star ratings (1–2 → negative, 3 → neutral, 4–5 → positive). A 3-star review is not always neutral in tone, which limits agreement with the neutral class.
- **Different domains per language.** Amazon products (en/de/es/fr/ja/zh), Brazilian e-commerce (pt), restaurant reviews (it), product/service reviews (hi) and political tweets (Hinglish). Language and domain difficulty are therefore intertwined.
- **Translated text.** Italian reviews are machine-translated from English Yelp reviews and Hindi reviews were translated from English by annotators.
- **Class imbalance.** Hindi has no neutral class; Italian and Portuguese are mostly positive.
- **Sarcasm, irony and implicit sentiment** are not modelled explicitly by either approach.
- **Single test split.** Results come from one stratified split (seed 42); small language subsets have wider uncertainty.
"""
    )
