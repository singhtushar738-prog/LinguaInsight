"""Page 6 - classical TF-IDF models versus multilingual XLM-RoBERTa."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings
from src import ui
from src import visualization as viz

ui.setup_page("Model Comparison", ":material/compare:")
ui.page_intro("Model Comparison", "Compare classical TF-IDF models with multilingual XLM-RoBERTa across predictive quality, language-level performance and computational cost.", "BENCHMARKING / HEAD-TO-HEAD")

if not ui.evaluation_required():
    st.stop()
comp = ui.cached_comparison()
per_lang = ui.cached_per_language()
metrics = ui.cached_metrics()

st.markdown('<div class="ms-section-kicker">01 · SELECT MODELS TO COMPARE</div>', unsafe_allow_html=True)
keys = list(comp["model"])
c1, c2 = st.columns(2)
a_default = keys.index(settings.MODEL_LR) if settings.MODEL_LR in keys else 0
model_a = c1.selectbox("Classical model", keys, index=a_default, format_func=lambda k: settings.MODEL_DISPLAY_NAMES[k])
b_default = keys.index(settings.MODEL_XLMR_FT) if settings.MODEL_XLMR_FT in keys else min(1, len(keys) - 1)
model_b = c2.selectbox("Transformer model", keys, index=b_default, format_func=lambda k: settings.MODEL_DISPLAY_NAMES[k])

ra = comp.set_index("model").loc[model_a]
rb = comp.set_index("model").loc[model_b]
rows = []
for col, label in [("accuracy", "Accuracy"), ("precision_macro", "Precision (macro)"), ("recall_macro", "Recall (macro)"),
                   ("f1_macro", "Macro F1"), ("f1_weighted", "Weighted F1"), ("train_seconds", "Training time (s)"),
                   ("inference_ms_per_review", "Inference time (ms / review)")]:
    va, vb = ra[col], rb[col]
    rows.append({"Metric": label, ui.short_name(model_a): va, ui.short_name(model_b): vb,
                 "Difference (B - A)": (vb - va) if pd.notna(va) and pd.notna(vb) else None})
st.dataframe(pd.DataFrame(rows).style.format(precision=4, na_rep="n/a"), hide_index=True, width="stretch")
st.caption(f"Measured on {metrics['n_test']:,} test reviews. Training times were measured on this machine "
           f"(transformer on {metrics.get('device_for_transformer_inference', 'n/a')}); inference time includes "
           "preprocessing and is measured over the whole test set in batches.")

st.markdown('<div class="ms-section-kicker">02 · LANGUAGE-LEVEL RESULTS</div>', unsafe_allow_html=True)
# ---------------- per-language ---------------------------------------------
st.subheader("Language-wise macro-F1")
pivot = per_lang.pivot(index="language", columns="model", values="f1_macro")
if model_a in pivot and model_b in pivot:
    delta = (pivot[model_b] - pivot[model_a]).rename("difference")
    table = pd.DataFrame({ui.short_name(model_a): pivot[model_a], ui.short_name(model_b): pivot[model_b],
                          "Difference (B - A)": delta,
                          "Test reviews": per_lang.drop_duplicates("language").set_index("language")["n"]})
    table.index = table.index.map(settings.language_name)
    st.dataframe(table.sort_values("Difference (B - A)", ascending=False).style.format(
        {ui.short_name(model_a): "{:.4f}", ui.short_name(model_b): "{:.4f}", "Difference (B - A)": "{:+.4f}"}),
        width="stretch")
    fig = px.bar(table.reset_index(), x="language", y="Difference (B - A)", color="Difference (B - A)",
                 color_continuous_scale=["#D64545", "#EEEEEE", "#2E9E5B"], color_continuous_midpoint=0)
    fig.update_layout(height=360, xaxis_title=None, coloraxis_showscale=False,
                      title=f"Macro-F1 of {ui.short_name(model_b)} minus {ui.short_name(model_a)}")
    st.plotly_chart(fig, width="stretch")

    wins_b = int((delta > 0).sum())
    best_gain = delta.idxmax()
    worst = delta.idxmin()
    st.markdown(
        f"**Measured result:** {ui.short_name(model_b)} has a higher macro-F1 than {ui.short_name(model_a)} in "
        f"**{wins_b} of {len(delta)}** languages. The largest gain is for **{settings.language_name(best_gain)}** "
        f"({delta[best_gain]:+.3f}); the smallest difference (or largest loss) is for "
        f"**{settings.language_name(worst)}** ({delta[worst]:+.3f}). Overall macro-F1: "
        f"{ra['f1_macro']:.3f} vs {rb['f1_macro']:.3f}.")

heat = per_lang.pivot(index="model", columns="language", values="f1_macro")
heat.index = heat.index.map(ui.short_name)
heat.columns = heat.columns.map(settings.language_name)
fig = px.imshow(heat, text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1, aspect="auto")
fig.update_layout(height=120 + 50 * len(heat), title="Macro-F1 of every model in every language",
                  xaxis_title=None, yaxis_title=None)
st.plotly_chart(fig, width="stretch")
st.plotly_chart(viz.model_comparison_chart(comp), width="stretch")

st.markdown('<div class="ms-section-kicker">03 · COMPUTATIONAL COST</div>', unsafe_allow_html=True)
# ---------------- speed ----------------------------------------------------
c1, c2 = st.columns(2)
t = comp.assign(name=comp["model"].map(ui.short_name))
fig = px.bar(t.dropna(subset=["train_seconds"]), x="name", y="train_seconds", text_auto=".0f",
             color_discrete_sequence=["#4C78A8"])
fig.update_layout(height=320, xaxis_title=None, yaxis_title="seconds", title="Training time")
c1.plotly_chart(fig, width="stretch")
fig = px.bar(t, x="name", y="inference_ms_per_review", text_auto=".2f", color_discrete_sequence=["#F58518"])
fig.update_layout(height=320, xaxis_title=None, yaxis_title="ms per review", title="Inference time", yaxis_type="log")
c2.plotly_chart(fig, width="stretch")

st.markdown('<div class="ms-section-kicker">04 · ARCHITECTURAL DIFFERENCES</div>', unsafe_allow_html=True)
# ---------------- explanation ----------------------------------------------
st.subheader("Traditional machine learning vs multilingual transformers")
st.markdown(
    """
| Aspect | TF-IDF + Logistic Regression / SVM | XLM-RoBERTa |
|---|---|---|
| Text representation | Sparse bag of word and character n-grams; word order is lost beyond bigrams | Dense contextual sub-word embeddings; the whole sentence is modelled with self-attention |
| Multilingual knowledge | Only what is in the ~20k training reviews; each language mostly has its own vocabulary | Pre-trained on 2.5 TB of CC-100 text in 100 languages, so related words in different languages get similar representations |
| Negation / context | Captured only when it falls inside an n-gram ("not good") | Can model longer-range context ("I expected it to be good, it was not") |
| Unseen languages | Fails - no features overlap | Can transfer zero-shot, with lower accuracy |
| Training cost | Seconds to minutes on a CPU | Tens of minutes on a GPU; slow on CPU |
| Inference cost | Sub-millisecond per review | Several milliseconds per review on a GPU, more on CPU |
| Model size | Tens of MB | ~1.1 GB |
| Interpretability | Exact: feature weight × TF-IDF value | Needs post-hoc methods (occlusion used here) |
"""
)
st.caption("Whether the transformer is better is an empirical question - the tables above report the measured "
           "test-set results for this dataset; they are not assumed.")
