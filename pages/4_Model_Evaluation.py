"""Page 4 - test-set evaluation of every model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from src import ui
from src import visualization as viz

ui.setup_page("Model Evaluation", ":material/fact_check:")
ui.page_intro(
    "Model Evaluation",
    "A transparent benchmark of every sentiment model on the same unseen test reviews.",
    "EXPERIMENT RESULTS / MODEL QUALITY",
)

if not ui.evaluation_required():
    st.stop()

metrics = ui.cached_metrics()
comp = ui.cached_comparison()
per_lang = ui.cached_per_language()
models = metrics["models"]

st.markdown(
    f"<div class='ms-note'>All models are evaluated on <b>{metrics['n_test']:,} held-out reviews</b>. "
    "The test split was not used for model training, hyper-parameter selection or early stopping.</div>",
    unsafe_allow_html=True,
)

# ---------------- benchmark overview ----------------
st.markdown('<div class="ms-section-kicker">01 · OVERALL BENCHMARK</div>', unsafe_allow_html=True)
st.subheader("How the models perform")

ranked = comp.sort_values("f1_macro", ascending=False).reset_index(drop=True)
best = ranked.iloc[0]
fastest = comp.sort_values("inference_ms_per_review", ascending=True).iloc[0]
most_accurate = comp.sort_values("accuracy", ascending=False).iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Test reviews", f"{metrics['n_test']:,}")
k2.metric("Top macro-F1", f"{best['f1_macro']:.3f}", str(best.get("model_name", best.get("model", "Best model"))))
k3.metric("Top accuracy", f"{most_accurate['accuracy']:.3f}", str(most_accurate.get("model_name", most_accurate.get("model", "Best model"))))
k4.metric("Fastest inference", f"{fastest['inference_ms_per_review']:.2f} ms/review", str(fastest.get("model_name", fastest.get("model", "Fastest model"))))

with st.expander("What do these metrics mean?", expanded=False):
    st.markdown(
        "- **Accuracy:** share of all predictions that are correct.\n"
        "- **Macro-F1:** F1 is calculated for each class and averaged equally; useful when classes are imbalanced.\n"
        "- **Weighted-F1:** class F1 scores are averaged according to class frequency.\n"
        "- **Inference time:** average time to process one review; lower is faster."
    )

st.dataframe(
    comp[["model_name", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
          "train_seconds", "inference_ms_per_review"]].rename(columns={
        "model_name": "Model", "accuracy": "Accuracy", "precision_macro": "Precision · macro",
        "recall_macro": "Recall · macro", "f1_macro": "F1 · macro", "f1_weighted": "F1 · weighted",
        "train_seconds": "Training time (s)", "inference_ms_per_review": "Inference (ms/review)"
    }).style.format({c: "{:.4f}" for c in ["Accuracy", "Precision · macro", "Recall · macro", "F1 · macro", "F1 · weighted"]}
                    | {"Training time (s)": "{:.1f}", "Inference (ms/review)": "{:.2f}"}, na_rep="n/a")
      .highlight_max(subset=["Accuracy", "Precision · macro", "Recall · macro", "F1 · macro", "F1 · weighted"],
                     color="rgba(46,158,91,0.18)"),
    hide_index=True, width="stretch"
)
st.plotly_chart(viz.model_comparison_chart(comp), width="stretch")
st.caption(
    "Macro averages give every class equal weight; weighted averages reflect class frequency. "
    "The pretrained XLM-R sentiment model was trained by its authors, so project training time is not available. "
    f"Transformer inference device: {metrics.get('device_for_transformer_inference', 'n/a')}."
)

# ---------------- diagnostics ----------------
st.markdown('<div class="ms-section-kicker">02 · MODEL DIAGNOSTICS</div>', unsafe_allow_html=True)
st.subheader("Inspect a model in detail")
key = st.selectbox("Choose a model", list(models), format_func=lambda k: settings.MODEL_DISPLAY_NAMES[k])
res = models[key]

left, right = st.columns([1.05, 1])
with left:
    normalize = st.toggle("Normalize confusion matrix by true class", value=True)
    st.plotly_chart(
        viz.confusion_matrix_plotly(np.array(res["overall"]["confusion_matrix"]), settings.LABELS,
                                    f"{ui.short_name(key)} · confusion matrix", normalize),
        width="stretch"
    )
with right:
    st.markdown("**Classification report**")
    rep = res["overall"]["classification_report"]
    rows = [{"Class": lab, **{k: rep[lab][k] for k in ("precision", "recall", "f1-score", "support")}}
            for lab in settings.LABELS]
    for avg in ("macro avg", "weighted avg"):
        rows.append({"Class": avg, **{k: rep[avg][k] for k in ("precision", "recall", "f1-score", "support")}})
    st.dataframe(pd.DataFrame(rows).style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                                  "f1-score": "{:.3f}", "support": "{:.0f}"}),
                 hide_index=True, width="stretch")
    st.metric("Model accuracy", f"{res['overall']['accuracy']:.2%}")

st.markdown("**Performance by language**")
pl = per_lang[per_lang["model"] == key].copy()
pl["language"] = pl["language"].map(settings.language_name)
st.dataframe(
    pl.drop(columns="model").sort_values("f1_macro", ascending=False).style.format(
        {c: "{:.3f}" for c in ("accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro")}
    ), hide_index=True, width="stretch"
)
st.caption(
    "Per-language macro-F1 averages over the classes present in that language. Hindi has no neutral class in this dataset; "
    "a neutral prediction for a Hindi review is therefore counted as an error."
)
lang = st.selectbox("Inspect confusion matrix for a language", sorted(res["per_language"]),
                    format_func=settings.language_name)
st.plotly_chart(
    viz.confusion_matrix_plotly(np.array(res["per_language"][lang]["confusion_matrix"]), settings.LABELS,
                                f"{settings.language_name(lang)} · {ui.short_name(key)}"),
    width="stretch"
)

# ---------------- cross-language benchmark ----------------
st.markdown('<div class="ms-section-kicker">03 · CROSS-LANGUAGE PERFORMANCE</div>', unsafe_allow_html=True)
st.subheader("Where does each model perform best?")
metric = st.segmented_control("Compare using", ["Macro-F1", "Accuracy", "Weighted-F1"], default="Macro-F1")
metric_key = {"Macro-F1": "f1_macro", "Accuracy": "accuracy", "Weighted-F1": "f1_weighted"}[metric]
st.plotly_chart(viz.per_language_chart(per_lang, metric_key), width="stretch")
st.caption("Compare languages carefully: dataset size and label distribution differ by language.")

# ---------------- reproducibility ----------------
st.markdown('<div class="ms-section-kicker">04 · TRAINING & REPRODUCIBILITY</div>', unsafe_allow_html=True)
st.subheader("How the models were trained")
from src.classical_model import load_training_summary
from src.transformer_model import load_training_history

tab1, tab2, tab3 = st.tabs(["XLM-R fine-tuning", "Classical models", "Language detection"])
with tab1:
    hist = load_training_history()
    if not hist:
        st.info("No fine-tuning history found (models/transformer/xlmr-finetuned/training_history.json).")
    else:
        cfg = hist["config"]
        c = st.columns(5)
        c[0].metric("Best epoch", hist.get("best_epoch"))
        c[1].metric("Best validation macro-F1", f"{hist.get('best_val_macro_f1', 0):.3f}")
        c[2].metric("Training time", f"{hist.get('train_seconds', 0) / 60:.1f} min")
        c[3].metric("Batch size", cfg["batch_size"])
        c[4].metric("Learning rate", f"{cfg['learning_rate']:.0e}")
        fig = viz.training_curves(hist)
        if fig:
            st.plotly_chart(fig, width="stretch")
        fig = viz.step_loss_curve(hist)
        if fig:
            st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame(hist["history"]["epochs"]), hide_index=True, width="stretch")
        with st.expander("Full training configuration"):
            st.json(cfg)
with tab2:
    summ = load_training_summary()
    for model_key in (settings.MODEL_LR, settings.MODEL_SVM):
        if model_key in summ:
            summary = summ[model_key]
            st.markdown(f"**{settings.MODEL_DISPLAY_NAMES[model_key]}**")
            a, b, c = st.columns(3)
            a.metric("Best C", summary["best_C"])
            b.metric("Word features", f"{summary['n_word_features']:,}")
            c.metric("Character features", f"{summary['n_char_features']:,}")
            st.caption(f"Training time: {summary['train_seconds']:.1f} seconds")
            st.dataframe(pd.DataFrame([summary["val_macro_f1_by_C"]], index=["Validation macro-F1"]), width="stretch")
    if settings.MODEL_PER_LANG in summ:
        rows = [{"Language": settings.language_name(language), "Best C": value["best_C"],
                 "Training reviews": value["n_train"], "Validation macro-F1": value["val_macro_f1"]}
                for language, value in summ[settings.MODEL_PER_LANG]["languages"].items()]
        st.markdown("**Per-language Logistic Regression models**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
with tab3:
    from src.evaluation import load_language_detection
    lid = load_language_detection()
    if lid.empty:
        st.info("Language detection was not evaluated.")
    else:
        lid["language"] = lid["language"].map(
            lambda code: "All test reviews" if code == "ALL" else settings.language_name(code)
        )
        st.dataframe(lid.style.format({"accuracy": "{:.1%}"}), hide_index=True, width="stretch")
        st.caption("Automatic language detection (langdetect + Hinglish heuristic) compared with dataset language. "
                   "'hi-en' is counted as correct only when the code-mix heuristic fires.")
