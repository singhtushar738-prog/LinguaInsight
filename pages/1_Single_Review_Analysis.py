"""Page 1 - analyse one review with one or more models."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from src import ui
from src import visualization as viz

ui.setup_page("Single Review Analysis", ":material/rate_review:")
ui.page_intro("Single Review Analysis", "Inspect a customer review, detect its language and understand how each model reaches a sentiment prediction.", "ANALYSIS WORKSPACE / SINGLE REVIEW")
st.markdown("""<style>
.ms-work-card{background:#fff;border:1px solid #e2e8f2;border-radius:18px;padding:18px 20px;margin:8px 0 18px;box-shadow:0 5px 18px rgba(24,39,75,.045)}
.ms-step{display:flex;align-items:center;gap:9px;color:#536784;font-size:.72rem;font-weight:850;letter-spacing:.12em;text-transform:uppercase;margin:3px 0 10px}
.ms-step span{display:inline-grid;place-items:center;width:25px;height:25px;border-radius:8px;background:#edf2ff;color:#4265df;font-size:.76rem;letter-spacing:0}
.ms-result-note{font-size:.78rem;color:#66758d;margin-top:5px}
.ms-result-title{font-size:1.05rem;font-weight:800;color:#192844;margin:0 0 7px}
.ms-review-quote{background:#f6f8fc;border-left:3px solid #7088ee;border-radius:0 10px 10px 0;padding:13px 16px;color:#293750;line-height:1.6}
div[data-testid="stButton"]>button[kind="primary"]{min-height:46px;border-radius:11px;font-weight:750;padding:0 24px}
</style>""", unsafe_allow_html=True)

# Demonstration inputs written for the user interface. They are NOT dataset records
# and are never used for training or evaluation.
DEMO_SAMPLES = {
    "Hinglish (code-mixed)": "Product bahut accha hai, quality is amazing!",
    "Hinglish negative": "Yeh phone bilkul bekar hai, battery ek din bhi nahi chalti",
    "Hindi": "यह फ़ोन बहुत अच्छा है लेकिन बैटरी जल्दी खत्म हो जाती है।",
    "English (mixed opinion)": "The camera is great, but the battery barely lasts half a day.",
    "French": "Très déçu, l'appareil a cessé de fonctionner au bout d'une semaine.",
    "Spanish": "Llegó a tiempo y funciona perfectamente, muy recomendable.",
    "German": "Das Produkt ist okay, aber nichts Besonderes für den Preis.",
    "Italian": "Il servizio era lento e il cibo freddo, non tornerò.",
    "Portuguese": "Ótimo produto, chegou antes do prazo. Recomendo!",
    "Japanese": "思ったより小さかったですが、使いやすいです。",
    "Chinese": "质量很差，用了两天就坏了。",
}

if "single_text" not in st.session_state:
    st.session_state.single_text = DEMO_SAMPLES["Hinglish (code-mixed)"]
    st.session_state.single_gold = None

with st.expander("Load an example", expanded=False):
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Demonstration samples (written for this UI, not from the dataset)", list(DEMO_SAMPLES))
    if c1.button("Use demonstration sample"):
        st.session_state.single_text = DEMO_SAMPLES[choice]
        st.session_state.single_gold = None
    df = ui.dataset_required()
    if df is not None:
        test = df[df["split"] == "test"]
        lang_filter = c2.selectbox("Real test-set review - language", ["any"] + sorted(test["language"].unique()),
                                   format_func=lambda c: "any language" if c == "any" else settings.language_name(c))
        if c2.button("Load random test review"):
            pool = test if lang_filter == "any" else test[test["language"] == lang_filter]
            row = pool.sample(1).iloc[0]
            st.session_state.single_text = row["text"]
            st.session_state.single_gold = {"sentiment": row["sentiment"], "original_label": row["original_label"],
                                            "language": row["language"], "source": row["source"]}

st.markdown('<div class="ms-step"><span>01</span> Review input</div>', unsafe_allow_html=True)
with st.container(border=True):
    text = st.text_area("Paste a customer review", key="single_text", height=155, max_chars=10000,
                        placeholder="Write or paste a review in English, Hindi, Hinglish or another supported language…")
    st.caption("Up to 10,000 characters · Mixed-language reviews are supported")

st.markdown('<div class="ms-step"><span>02</span> Configure analysis</div>', unsafe_allow_html=True)
with st.container(border=True):
    col_a, col_b = st.columns([1, 1.25], gap="large")
    with col_a:
        st.markdown("**Language handling**")
        lang_mode = st.radio("Language", ["Automatic detection", "Manual selection"], horizontal=True)
        manual_lang = None
        if lang_mode == "Manual selection":
            codes = list(settings.DATASET_LANGUAGES) + ["other"]
            manual_lang = st.selectbox("Select language", codes,
                                       format_func=lambda c: "Other / not listed" if c == "other" else settings.language_name(c))
            if manual_lang == "other":
                manual_lang = st.text_input("ISO 639-1 code (e.g. nl, ru, ar)", value="nl").strip().lower() or None
        else:
            st.caption("The system identifies the review language automatically.")
    with col_b:
        st.markdown("**Prediction models**")
        options = ui.model_options()
        avail = ui.cached_available_models()
        if not avail.get(settings.MODEL_XLMR_FT) and avail.get(settings.MODEL_XLMR_PRE):
            options = [settings.MODEL_XLMR_FT] + options   # resolved to the documented fallback
        default = [k for k in (settings.MODEL_LR, settings.MODEL_XLMR_FT) if k in options] or options[:1]
        model_keys = st.multiselect("Choose one or more models", options, default=default, format_func=lambda k: settings.MODEL_DISPLAY_NAMES[k])
        explain = st.checkbox("Generate word-level explanation", value=True, help="Shows which words or features support or oppose the predicted sentiment.")

if not options:
    st.error("No trained models were found. Run `python -m src.classical_model` and/or "
             "`python -m src.transformer_model` from the project root.")
    st.stop()

st.markdown('<div class="ms-step"><span>03</span> Run prediction</div>', unsafe_allow_html=True)
run_col, run_hint = st.columns([1, 2], vertical_alignment="center")
with run_col:
    run_clicked = st.button("✦  Analyse sentiment", type="primary", disabled=not model_keys, use_container_width=True)
with run_hint:
    st.caption(f"{len(model_keys)} model(s) selected · Results include class probabilities and inference time")
if run_clicked:
    from src.prediction import analyze_review, explain_prediction
    try:
        with st.spinner("Loading models and analysing (the first run loads the models into memory)..."):
            for k in model_keys:
                ui.cached_model(k if avail.get(k) else settings.MODEL_XLMR_PRE)
            result = analyze_review(text, model_keys, language=manual_lang)
            explanations = {}
            if explain:
                for p in result.predictions:
                    explanations[p.model_key] = explain_prediction(p.model_key, text, result.language_code, p.sentiment)
        st.session_state.single_result = (result, explanations, st.session_state.get("single_gold"))
    except ValueError as exc:
        st.error(str(exc))
        st.session_state.pop("single_result", None)

if "single_result" not in st.session_state:
    st.stop()

result, explanations, gold = st.session_state.single_result
if result.text != text:
    st.info("The text has changed since the last analysis - press **Analyse sentiment** to update.")

st.markdown('<div class="ms-step"><span>04</span> Analysis results</div>', unsafe_allow_html=True)
st.markdown('<div class="ms-result-title">Review analysed</div>', unsafe_allow_html=True)
st.markdown(f'<div class="ms-review-quote">{__import__("html").escape(result.text)}</div>', unsafe_allow_html=True)
if gold and gold.get("sentiment"):
    st.caption(f"Real test-set record from **{gold['source']}** - dataset label: **{gold['sentiment']}** "
               f"(original label: {gold['original_label']})")

# ---------------- language -------------------------------------------------
lang = result.language
st.markdown("**Language intelligence**")
c1, c2, c3 = st.columns(3)
c1.metric("Detected language", lang.name)
c2.metric("Detection confidence", f"{lang.confidence:.0%}" if lang.code else "-")
c3.metric("Language used for analysis",
          settings.language_name(result.language_code) if result.language_code else "Unknown",
          help="Manual selection overrides detection. The language controls TF-IDF preprocessing "
               "(e.g. character segmentation for Chinese/Japanese) and the per-language model.")
st.caption(f"Method: {lang.method}. Source: {result.language_source}.")
for msg in lang.messages:
    st.warning(msg)
with st.expander("Language detection details"):
    if lang.candidates:
        st.markdown("**langdetect candidates:** " + ", ".join(
            f"{settings.language_name(c)} ({p:.1%})" for c, p in lang.candidates))
    if lang.scripts:
        st.markdown("**Script profile:** " + ", ".join(f"{s} {v:.0%}" for s, v in lang.scripts.items()))
    if lang.is_code_mixed:
        st.markdown("**Components:** " + ", ".join(lang.mixed_components))
    if lang.token_tags:
        st.markdown("**Token-level tags** (lexicon learned from SentiMix token annotations; "
                    "`other` = not in the lexicon):")
        st.markdown(ui.token_tags_html(lang.token_tags), unsafe_allow_html=True)

# ---------------- predictions ----------------------------------------------
st.markdown('<div class="ms-step"><span>05</span> Model predictions</div>', unsafe_allow_html=True)
st.subheader("Sentiment results")
if len(result.predictions) > 1:
    if result.models_agree:
        st.success(f"All models agree: **{result.predictions[0].sentiment}**")
    else:
        st.warning("The models disagree: " + ", ".join(
            f"{ui.short_name(p.model_key)} → **{p.sentiment}**" for p in result.predictions))

cols = st.columns(min(len(result.predictions), 3))
for i, p in enumerate(result.predictions):
    with cols[i % len(cols)]:
        probs = " · ".join(f"{lab} {p.probabilities[lab]:.1%}" for lab in settings.LABELS)
        st.markdown(f'<div class="ms-work-card"><div class="ms-result-title">{__import__("html").escape(p.model_name)}</div>'
                    f'{ui.sentiment_badge(p.sentiment)}'
                    f'<div class="ms-big">{p.confidence:.1%}</div>'
                    f'<div class="ms-muted">Highest class probability · not model accuracy</div>'
                    f'<div class="ms-result-note">{probs}</div>'
                    f'<div class="ms-result-note">Inference time: {p.latency_ms:.0f} ms</div></div>', unsafe_allow_html=True)
        st.plotly_chart(viz.probability_bars(p.probabilities), width="stretch", key=f"prob_{i}")
        for n in p.notes:
            st.info(n)

if len(result.predictions) > 1:
    rows = [{"model": ui.short_name(p.model_key), **p.probabilities} for p in result.predictions]
    st.plotly_chart(viz.model_probability_comparison(rows), width="stretch")
    st.dataframe(pd.DataFrame([{
        "Model": p.model_name, "Prediction": p.sentiment, "Confidence": round(p.confidence, 4),
        **{f"P({lab})": round(p.probabilities[lab], 4) for lab in settings.LABELS},
        "Time (ms)": round(p.latency_ms, 1)} for p in result.predictions]), hide_index=True, width="stretch")

st.caption("Confidence is the model's highest class probability. It describes how certain the model is, "
           "not whether the prediction is correct - see the Model Evaluation page for measured accuracy.")

# ---------------- explanations ---------------------------------------------
if explanations:
    st.markdown('<div class="ms-step"><span>06</span> Explainability</div>', unsafe_allow_html=True)
    st.subheader("Why did the models predict this?")
    for p in result.predictions:
        exp = explanations.get(p.model_key)
        if not exp:
            continue
        with st.expander(f"{p.model_name} → {p.sentiment}", expanded=True):
            if exp["type"] == "tfidf":
                st.markdown(
                    "For a linear TF-IDF model the class score is the sum of *TF-IDF value × learned weight* over "
                    "all features, so each product is exactly that feature's contribution. `char` features are "
                    "character n-grams (spaces shown as quotes).")
                sup = [{"feature": f"{r.feature} [{r.kind}]", "contribution": r.contribution} for r in exp["supporting"]]
                opp = [{"feature": f"{r.feature} [{r.kind}]", "contribution": r.contribution} for r in exp["opposing"]]
                c1, c2 = st.columns(2)
                fig = viz.feature_contributions(sup, f"Features supporting '{exp['target']}'")
                if fig:
                    c1.plotly_chart(fig, width="stretch", key=f"sup_{p.model_key}")
                fig = viz.feature_contributions(opp, f"Features against '{exp['target']}'")
                if fig:
                    c2.plotly_chart(fig, width="stretch", key=f"opp_{p.model_key}")
                if not sup and not opp:
                    st.info("None of the words in this review were in the model's vocabulary.")
                st.caption(f"Preprocessed text seen by the model: `{exp['clean_text']}`")
            else:
                st.markdown(
                    "**Occlusion analysis:** each word is removed in turn and the change in the probability of "
                    f"**{exp['target']}** is measured (green = removing the word lowers that probability, i.e. "
                    "the word supported the prediction; red = it pushed against it). This measures the model's "
                    "actual output, unlike attention weights, which alone do not explain a prediction.")
                st.markdown(viz.token_importance_html(exp["tokens"], result.text), unsafe_allow_html=True)
                st.caption("Colours are scaled to the most important word. When the model is very confident, "
                           "removing a single word changes the probability only slightly (saturation), so look "
                           "at the ranking rather than the absolute values.")
                if exp["tokens"]:
                    top = sorted(exp["tokens"], key=lambda t: -abs(t["importance"]))[:8]
                    st.dataframe(pd.DataFrame(top).rename(columns={"importance": "probability drop when removed"}),
                                 hide_index=True)
