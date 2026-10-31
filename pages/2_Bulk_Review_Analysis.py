"""Page 2 - polished batch sentiment analysis workspace."""
from __future__ import annotations

import time
import html

import streamlit as st

from config import settings
from src import bulk_analysis as ba
from src import ui
from src import visualization as viz

ui.setup_page("Bulk Review Analysis", ":material/upload_file:")
ui.page_intro(
    "Bulk Review Analysis",
    f"Process up to {settings.BULK_MAX_ROWS:,} customer reviews in one run. Preview your data, configure language handling and models, then export the complete results.",
    "ANALYSIS WORKSPACE / BATCH INTELLIGENCE",
)

st.markdown("""<style>
.bulk-step{display:flex;align-items:center;gap:11px;margin:1.25rem 0 .65rem;color:#344765;font-size:.76rem;font-weight:850;letter-spacing:.11em;text-transform:uppercase}
.bulk-step .num{display:grid;place-items:center;width:30px;height:30px;border-radius:10px;background:#eaf0ff;color:#4169e8;font-size:.82rem;letter-spacing:0}
.bulk-card{background:#fff;border:1px solid #e3e9f3;border-radius:17px;padding:18px 20px;margin:0 0 15px;box-shadow:0 5px 18px rgba(24,39,75,.035)}
.bulk-card-title{font-family:'Manrope',sans-serif;color:#1b2943;font-size:1rem;font-weight:800;margin:0 0 5px}
.bulk-card-copy{color:#71809a;font-size:.84rem;line-height:1.55;margin:0}
.bulk-file-pill{display:inline-block;padding:5px 10px;border-radius:8px;background:#edf3ff;color:#3158bf;font-size:.76rem;font-weight:750}
.bulk-mini-label{font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#71809a;font-weight:800}
.bulk-result-head{background:linear-gradient(110deg,#102342,#274b7e 70%,#5b58a2);border-radius:17px;padding:19px 22px;color:white;margin:1.2rem 0 1rem}
.bulk-result-head h3{color:#fff!important;margin:2px 0 4px;font-size:1.25rem}.bulk-result-head p{color:#dce7ff;margin:0;font-size:.84rem}
.bulk-hint{background:#f5f7fc;border:1px solid #e5eaf3;border-radius:12px;padding:11px 13px;color:#65748d;font-size:.82rem}
[data-testid="stFileUploader"]{min-height:155px}
@media(max-width:700px){.bulk-card{padding:14px}.bulk-result-head{padding:16px}}
</style>""", unsafe_allow_html=True)

SAMPLE_PATH = settings.PROJECT_ROOT / "assets" / "sample_reviews.csv"

# 01 — Choose source
st.markdown('<div class="bulk-step"><span class="num">01</span> Choose your review data</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown('<div class="bulk-card-title">Bring in a review dataset</div><p class="bulk-card-copy">Upload a CSV, TXT or TSV file, or use the included held-out sample to try the complete workflow.</p>', unsafe_allow_html=True)
    source = st.radio("Data source", ["Upload a file", "Use the sample test-set file"], horizontal=True, label_visibility="collapsed")
    raw_bytes, file_name = None, None
    if source == "Upload a file":
        uploaded = st.file_uploader("Drop your dataset here or browse files", type=["csv", "txt", "tsv"],
                                    help="The source file is read for analysis; it is not modified.")
        if uploaded is not None:
            raw_bytes, file_name = uploaded.getvalue(), uploaded.name
    elif SAMPLE_PATH.exists():
        raw_bytes, file_name = SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name
        st.caption("The sample contains real reviews from the held-out test split with dataset labels, so you can compare predictions against known labels.")
    else:
        st.warning("Sample file not found. Run `python -m src.data_loader` or upload your own file.")

if raw_bytes is None:
    st.markdown('<div class="bulk-hint">Select a data source above to continue. Once a file is loaded, you can preview its rows and map the relevant columns.</div>', unsafe_allow_html=True)
    st.stop()

# Read and validate the source file.
try:
    read = ba.read_csv_safely(raw_bytes)
except ValueError as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Could not parse the file as CSV ({type(exc).__name__}: {exc}).")
    st.stop()

df = read.df
for warning in read.warnings:
    st.warning(warning)
if len(df) > settings.BULK_MAX_ROWS:
    st.warning(f"This file contains {len(df):,} rows. The configured limit is {settings.BULK_MAX_ROWS:,}; check the project settings before analysis.")

# File summary strip
f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
f1.markdown(f'<div class="bulk-card"><div class="bulk-mini-label">Selected file</div><div style="font-weight:800;color:#1c2b46;overflow-wrap:anywhere">{html.escape(file_name or "Uploaded dataset")}</div><div class="bulk-card-copy">Encoding: {html.escape(read.encoding)} · Delimiter: {html.escape(repr(read.delimiter))}</div></div>', unsafe_allow_html=True)
f2.metric("Total rows", f"{len(df):,}")
f3.metric("Columns", len(df.columns))
f4.metric("Empty cells", f"{int(df.isna().sum().sum()):,}")
with st.expander("Preview dataset · first 10 rows", expanded=True):
    st.dataframe(df.head(10), width="stretch", hide_index=True)

# 02 — Map data
st.markdown('<div class="bulk-step"><span class="num">02</span> Map your data</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown('<div class="bulk-card-title">Tell LinguaInsight where the review information lives</div><p class="bulk-card-copy">Choose the text column and, optionally, a language or ground-truth label column.</p>', unsafe_allow_html=True)
    cols = list(df.columns)
    c1, c2, c3 = st.columns(3, gap="large")
    guessed_text = ba.guess_text_column(df)
    text_col = c1.selectbox("Review text column", cols, index=cols.index(guessed_text))
    mode_label = c2.radio("Language handling", ["Detect automatically", "Use a language column", "Same language for all rows"])
    lang_col, fixed = None, None
    if mode_label == "Use a language column":
        guess = ba.guess_language_column(df)
        lang_col = c2.selectbox("Language column", cols, index=cols.index(guess) if guess in cols else 0)
        c2.caption("Accepts language codes (hi, pt) or names (Hindi). Empty cells are detected automatically.")
    elif mode_label == "Same language for all rows":
        fixed = c2.selectbox("Select language", list(settings.DATASET_LANGUAGES), format_func=settings.language_name)
    mode = {"Detect automatically": "auto", "Use a language column": "column", "Same language for all rows": "fixed"}[mode_label]
    label_guess = ba.guess_label_column(df)
    label_options = ["(none)"] + cols
    label_col = c3.selectbox("Ground-truth label (optional)", label_options,
                             index=label_options.index(label_guess) if label_guess in cols else 0,
                             help="If present (positive/negative/neutral or 1–5 stars), agreement metrics can be calculated.")
    label_col = None if label_col == "(none)" else label_col

# 03 — Configure models
st.markdown('<div class="bulk-step"><span class="num">03</span> Configure analysis</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown('<div class="bulk-card-title">Select prediction models</div><p class="bulk-card-copy">Choose one or more available models. Results and confidence scores will be added as new columns.</p>', unsafe_allow_html=True)
    avail = ui.cached_available_models()
    options = ui.model_options()
    if not avail.get(settings.MODEL_XLMR_FT) and avail.get(settings.MODEL_XLMR_PRE):
        options = [settings.MODEL_XLMR_FT] + options
    default = [k for k in (settings.MODEL_LR, settings.MODEL_XLMR_FT) if k in options] or options[:1]
    model_keys = st.multiselect("Models to run", options, default=default,
                                format_func=lambda k: settings.MODEL_DISPLAY_NAMES[k],
                                placeholder="Choose at least one model")
    if model_keys:
        st.caption(f"{len(model_keys)} model(s) selected · {len(df):,} input rows · {mode_label.lower()}")
    if any(k in (settings.MODEL_XLMR_FT, settings.MODEL_XLMR_PRE) for k in model_keys):
        from src.transformer_model import get_device
        if get_device().type == "cpu" and len(df) > 2000:
            st.info("The transformer is running on CPU. Large files may take several minutes to process.")

# 04 — Run
st.markdown('<div class="bulk-step"><span class="num">04</span> Run batch analysis</div>', unsafe_allow_html=True)
run_col, run_note = st.columns([1, 2], vertical_alignment="center")
with run_col:
    run_clicked = st.button("✦  Start batch analysis", type="primary", disabled=not model_keys, use_container_width=True)
with run_note:
    st.markdown('<div class="bulk-hint">Your original file remains unchanged. Empty review rows are retained and marked as skipped in the output.</div>', unsafe_allow_html=True)
if run_clicked:
    progress = st.progress(0.0, text="Preparing analysis…")
    started = time.perf_counter()
    try:
        with st.spinner("Loading selected models…"):
            for model_key in model_keys:
                ui.cached_model(model_key if avail.get(model_key) else settings.MODEL_XLMR_PRE)
        results = ba.analyze_dataframe(
            df, text_col, model_keys, language_mode=mode, language_column=lang_col,
            fixed_language=fixed,
            progress_callback=lambda fraction: progress.progress(fraction, text=f"Analysing reviews · {fraction:.0%}"),
        )
    except ValueError as exc:
        progress.empty()
        st.error(str(exc))
        st.stop()
    elapsed = time.perf_counter() - started
    progress.progress(1.0, text=f"Analysis complete · {elapsed:.1f} seconds")
    st.session_state.bulk_results = results
    st.session_state.bulk_meta = {"file": file_name, "text_col": text_col, "label_col": label_col,
                                  "models": model_keys, "elapsed": elapsed}

if "bulk_results" not in st.session_state:
    st.stop()

# 05 — Results
results = st.session_state.bulk_results
meta = st.session_state.bulk_meta
keys = ba.prediction_columns(results)
if not keys:
    st.warning("No prediction columns were returned. Check the selected models and input data.")
    st.stop()
st.markdown('<div class="bulk-result-head"><div class="bulk-mini-label" style="color:#c8d6ff">05 · ANALYSIS COMPLETE</div><h3>Batch results</h3><p>Review-level predictions, sentiment patterns and export-ready data for ' + html.escape(str(meta.get("file", "your dataset"))) + '.</p></div>', unsafe_allow_html=True)
key = st.selectbox("View results for model", keys, format_func=lambda k: settings.MODEL_DISPLAY_NAMES[k])
s = ba.summarize_results(results, key)
metrics = st.columns(5)
metrics[0].metric("Total reviews", f"{s['total_rows']:,}")
metrics[1].metric("Analysed", f"{s['analysed']:,}", delta=f"{s['skipped']} skipped" if s["skipped"] else None, delta_color="off")
metrics[2].metric("Positive", f"{s['positive']:,}")
metrics[3].metric("Negative", f"{s['negative']:,}")
metrics[4].metric("Neutral", f"{s['neutral']:,}")

if meta.get("label_col") and meta["label_col"] in results.columns:
    scores = [(model_key, ba.score_against_labels(results, meta["label_col"], model_key)) for model_key in keys]
    scores = [(model_key, score) for model_key, score in scores if score]
    if scores:
        st.markdown("**Agreement with ground-truth labels**")
        st.dataframe([{"Model": settings.MODEL_DISPLAY_NAMES[model_key], "Rows scored": score["n_scored"],
                       "Accuracy": round(score["accuracy"], 4), "Macro-F1": round(score["macro_f1"], 4)}
                      for model_key, score in scores], hide_index=True, width="stretch")

analysed = results[results["analysis_status"] == "analysed"].copy()
analysed["sentiment"] = analysed[f"{key}_sentiment"]
analysed["length_chars"] = analysed["review_length_chars"]
left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Sentiment mix**")
    st.plotly_chart(viz.sentiment_distribution(analysed, title="Predicted sentiment distribution"), width="stretch")
with right:
    st.markdown("**Language coverage**")
    st.plotly_chart(viz.language_distribution(analysed, title="Reviews by detected/provided language"), width="stretch")
st.markdown("**Sentiment by language**")
st.plotly_chart(viz.sentiment_by_language(analysed, normalize=False, title="Predicted sentiment by language"), width="stretch")

with st.expander("Language-wise sentiment statistics", expanded=False):
    st.dataframe(ba.language_sentiment_table(results, key), width="stretch", hide_index=True)

show_cols = [meta["text_col"], "analysis_status", "language_name", "language_source"] + \
            [column for model_key in keys for column in (f"{model_key}_sentiment", f"{model_key}_confidence")]
if meta.get("label_col"):
    show_cols.insert(1, meta["label_col"])
st.markdown("**Review-level predictions**")
st.caption("Search, sort and inspect individual predictions. The download includes all original columns and model outputs.")
st.dataframe(results[[column for column in show_cols if column in results.columns]], width="stretch", height=390)

csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
download_col, details_col = st.columns([1, 2], vertical_alignment="center")
with download_col:
    st.download_button("↓  Download complete results (CSV)", csv_bytes,
                       file_name=f"sentiment_results_{str(meta['file']).rsplit('.', 1)[0]}.csv",
                       mime="text/csv", type="primary", use_container_width=True)
with details_col:
    st.caption(f"Completed in {meta.get('elapsed', 0):.1f} seconds · {len(results):,} rows exported · Original input columns preserved")
st.caption("Skipped rows (empty text) are retained and marked in `analysis_status`. The CSV also includes language details, per-model predictions, confidence and class probabilities.")
