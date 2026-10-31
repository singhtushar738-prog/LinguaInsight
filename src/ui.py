"""
Shared Streamlit UI helpers (page setup, styling, cards, cached loaders).

Kept separate from the page scripts so every page looks consistent and the
pages themselves only contain page-specific logic.
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from config import settings
from src.visualization import SENTIMENT_COLORS

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--ink:#14213b;--muted:#71809a;--blue:#4169f5;--violet:#7658e8;--line:#e5eaf3;--surface:#fff;--canvas:#f4f6fb;--navy:#0b1932;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:var(--canvas)}
.block-container{width:100%;max-width:1640px;padding:1rem clamp(1rem,2.5vw,2.5rem) 3.5rem}
h1,h2,h3,h4{font-family:'Manrope',sans-serif;letter-spacing:-.035em;color:var(--ink)}
h1{font-size:clamp(1.8rem,2.4vw,2.35rem);font-weight:800;margin:.2rem 0 .45rem}h2{font-weight:800}h3{font-weight:750}
p,li{line-height:1.65}
[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"]{right:1rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#09172f 0%,#142b4b 100%);border-right:1px solid #203958}
[data-testid="stSidebar"] *{color:#edf3ff}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#9db0ce!important}
[data-testid="stSidebarNav"]{display:none}
.ms-nav-label{font-size:.67rem;letter-spacing:.14em;text-transform:uppercase;color:#a9bad4;font-weight:850;margin:1.05rem 0 .3rem;padding-left:.35rem}
[data-testid="stSidebar"] [data-testid="stPageLink"] a{border-radius:10px;padding:.58rem .7rem;font-weight:650;border:1px solid transparent}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover{background:#1d3b63;border-color:#35547b}
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"]{background:#31558c;border-color:#466ca7;color:#fff}
[data-testid="stSidebar"] [data-testid="stExpander"]{background:#172e50;border:1px solid #294365;border-radius:12px}
[data-testid="stMetric"]{background:linear-gradient(145deg,#fff,#fbfcff);border:1px solid var(--line);border-radius:16px;padding:15px 17px;box-shadow:0 5px 18px rgba(21,42,80,.045);min-height:100px;overflow:hidden}
[data-testid="stMetricLabel"]{color:#718096;font-size:.82rem;font-weight:650}
[data-testid="stMetricValue"]{font-family:'Manrope',sans-serif;font-size:clamp(1.25rem,1.8vw,1.85rem);font-weight:800;color:#14213a;white-space:normal;overflow-wrap:anywhere}
[data-testid="stMetricDelta"]{font-size:.78rem}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid #dce4f0;font-weight:700;min-height:2.55rem;transition:.18s ease}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:linear-gradient(105deg,#3867f5,#7657e8);border:0;color:#fff;box-shadow:0 5px 13px #3867f52b}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#8099f7;transform:translateY(-1px)}
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input,[data-testid="stSelectbox"]>div>div,[data-testid="stMultiSelect"]>div>div{border-radius:10px;border-color:#dce4f0;background:#fff}
[data-testid="stFileUploader"]{background:#fff;border:1.5px dashed #b8c7e5;border-radius:16px;padding:1rem;transition:.2s}
[data-testid="stFileUploader"]:hover{border-color:#5578f5;background:#f9faff}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid var(--line);border-radius:13px;overflow:hidden;background:#fff}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:7px;border-bottom:1px solid var(--line);flex-wrap:wrap}
[data-testid="stTabs"] button[role="tab"]{border-radius:9px 9px 0 0;font-weight:650;padding:.55rem .8rem}
[data-testid="stAlert"]{border-radius:12px;border:1px solid var(--line)}
[data-testid="stPlotlyChart"]{background:#fff;border:1px solid var(--line);border-radius:15px;padding:8px 8px 0;box-shadow:0 4px 16px rgba(24,39,75,.035);overflow:hidden}
[data-testid="stPlotlyChart"] iframe{max-width:100%}
hr{border-color:#e6ebf3}
.ms-topbar{display:flex;align-items:center;justify-content:space-between;gap:12px;background:#fff;border:1px solid var(--line);border-radius:14px;padding:11px 17px;margin:0 0 22px;box-shadow:0 3px 12px rgba(20,40,75,.035);max-width:100%;overflow:hidden}
.ms-topbrand{font-weight:800;color:#17233d;letter-spacing:-.03em;white-space:nowrap}.ms-topbrand span{color:#496cf2}.ms-topmeta{color:#7b879b;font-size:.76rem;letter-spacing:.05em;text-align:center}.ms-status{display:inline-flex;align-items:center;gap:7px;background:#effaf4;color:#188453;padding:6px 10px;border-radius:999px;font-size:.72rem;font-weight:750;white-space:nowrap}
.ms-hero{position:relative;overflow:hidden;background:radial-gradient(circle at 88% 12%,rgba(134,112,255,.42),transparent 28%),linear-gradient(112deg,#0d1c38 0%,#1b3766 57%,#554c9b 100%);padding:clamp(24px,3vw,40px);border-radius:20px;color:#fff;margin:0 0 25px;box-shadow:0 15px 35px rgba(25,45,87,.17)}
.ms-hero h1{color:#fff!important;font-size:clamp(1.8rem,3vw,2.65rem);margin:8px 0 10px}.ms-hero p{color:#dce6ff!important;max-width:780px;margin:0;font-size:1rem}.ms-eyebrow{font-size:.69rem;letter-spacing:.15em;text-transform:uppercase;font-weight:800;color:#bfcaff}
.ms-pagehead{padding:4px 0 15px;margin-bottom:18px;border-bottom:1px solid #e7ecf4}.ms-pagehead .ms-eyebrow{color:#7185aa;margin-bottom:5px}.ms-pagehead p{color:#78849a;margin:.2rem 0 0;max-width:950px;font-size:.92rem}
.ms-section-kicker{font-size:.7rem;text-transform:uppercase;letter-spacing:.13em;color:#536784;font-weight:850;margin:1.1rem 0 .15rem}
.ms-card{border:1px solid var(--line);border-radius:15px;padding:1.05rem 1.15rem;margin-bottom:.8rem;background:#fff;box-shadow:0 4px 16px rgba(24,39,75,.04)}.ms-card h4{margin:0 0 .5rem;font-size:1rem;font-weight:750;color:#202b43}
.ms-badge{display:inline-block;padding:.22rem .75rem;border-radius:999px;color:white;font-weight:750;font-size:.82rem;text-transform:capitalize}.ms-muted{color:#758096;font-size:.84rem}.ms-big{font-size:1.75rem;font-weight:800;margin:.2rem 0;color:#17243e}.ms-tag{display:inline-block;padding:2px 9px;margin:2px;border-radius:7px;font-size:.85rem;border:1px solid #e1e6f0}
.ms-panel{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;box-shadow:0 4px 16px rgba(24,39,75,.035);margin:.3rem 0 1rem}.ms-panel-title{font-family:'Manrope',sans-serif;font-size:1rem;font-weight:800;color:#1b2943;margin-bottom:.35rem}
@media(max-width:1050px){.ms-topmeta{display:none}.block-container{padding:1rem 1rem 3rem}}
@media(max-width:700px){.block-container{padding:.8rem .7rem 2rem}.ms-hero{padding:22px;border-radius:15px}.ms-topbar{padding:9px 11px}.ms-status{font-size:.65rem}.ms-pagehead p{font-size:.85rem}}
</style>
"""

TAG_COLORS = {"hi": "rgba(245,133,24,0.25)", "hi-rom": "rgba(245,133,24,0.25)",
              "en": "rgba(76,120,168,0.25)", "other": "rgba(128,128,128,0.12)"}


def setup_page(title: str, icon: str = ":material/translate:") -> None:
    st.set_page_config(page_title=f"{title} | Multilingual Sentiment", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="ms-topbar"><div class="ms-topbrand"><span>✦</span> LINGUA / INSIGHT <span style="font-weight:500;color:#8793a8">· NLP WORKSPACE</span></div><div class="ms-topmeta">MULTILINGUAL REVIEW INTELLIGENCE</div><div class="ms-status">● Models ready to analyse</div></div>""", unsafe_allow_html=True)
    sidebar_status()


def page_intro(title: str, description: str, eyebrow: str = "LINGUAINSIGHT / ANALYTICS") -> None:
    """Render a consistent editorial header for each workspace page."""
    st.markdown(
        f'<div class="ms-pagehead"><div class="ms-eyebrow">{html.escape(eyebrow)}</div>'
        f'<h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_available_models() -> dict[str, bool]:
    from src.prediction import available_models
    return available_models()


def sidebar_status() -> None:
    with st.sidebar:
        st.markdown("<div style='font-size:1.48rem;font-weight:800;letter-spacing:-.04em'>✦ Lingua<span style='color:#8da8ff'>Insight</span></div><div style='font-size:.72rem;letter-spacing:.13em;color:#9db0ce;margin-top:3px'>SENTIMENT INTELLIGENCE</div>", unsafe_allow_html=True)
        st.caption("Multilingual NLP analytics")
        st.markdown("<div style='height:1px;background:#344563;margin:14px 0'></div>", unsafe_allow_html=True)
        st.caption("ETMMTI303 · Language Technologies")
        st.markdown("<div class='ms-nav-label'>Workspace</div>", unsafe_allow_html=True)
        st.page_link("app.py", label="Dashboard", icon=":material/dashboard:")
        st.markdown("<div class='ms-nav-label'>Analysis</div>", unsafe_allow_html=True)
        st.page_link("pages/1_Single_Review_Analysis.py", label="Single review", icon=":material/rate_review:")
        st.page_link("pages/2_Bulk_Review_Analysis.py", label="Bulk review", icon=":material/upload_file:")
        st.page_link("pages/3_Multilingual_Analytics.py", label="Multilingual analytics", icon=":material/bar_chart:")
        st.markdown("<div class='ms-nav-label'>Evaluation</div>", unsafe_allow_html=True)
        st.page_link("pages/4_Model_Evaluation.py", label="Model evaluation", icon=":material/fact_check:")
        st.page_link("pages/5_Error_Analysis.py", label="Error analysis", icon=":material/bug_report:")
        st.page_link("pages/6_Model_Comparison.py", label="Model comparison", icon=":material/compare:")
        st.markdown("<div style='height:1px;background:#344563;margin:15px 0'></div>", unsafe_allow_html=True)
        avail = cached_available_models()
        with st.expander("Model status", expanded=False):
            for key, ok in avail.items():
                st.markdown(f"{'🟢' if ok else '⚪'} {settings.MODEL_DISPLAY_NAMES[key]}")
            if not avail.get(settings.MODEL_XLMR_FT):
                st.caption("Fine-tuned XLM-R not found: the pretrained XLM-R sentiment model is used as a fallback.")


def sentiment_badge(label: str | None) -> str:
    if not label:
        return '<span class="ms-badge" style="background:#888">n/a</span>'
    return f'<span class="ms-badge" style="background:{SENTIMENT_COLORS[label]}">{html.escape(label)}</span>'


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="ms-card"><h4>{html.escape(title)}</h4>{body_html}</div>', unsafe_allow_html=True)


def token_tags_html(token_tags: list[tuple[str, str]]) -> str:
    labels = {"hi": "Hindi", "hi-rom": "Hindi (romanised)", "en": "English", "other": "unknown/other"}
    return " ".join(f'<span class="ms-tag" title="{labels.get(t, t)}" style="background:{TAG_COLORS.get(t)}">'
                    f'{html.escape(tok)}<sub>&nbsp;{html.escape(t)}</sub></span>' for tok, t in token_tags)


@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    from src.data_exploration import add_length_columns
    from src.data_loader import load_processed
    return add_length_columns(load_processed())


@st.cache_data(show_spinner=False)
def cached_summary() -> dict:
    from src.data_loader import load_summary
    return load_summary()


@st.cache_data(show_spinner=False)
def cached_metrics() -> dict:
    from src.evaluation import load_metrics
    return load_metrics()


@st.cache_data(show_spinner=False)
def cached_predictions() -> pd.DataFrame:
    from src.error_analysis import add_cues
    from src.evaluation import load_predictions
    preds = load_predictions()
    return add_cues(preds) if not preds.empty else preds


@st.cache_data(show_spinner=False)
def cached_comparison() -> pd.DataFrame:
    from src.evaluation import load_comparison
    return load_comparison()


@st.cache_data(show_spinner=False)
def cached_per_language() -> pd.DataFrame:
    from src.evaluation import load_per_language
    return load_per_language()


@st.cache_resource(show_spinner=False)
def cached_model(key: str):
    from src.prediction import load_model
    return load_model(key)


def dataset_required() -> pd.DataFrame | None:
    try:
        return cached_dataset()
    except FileNotFoundError:
        st.error("The processed dataset was not found. Run `python -m src.data_loader` from the project root.")
        return None


def evaluation_required() -> bool:
    if not cached_metrics():
        st.warning("No evaluation results yet. Train the models and run `python -m src.evaluation` "
                   "from the project root, then reload this page.")
        return False
    return True


def model_options(include_unavailable: bool = False) -> list[str]:
    avail = cached_available_models()
    keys = list(settings.MODEL_DISPLAY_NAMES)
    if include_unavailable:
        return keys
    usable = [k for k in keys if avail.get(k)]
    # The fine-tuned key stays selectable when only the fallback exists (resolved at prediction time).
    return usable


def short_name(key: str) -> str:
    from src.visualization import SHORT_MODEL_NAMES
    return SHORT_MODEL_NAMES.get(key, key)
