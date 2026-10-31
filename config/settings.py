"""
Central configuration for the Multilingual Review Sentiment Analysis System.

Every path, label list, model name and hyper-parameter used by the project is
defined here so that the rest of the code (and the viva explanation) has a
single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (everything lives inside the project directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"
NLTK_DATA_DIR = DATA_DIR / "nltk_data"

MODELS_DIR = PROJECT_ROOT / "models"
CLASSICAL_MODEL_DIR = MODELS_DIR / "classical"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "transformer"
FINETUNED_MODEL_DIR = TRANSFORMER_MODEL_DIR / "xlmr-finetuned"
HF_CACHE_DIR = MODELS_DIR / "hf_cache"

REPORTS_DIR = PROJECT_ROOT / "reports"
EVAL_RESULTS_DIR = REPORTS_DIR / "evaluation_results"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

# Hugging Face downloads (models/tokenizers) are cached inside the project so
# they are downloaded only once. This must be set before `transformers` is
# imported anywhere, which is why every module imports `config.settings` first.
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DIR))

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, EVALUATION_DATA_DIR, CLASSICAL_MODEL_DIR,
           TRANSFORMER_MODEL_DIR, HF_CACHE_DIR, EVAL_RESULTS_DIR, FIGURES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Star ratings (MARC, B2W) are converted into sentiment with this documented
# mapping. The original rating is always kept in the `original_label` column.
RATING_TO_SENTIMENT = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
# Languages present in the project dataset (code -> display name).
DATASET_LANGUAGES = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "hi": "Hindi",
    "ja": "Japanese",
    "zh": "Chinese",
    "hi-en": "Hinglish (Hindi-English code-mixed)",
}

# Display names for codes returned by langdetect (subset of its 55 languages).
LANGDETECT_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan",
    "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "et": "Estonian", "fa": "Persian", "fi": "Finnish",
    "fr": "French", "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "kn": "Kannada",
    "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian", "ml": "Malayalam",
    "mr": "Marathi", "ne": "Nepali", "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Tagalog", "tr": "Turkish",
    "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}

# langdetect returns zh-cn / zh-tw; the dataset uses "zh".
LANGDETECT_TO_DATASET = {"zh-cn": "zh", "zh-tw": "zh"}

# XLM-RoBERTa was pre-trained on 100 languages (CC-100). Languages outside the
# project dataset can still be analysed by the transformer, but results are
# not evaluated for them.
def language_name(code: str | None) -> str:
    if not code:
        return "Unknown"
    return DATASET_LANGUAGES.get(code) or LANGDETECT_NAMES.get(code) or code


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
# Maximum number of reviews sampled per language (keeps training CPU/GPU friendly).
SAMPLES_PER_LANGUAGE = 3000
TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION = 0.70, 0.15, 0.15
MIN_REVIEW_CHARS = 3

# ---------------------------------------------------------------------------
# Classical model
# ---------------------------------------------------------------------------
CLASSICAL_LR_PATH = CLASSICAL_MODEL_DIR / "tfidf_logreg.joblib"
CLASSICAL_SVM_PATH = CLASSICAL_MODEL_DIR / "tfidf_svm.joblib"
CLASSICAL_PER_LANG_PATH = CLASSICAL_MODEL_DIR / "tfidf_logreg_per_language.joblib"
CLASSICAL_META_PATH = CLASSICAL_MODEL_DIR / "training_summary.json"

# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------
BASE_TRANSFORMER = "xlm-roberta-base"
# Pretrained (not fine-tuned on this project) multilingual XLM-R sentiment model,
# used as a documented fallback and as an extra comparison point.
PRETRAINED_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

TRANSFORMER_MAX_LENGTH = 160
TRANSFORMER_STRIDE = 32           # token overlap between chunks of long reviews
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EVAL_BATCH_SIZE = 64
TRANSFORMER_LEARNING_RATE = 3e-5
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_EPOCHS = 4
TRANSFORMER_WARMUP_RATIO = 0.1
TRANSFORMER_EARLY_STOPPING_PATIENCE = 1
# Freezing the 192M-parameter embedding matrix lets XLM-R base be fine-tuned on
# a 4 GB GPU. The remaining 86M encoder/classifier parameters are trained.
TRANSFORMER_FREEZE_EMBEDDINGS = True

# Model keys used across the app / evaluation.
MODEL_LR = "tfidf_logreg"
MODEL_SVM = "tfidf_svm"
MODEL_PER_LANG = "tfidf_logreg_per_language"
MODEL_XLMR_FT = "xlmr_finetuned"
MODEL_XLMR_PRE = "xlmr_pretrained_sentiment"

MODEL_DISPLAY_NAMES = {
    MODEL_LR: "TF-IDF + Logistic Regression (shared multilingual)",
    MODEL_SVM: "TF-IDF + Linear SVM (shared multilingual)",
    MODEL_PER_LANG: "TF-IDF + Logistic Regression (one model per language)",
    MODEL_XLMR_FT: "XLM-RoBERTa base (fine-tuned on project data)",
    MODEL_XLMR_PRE: "XLM-RoBERTa sentiment (pretrained, not fine-tuned here)",
}

# Bulk analysis safety limit (rows per upload).
BULK_MAX_ROWS = 20000
