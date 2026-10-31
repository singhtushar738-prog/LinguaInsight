"""
Unified prediction interface used by the Streamlit app, bulk analysis and evaluation.

    from src.prediction import analyze_review
    result = analyze_review("Product bahut accha hai, quality is amazing!",
                            model_keys=["tfidf_logreg", "xlmr_finetuned"])

Every probability returned here comes from ``predict_proba`` of a trained
model; the "confidence" is simply the highest class probability. It measures
how sure the model is, not whether the prediction is correct.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from config import settings
from src.language_detection import LanguageResult, detect_language
from src.preprocessing import is_valid_text

CLASSICAL_KEYS = (settings.MODEL_LR, settings.MODEL_SVM, settings.MODEL_PER_LANG)
TRANSFORMER_KEYS = (settings.MODEL_XLMR_FT, settings.MODEL_XLMR_PRE)


# --------------------------------------------------------------------------
# Model availability and loading
# --------------------------------------------------------------------------
def _pretrained_cached() -> bool:
    """True when the pretrained sentiment model is already in models/hf_cache."""
    folder = settings.HF_CACHE_DIR / "hub" / ("models--" + settings.PRETRAINED_SENTIMENT_MODEL.replace("/", "--"))
    return folder.exists() and any(folder.rglob("config.json"))


def available_models() -> dict[str, bool]:
    """Which models have saved artefacts on disk (no loading involved)."""
    return {
        settings.MODEL_LR: settings.CLASSICAL_LR_PATH.exists(),
        settings.MODEL_SVM: settings.CLASSICAL_SVM_PATH.exists(),
        settings.MODEL_PER_LANG: settings.CLASSICAL_PER_LANG_PATH.exists(),
        settings.MODEL_XLMR_FT: (settings.FINETUNED_MODEL_DIR / "config.json").exists(),
        settings.MODEL_XLMR_PRE: _pretrained_cached(),
    }


@lru_cache(maxsize=None)
def load_model(key: str):
    """Load (and cache in memory) a model by key."""
    if key in (settings.MODEL_LR, settings.MODEL_SVM):
        from src.classical_model import ClassicalSentimentModel
        return ClassicalSentimentModel.load(key)
    if key == settings.MODEL_PER_LANG:
        from src.classical_model import PerLanguageSentimentModel
        return PerLanguageSentimentModel.load()
    if key in TRANSFORMER_KEYS:
        from src.transformer_model import TransformerSentimentModel
        return TransformerSentimentModel.load(key)
    raise ValueError(f"Unknown model key: {key}")


def resolve_model_key(key: str) -> tuple[str, str | None]:
    """Return the key to use and a note when a documented fallback applies.

    If the fine-tuned XLM-R model has not been trained yet, the pretrained
    multilingual XLM-R sentiment model is used instead - never an
    English-only model - and the user is told so.
    """
    avail = available_models()
    if key == settings.MODEL_XLMR_FT and not avail[key]:
        return settings.MODEL_XLMR_PRE, ("The fine-tuned XLM-R model is not available, so the pretrained "
                                         "multilingual XLM-R sentiment model was used instead.")
    return key, None


# --------------------------------------------------------------------------
# Single-review analysis
# --------------------------------------------------------------------------
@dataclass
class ModelPrediction:
    model_key: str
    model_name: str
    sentiment: str
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float
    language_used: str | None
    notes: list[str] = field(default_factory=list)


@dataclass
class ReviewAnalysis:
    text: str
    language: LanguageResult
    language_code: str | None          # language actually used (manual or detected)
    language_source: str               # "automatic" or "manual"
    predictions: list[ModelPrediction]

    @property
    def models_agree(self) -> bool:
        return len({p.sentiment for p in self.predictions}) <= 1


def predict_with_model(key: str, texts: list[str], languages: list[str | None] | None = None,
                       progress_callback=None) -> np.ndarray:
    """Probability matrix (n x 3, settings.LABELS order) from one model."""
    model = load_model(key)
    if key in TRANSFORMER_KEYS:
        return model.predict_proba(texts, progress_callback=progress_callback)
    return model.predict_proba(texts, languages)


def analyze_review(text: str, model_keys: list[str], language: str | None = None) -> ReviewAnalysis:
    """Detect the language and run every requested model on one review.

    ``language`` - a manually selected language code; ``None`` = automatic.
    """
    if not is_valid_text(text):
        raise ValueError("Please enter a review that contains at least one letter or digit.")
    detected = detect_language(text)
    lang_code = language if language else detected.code
    lang_source = "manual" if language else "automatic"

    predictions = []
    for requested in model_keys:
        key, note = resolve_model_key(requested)
        notes = [note] if note else []
        if key == settings.MODEL_PER_LANG and lang_code not in load_model(key).models:
            notes.append(f"No dedicated model for '{settings.language_name(lang_code)}'; "
                         "the shared multilingual Logistic Regression was used.")
        if key in CLASSICAL_KEYS and lang_code and lang_code not in settings.DATASET_LANGUAGES:
            notes.append("This language was not in the training data; TF-IDF features may not transfer.")
        t0 = time.perf_counter()
        proba = predict_with_model(key, [text], [lang_code])[0]
        latency = (time.perf_counter() - t0) * 1000
        idx = int(np.argmax(proba))
        predictions.append(ModelPrediction(
            model_key=key, model_name=settings.MODEL_DISPLAY_NAMES[key], sentiment=settings.LABELS[idx],
            probabilities={lab: float(p) for lab, p in zip(settings.LABELS, proba)},
            confidence=float(proba[idx]), latency_ms=latency, language_used=lang_code, notes=notes))
    return ReviewAnalysis(text=text, language=detected, language_code=lang_code,
                          language_source=lang_source, predictions=predictions)


def explain_prediction(key: str, text: str, language: str | None, target: str | None = None) -> dict:
    """Model-specific explanation (TF-IDF contributions or occlusion importance)."""
    model = load_model(key)
    if key in TRANSFORMER_KEYS:
        return {"type": "occlusion", **model.explain(text, target)}
    return {"type": "tfidf", **model.explain(text, language, target)}
