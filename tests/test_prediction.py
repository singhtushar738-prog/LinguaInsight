"""Tests for the unified prediction interface."""
import numpy as np
import pytest

from config import settings
from src import prediction


def test_empty_review_is_rejected():
    with pytest.raises(ValueError):
        prediction.analyze_review("   ", [settings.MODEL_LR])


def test_fallback_to_pretrained_transformer(monkeypatch):
    avail = {k: True for k in settings.MODEL_DISPLAY_NAMES}
    avail[settings.MODEL_XLMR_FT] = False
    monkeypatch.setattr(prediction, "available_models", lambda: avail)
    key, note = prediction.resolve_model_key(settings.MODEL_XLMR_FT)
    assert key == settings.MODEL_XLMR_PRE and "pretrained" in note
    assert prediction.resolve_model_key(settings.MODEL_LR) == (settings.MODEL_LR, None)


class _FakeModel:
    def predict_proba(self, texts, languages=None):
        return np.tile([0.1, 0.2, 0.7], (len(texts), 1))


def test_analyze_review_structure(monkeypatch):
    monkeypatch.setattr(prediction, "load_model", lambda key: _FakeModel())
    res = prediction.analyze_review("Product bahut accha hai, quality is amazing!", [settings.MODEL_LR])
    p = res.predictions[0]
    assert res.language_code == "hi-en" and res.language_source == "automatic"
    assert p.sentiment == "positive" and p.confidence == pytest.approx(0.7)
    assert sum(p.probabilities.values()) == pytest.approx(1.0)


def test_manual_language_overrides_detection(monkeypatch):
    monkeypatch.setattr(prediction, "load_model", lambda key: _FakeModel())
    res = prediction.analyze_review("Das ist gut", [settings.MODEL_LR], language="de")
    assert res.language_code == "de" and res.language_source == "manual"


def test_real_models_if_available():
    avail = prediction.available_models()
    keys = [k for k in (settings.MODEL_LR, settings.MODEL_XLMR_FT) if avail[k]]
    if not keys:
        pytest.skip("no trained models available")
    res = prediction.analyze_review("Ótimo produto, chegou antes do prazo. Recomendo!", keys)
    for p in res.predictions:
        assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
        assert p.confidence == pytest.approx(max(p.probabilities.values()))
