"""Tests for the TF-IDF classical models.

A tiny hand-written fixture is used so the tests do not depend on the trained
artefacts; it is test data only and is never used as project data.
"""
import numpy as np
import pytest

from config import settings
from src.classical_model import (ClassicalSentimentModel, _proba_3class, build_logreg,
                                 build_vectorizer, prepare_texts)
from sklearn.pipeline import Pipeline

FIXTURE = [
    ("great product, works perfectly", "positive"), ("excellent quality, very happy", "positive"),
    ("love it, great value", "positive"), ("perfect, highly recommend", "positive"),
    ("terrible, broke after one day", "negative"), ("awful quality, waste of money", "negative"),
    ("very bad, do not buy", "negative"), ("horrible product, broke quickly", "negative"),
    ("it is okay, nothing special", "neutral"), ("average product, fine for the price", "neutral"),
    ("okay but could be better", "neutral"), ("fine, nothing special", "neutral"),
] * 3


@pytest.fixture(scope="module")
def tiny_model():
    texts, labels = zip(*FIXTURE)
    pipe = Pipeline([("tfidf", build_vectorizer()), ("clf", build_logreg(5.0))])
    pipe.set_params(tfidf__word__min_df=1, tfidf__char__min_df=1, tfidf__word__max_df=1.0, tfidf__char__max_df=1.0)
    pipe.fit(prepare_texts(list(texts), ["en"] * len(texts)), list(labels))
    return ClassicalSentimentModel(pipe, settings.MODEL_LR)


def test_probabilities_are_valid(tiny_model):
    proba = tiny_model.predict_proba(["great quality", "broke, terrible"], ["en", "en"])
    assert proba.shape == (2, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert tiny_model.predict(["great quality", "terrible, broke"], ["en", "en"]) == ["positive", "negative"]


def test_explanation_is_exact_for_linear_model(tiny_model):
    text = "great quality but broke"
    exp = tiny_model.explain(text, "en", target="positive", top_k=10_000)
    clf = tiny_model.pipeline.named_steps["clf"]
    x = tiny_model.pipeline.named_steps["tfidf"].transform(prepare_texts([text], ["en"]))
    k = list(clf.classes_).index("positive")
    score = (x @ clf.coef_[k])[0] + clf.intercept_[k]
    total = sum(r.contribution for r in exp["supporting"] + exp["opposing"]) + clf.intercept_[k]
    assert np.isclose(score, total)
    assert any("great" in r.feature for r in exp["supporting"])


def test_two_class_probabilities_are_mapped_to_three():
    out = _proba_3class(np.array(["negative", "positive"]), np.array([[0.2, 0.8]]))
    assert out.tolist() == [[0.2, 0.0, 0.8]]


def test_saved_model_if_available():
    if not settings.CLASSICAL_LR_PATH.exists():
        pytest.skip("trained model not available")
    model = ClassicalSentimentModel.load(settings.MODEL_LR)
    proba = model.predict_proba(["Excellent produit, je recommande", "Muy malo, no funciona"], ["fr", "es"])
    assert np.allclose(proba.sum(axis=1), 1.0)
