"""
Classical machine-learning baseline: TF-IDF + Logistic Regression / Linear SVM.

Run from the project root:

    python -m src.classical_model

Text representation
-------------------
Each review is cleaned by ``preprocess_for_classical`` (language-aware: Chinese
and Japanese are segmented into characters) and converted into the
concatenation of two TF-IDF matrices:

* word TF-IDF, unigrams + bigrams, with a Unicode-aware tokenizer that keeps
  Indic vowel signs attached to words;
* character TF-IDF, 2-5 character n-grams inside word boundaries
  (``char_wb``). Character n-grams work for every script, share sub-words
  across related languages (e.g. "excel" in excellent/excelente/eccellente)
  and are robust to spelling variation in Hinglish.

Three models are trained
------------------------
1. Shared multilingual Logistic Regression (one model, all languages).
2. Shared multilingual Linear SVM, wrapped in ``CalibratedClassifierCV`` so that
   it outputs probabilities (an SVM by itself only gives margins).
3. Per-language Logistic Regression (one model per language).

The regularisation strength ``C`` is selected on the validation split only;
the test split is never used here.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from config import settings
from src.preprocessing import classical_tokenizer, preprocess_for_classical

logger = logging.getLogger(__name__)

LR_C_GRID = [0.1, 0.25, 0.5, 1.0, 2.0]
SVM_C_GRID = [0.02, 0.05, 0.1, 0.3, 1.0]


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def build_vectorizer(max_word_features: int = 200_000, max_char_features: int = 300_000) -> FeatureUnion:
    word = TfidfVectorizer(tokenizer=classical_tokenizer, token_pattern=None, lowercase=False,
                           ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                           max_features=max_word_features)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=False,
                           min_df=3, max_df=0.9, sublinear_tf=True, max_features=max_char_features)
    return FeatureUnion([("word", word), ("char", char)])


def build_logreg(C: float) -> LogisticRegression:
    # class_weight="balanced" compensates for the smaller neutral class.
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced", solver="lbfgs")


def build_svm(C: float) -> CalibratedClassifierCV:
    # Class balancing is done with sample weights (see fit_classifier) so that it
    # applies to the SVM *and* to the sigmoid calibrators. With class_weight on the
    # SVM only, calibration re-learns the raw class frequencies and the minority
    # neutral class is almost never predicted.
    svm = LinearSVC(C=C, max_iter=5000)
    return CalibratedClassifierCV(svm, method="sigmoid", cv=3)


def fit_classifier(clf, X, y):
    """Fit a classifier; calibrated SVMs receive balanced sample weights."""
    if isinstance(clf, CalibratedClassifierCV):
        return clf.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return clf.fit(X, y)


def prepare_texts(texts, languages) -> list[str]:
    """Apply the language-aware classical preprocessing to every review."""
    if languages is None:
        languages = [None] * len(texts)
    return [preprocess_for_classical(t, l) for t, l in zip(texts, languages)]


def _proba_3class(classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Re-order probability columns into settings.LABELS order.

    A per-language model trained on a language without neutral reviews (Hindi)
    only knows two classes; the missing class receives probability 0.
    """
    out = np.zeros((proba.shape[0], len(settings.LABELS)))
    for j, cls in enumerate(classes):
        out[:, settings.LABEL2ID[cls]] = proba[:, j]
    return out


def tune_and_fit(X_train, y_train, X_val, y_val, make_clf, grid) -> tuple[Pipeline, dict]:
    """Fit the TF-IDF vocabulary once, pick C on validation macro-F1, refit."""
    vec = build_vectorizer()
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)
    scores = {}
    for C in grid:
        clf = fit_classifier(make_clf(C), Xtr, y_train)
        scores[C] = f1_score(y_val, clf.predict(Xva), average="macro")
        logger.info("  C=%-5s val macro-F1=%.4f", C, scores[C])
    best_C = max(scores, key=scores.get)
    t0 = time.perf_counter()
    pipe = Pipeline([("tfidf", build_vectorizer()), ("clf", make_clf(best_C))])
    if isinstance(pipe.named_steps["clf"], CalibratedClassifierCV):
        pipe.fit(X_train, y_train, clf__sample_weight=compute_sample_weight("balanced", y_train))
    else:
        pipe.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    val_pred = pipe.predict(X_val)
    info = {
        "best_C": best_C,
        "val_macro_f1_by_C": {str(k): round(v, 4) for k, v in scores.items()},
        "val_accuracy": round(accuracy_score(y_val, val_pred), 4),
        "val_macro_f1": round(f1_score(y_val, val_pred, average="macro"), 4),
        "train_seconds": round(fit_seconds, 2),
        "n_word_features": len(pipe.named_steps["tfidf"].transformer_list[0][1].vocabulary_),
        "n_char_features": len(pipe.named_steps["tfidf"].transformer_list[1][1].vocabulary_),
        "n_train": len(X_train),
    }
    return pipe, info


# --------------------------------------------------------------------------
# Training entry point
# --------------------------------------------------------------------------
def train_all(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    X_train = prepare_texts(train["text"].tolist(), train["language"].tolist())
    X_val = prepare_texts(val["text"].tolist(), val["language"].tolist())
    # Plain NumPy arrays (pandas 3 returns Arrow-backed arrays from .values).
    y_train = train["sentiment"].to_numpy(dtype=object)
    y_val = val["sentiment"].to_numpy(dtype=object)
    summary: dict = {}

    logger.info("Shared multilingual TF-IDF + Logistic Regression")
    lr, summary[settings.MODEL_LR] = tune_and_fit(X_train, y_train, X_val, y_val, build_logreg, LR_C_GRID)
    joblib.dump(lr, settings.CLASSICAL_LR_PATH, compress=3)

    logger.info("Shared multilingual TF-IDF + Linear SVM (calibrated)")
    svm, summary[settings.MODEL_SVM] = tune_and_fit(X_train, y_train, X_val, y_val, build_svm, SVM_C_GRID)
    joblib.dump(svm, settings.CLASSICAL_SVM_PATH, compress=3)

    logger.info("Per-language TF-IDF + Logistic Regression")
    per_lang, per_lang_info, total_seconds = {}, {}, 0.0
    for lang in sorted(train["language"].unique()):
        tr = train["language"].to_numpy(dtype=object) == lang
        va = val["language"].to_numpy(dtype=object) == lang
        logger.info(" language=%s (n_train=%d)", lang, tr.sum())
        pipe, info = tune_and_fit([X_train[i] for i in np.flatnonzero(tr)], y_train[tr],
                                  [X_val[i] for i in np.flatnonzero(va)], y_val[va],
                                  build_logreg, LR_C_GRID)
        per_lang[lang], per_lang_info[lang] = pipe, info
        total_seconds += info["train_seconds"]
    joblib.dump(per_lang, settings.CLASSICAL_PER_LANG_PATH, compress=3)
    summary[settings.MODEL_PER_LANG] = {"train_seconds": round(total_seconds, 2), "languages": per_lang_info}

    with open(settings.CLASSICAL_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# --------------------------------------------------------------------------
# Inference wrappers
# --------------------------------------------------------------------------
@dataclass
class FeatureContribution:
    feature: str
    kind: str          # "word" or "char"
    tfidf: float
    weight: float      # coefficient for the explained class
    contribution: float


class ClassicalSentimentModel:
    """Loads a saved shared pipeline and offers predict / explain."""

    def __init__(self, pipeline: Pipeline, key: str):
        self.pipeline = pipeline
        self.key = key
        self.name = settings.MODEL_DISPLAY_NAMES[key]

    @classmethod
    def load(cls, key: str = settings.MODEL_LR) -> "ClassicalSentimentModel":
        path = {settings.MODEL_LR: settings.CLASSICAL_LR_PATH,
                settings.MODEL_SVM: settings.CLASSICAL_SVM_PATH}[key]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python -m src.classical_model` first.")
        return cls(joblib.load(path), key)

    def predict_proba(self, texts, languages=None) -> np.ndarray:
        X = prepare_texts(list(texts), languages)
        proba = self.pipeline.predict_proba(X)
        return _proba_3class(self.pipeline.classes_, proba)

    def predict(self, texts, languages=None) -> list[str]:
        return [settings.LABELS[i] for i in self.predict_proba(texts, languages).argmax(1)]

    # ---- explainability ---------------------------------------------------
    def _coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """(coef matrix [n_classes, n_features], classes) of the linear model."""
        clf = self.pipeline.named_steps["clf"]
        if isinstance(clf, CalibratedClassifierCV):
            # Average the LinearSVC weights of the calibration folds.
            coefs = np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)
            return coefs, clf.classes_
        return clf.coef_, clf.classes_

    def explain(self, text: str, language: str | None = None, target: str | None = None,
                top_k: int = 10) -> dict:
        """Features that push the prediction towards / away from ``target``.

        For a linear model the class score is sum(tfidf_value * weight), so the
        product of a feature's TF-IDF value and its class weight is exactly that
        feature's contribution to the score (before the softmax).
        """
        clean = preprocess_for_classical(text, language)
        tfidf = self.pipeline.named_steps["tfidf"]
        x = tfidf.transform([clean]).tocsr()
        names = tfidf.get_feature_names_out()
        coefs, classes = self._coefficients()
        if target is None:
            target = self.predict([text], [language])[0]
        cls_idx = list(classes).index(target) if target in classes else None
        if cls_idx is None:
            return {"target": target, "supporting": [], "opposing": [], "clean_text": clean}
        w = coefs[cls_idx] if coefs.shape[0] > 1 else coefs[0] * (1 if cls_idx == 1 else -1)
        idx, vals = x.indices, x.data
        contrib = vals * w[idx]
        rows = [FeatureContribution(feature=names[i].split("__", 1)[1], kind=names[i].split("__", 1)[0],
                                    tfidf=float(v), weight=float(w[i]), contribution=float(c))
                for i, v, c in zip(idx, vals, contrib)]
        rows.sort(key=lambda r: r.contribution, reverse=True)
        supporting = [r for r in rows if r.contribution > 0][:top_k]
        opposing = sorted([r for r in rows if r.contribution < 0], key=lambda r: r.contribution)[:top_k]
        return {"target": target, "supporting": supporting, "opposing": opposing, "clean_text": clean}


class PerLanguageSentimentModel:
    """One TF-IDF + LR model per language; falls back to the shared model."""

    key = settings.MODEL_PER_LANG
    name = settings.MODEL_DISPLAY_NAMES[settings.MODEL_PER_LANG]

    def __init__(self, models: dict[str, Pipeline], fallback: ClassicalSentimentModel | None):
        self.models = models
        self.fallback = fallback

    @classmethod
    def load(cls) -> "PerLanguageSentimentModel":
        path = settings.CLASSICAL_PER_LANG_PATH
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python -m src.classical_model` first.")
        fallback = ClassicalSentimentModel.load(settings.MODEL_LR) if settings.CLASSICAL_LR_PATH.exists() else None
        return cls(joblib.load(path), fallback)

    def has_language(self, language: str | None) -> bool:
        return language in self.models

    def predict_proba(self, texts, languages=None) -> np.ndarray:
        texts = list(texts)
        languages = list(languages) if languages is not None else [None] * len(texts)
        out = np.zeros((len(texts), len(settings.LABELS)))
        langs = np.array([l if l in self.models else "__fallback__" for l in languages], dtype=object)
        for lang in set(langs):
            mask = langs == lang
            sub_texts = [texts[i] for i in np.flatnonzero(mask)]
            sub_langs = [languages[i] for i in np.flatnonzero(mask)]
            if lang == "__fallback__":
                if self.fallback is None:
                    raise ValueError("No per-language model for these languages and no fallback model.")
                out[mask] = self.fallback.predict_proba(sub_texts, sub_langs)
            else:
                pipe = self.models[lang]
                out[mask] = _proba_3class(pipe.classes_, pipe.predict_proba(prepare_texts(sub_texts, sub_langs)))
        return out

    def predict(self, texts, languages=None) -> list[str]:
        return [settings.LABELS[i] for i in self.predict_proba(texts, languages).argmax(1)]

    def explain(self, text: str, language: str | None = None, target: str | None = None, top_k: int = 10) -> dict:
        if language in self.models:
            return ClassicalSentimentModel(self.models[language], settings.MODEL_LR).explain(
                text, language, target, top_k)
        if self.fallback is not None:
            return self.fallback.explain(text, language, target, top_k)
        return {"target": target, "supporting": [], "opposing": [], "clean_text": text}


def load_training_summary() -> dict:
    if not settings.CLASSICAL_META_PATH.exists():
        return {}
    with open(settings.CLASSICAL_META_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    from src.data_loader import load_processed

    parser = argparse.ArgumentParser(description="Train the TF-IDF classical sentiment models")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train, val = load_processed("train"), load_processed("val")
    summary = train_all(train, val)
    for key in (settings.MODEL_LR, settings.MODEL_SVM):
        s = summary[key]
        print(f"{key}: best C={s['best_C']} val acc={s['val_accuracy']} val macro-F1={s['val_macro_f1']} "
              f"train {s['train_seconds']}s")
    print(f"{settings.MODEL_PER_LANG}: total train {summary[settings.MODEL_PER_LANG]['train_seconds']}s")


if __name__ == "__main__":
    main()
