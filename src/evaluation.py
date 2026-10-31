"""
Model evaluation on the held-out test split.

Run from the project root (after training):

    python -m src.evaluation

Every available model is evaluated on exactly the same test reviews, which
were never used for training or model selection. Results are written to:

* ``data/evaluation/test_predictions.csv``  - one row per test review with the
  gold label and every model's prediction and probabilities;
* ``reports/evaluation_results/metrics.json`` - overall + per-language metrics,
  confusion matrices, classification reports and timing;
* ``reports/evaluation_results/*.csv|*.txt|*.png`` - tables and figures.

Per-language macro-F1 is averaged over the classes that occur in that
language's gold labels (Hindi has no neutral reviews), but a "neutral"
prediction for a Hindi review still counts as an error.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from config import settings

logger = logging.getLogger(__name__)

PREDICTIONS_PATH = settings.EVALUATION_DATA_DIR / "test_predictions.csv"
METRICS_PATH = settings.EVAL_RESULTS_DIR / "metrics.json"
COMPARISON_PATH = settings.EVAL_RESULTS_DIR / "model_comparison.csv"
PER_LANGUAGE_PATH = settings.EVAL_RESULTS_DIR / "per_language_metrics.csv"
LANGID_PATH = settings.EVAL_RESULTS_DIR / "language_detection_accuracy.csv"


def compute_metrics(y_true, y_pred, labels: list[str] | None = None) -> dict:
    """Accuracy, macro/weighted precision-recall-F1, per-class report, confusion matrix."""
    y_true, y_pred = list(y_true), list(y_pred)
    present = labels or [lab for lab in settings.LABELS if lab in set(y_true)]
    out = {"n": len(y_true), "accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average=avg, zero_division=0)
        out.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    out["classification_report"] = classification_report(
        y_true, y_pred, labels=settings.LABELS, output_dict=True, zero_division=0)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=settings.LABELS).tolist()
    out["labels_used_for_macro"] = present
    return out


def _round(obj, nd=4):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj


# --------------------------------------------------------------------------
# Language-identification accuracy on the test set
# --------------------------------------------------------------------------
def evaluate_language_detection(test: pd.DataFrame) -> pd.DataFrame:
    from src.language_detection import detect_language

    detected = [detect_language(t).code for t in test["text"]]
    df = pd.DataFrame({"language": test["language"].to_numpy(dtype=object), "detected": detected})
    df["correct"] = df["language"] == df["detected"]
    table = df.groupby("language").agg(n=("correct", "size"), accuracy=("correct", "mean"))
    table["most_common_wrong"] = df[~df["correct"]].groupby("language")["detected"].agg(
        lambda s: ", ".join(f"{k}:{v}" for k, v in s.fillna("none").value_counts().head(3).items()))
    table.loc["ALL"] = [len(df), df["correct"].mean(), ""]
    return table.reset_index()


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------
def evaluate_all(model_keys: list[str] | None = None, skip_langid: bool = False) -> dict:
    from src.classical_model import load_training_summary
    from src.data_loader import load_processed
    from src.prediction import available_models, predict_with_model
    from src.transformer_model import load_training_history

    test = load_processed("test")
    avail = available_models()
    model_keys = model_keys or [k for k, ok in avail.items() if ok]
    if not model_keys:
        raise RuntimeError("No trained models found. Train models before evaluating.")
    logger.info("Evaluating %s on %d test reviews", model_keys, len(test))

    preds = test[["review_id", "language", "source", "text", "sentiment", "original_label"]].copy()
    preds["length_chars"] = preds["text"].str.len()
    texts, langs = test["text"].tolist(), test["language"].tolist()
    timing = {}
    for key in model_keys:
        t0 = time.perf_counter()
        proba = predict_with_model(key, texts, langs)
        elapsed = time.perf_counter() - t0
        timing[key] = {"total_seconds": elapsed, "ms_per_review": 1000 * elapsed / len(texts)}
        preds[f"pred_{key}"] = [settings.LABELS[i] for i in proba.argmax(1)]
        preds[f"conf_{key}"] = proba.max(1).round(4)
        for j, lab in enumerate(settings.LABELS):
            preds[f"p_{lab}_{key}"] = proba[:, j].round(4)
        logger.info("%s: %.1fs (%.2f ms/review)", key, elapsed, timing[key]["ms_per_review"])

    from src.transformer_model import get_device
    device = str(get_device())

    # ---- metrics -----------------------------------------------------------
    classical_summary = load_training_summary()
    ft_history = load_training_history()
    train_seconds = {k: classical_summary.get(k, {}).get("train_seconds") for k in
                     (settings.MODEL_LR, settings.MODEL_SVM, settings.MODEL_PER_LANG)}
    train_seconds[settings.MODEL_XLMR_FT] = ft_history.get("train_seconds")
    train_seconds[settings.MODEL_XLMR_PRE] = None   # trained by its authors, not here

    results: dict = {"n_test": len(test), "models": {}, "device_for_transformer_inference": device}
    comparison_rows, per_lang_rows = [], []
    for key in model_keys:
        overall = compute_metrics(test["sentiment"], preds[f"pred_{key}"], labels=settings.LABELS)
        per_language = {}
        for lang, grp in preds.groupby("language"):
            m = compute_metrics(grp["sentiment"], grp[f"pred_{key}"])
            per_language[lang] = m
            per_lang_rows.append({"model": key, "language": lang, "n": m["n"], "accuracy": m["accuracy"],
                                  "f1_macro": m["f1_macro"], "f1_weighted": m["f1_weighted"],
                                  "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"]})
        results["models"][key] = {"name": settings.MODEL_DISPLAY_NAMES[key], "overall": overall,
                                  "per_language": per_language, "timing": timing[key],
                                  "train_seconds": train_seconds.get(key)}
        comparison_rows.append({
            "model": key, "model_name": settings.MODEL_DISPLAY_NAMES[key],
            "accuracy": overall["accuracy"], "precision_macro": overall["precision_macro"],
            "recall_macro": overall["recall_macro"], "f1_macro": overall["f1_macro"],
            "f1_weighted": overall["f1_weighted"], "train_seconds": train_seconds.get(key),
            "inference_ms_per_review": timing[key]["ms_per_review"],
            "trained_on_project_data": key != settings.MODEL_XLMR_PRE,
        })
        with open(settings.EVAL_RESULTS_DIR / f"classification_report_{key}.txt", "w", encoding="utf-8") as fh:
            fh.write(f"{settings.MODEL_DISPLAY_NAMES[key]}\nTest reviews: {len(test)}\n\n")
            fh.write(classification_report(test["sentiment"], preds[f"pred_{key}"], labels=settings.LABELS,
                                           digits=4, zero_division=0))

    # ---- disagreement between classical and transformer --------------------
    if settings.MODEL_LR in model_keys and settings.MODEL_XLMR_FT in model_keys:
        disagree = preds[f"pred_{settings.MODEL_LR}"] != preds[f"pred_{settings.MODEL_XLMR_FT}"]
        results["lr_vs_xlmr_disagreement_rate"] = float(disagree.mean())

    if not skip_langid:
        logger.info("Evaluating language detection on the test set")
        langid = evaluate_language_detection(test)
        langid.to_csv(LANGID_PATH, index=False)
        results["language_detection"] = langid.to_dict(orient="records")

    preds.to_csv(PREDICTIONS_PATH, index=False)
    pd.DataFrame(comparison_rows).to_csv(COMPARISON_PATH, index=False)
    pd.DataFrame(per_lang_rows).to_csv(PER_LANGUAGE_PATH, index=False)
    results = _round(results)
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    save_figures(results)
    return results


def save_figures(results: dict) -> None:
    """Static PNG figures for the report (confusion matrices, comparisons)."""
    from src import visualization as viz

    for key, res in results["models"].items():
        fig = viz.confusion_matrix_matplotlib(np.array(res["overall"]["confusion_matrix"]), settings.LABELS,
                                              title=settings.MODEL_DISPLAY_NAMES[key])
        fig.savefig(settings.EVAL_RESULTS_DIR / f"confusion_matrix_{key}.png", dpi=140, bbox_inches="tight")
    comp = pd.read_csv(COMPARISON_PATH)
    fig = viz.model_comparison_matplotlib(comp)
    fig.savefig(settings.EVAL_RESULTS_DIR / "model_comparison.png", dpi=140, bbox_inches="tight")
    per_lang = pd.read_csv(PER_LANGUAGE_PATH)
    fig = viz.per_language_matplotlib(per_lang)
    fig.savefig(settings.EVAL_RESULTS_DIR / "per_language_f1.png", dpi=140, bbox_inches="tight")


# --------------------------------------------------------------------------
# Loading saved results (used by the Streamlit pages)
# --------------------------------------------------------------------------
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_PATH, keep_default_na=False, dtype={"original_label": str})


def load_comparison() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH) if COMPARISON_PATH.exists() else pd.DataFrame()


def load_per_language() -> pd.DataFrame:
    return pd.read_csv(PER_LANGUAGE_PATH) if PER_LANGUAGE_PATH.exists() else pd.DataFrame()


def load_language_detection() -> pd.DataFrame:
    return pd.read_csv(LANGID_PATH, keep_default_na=False) if LANGID_PATH.exists() else pd.DataFrame()


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate all trained models on the test split")
    p.add_argument("--models", nargs="*", default=None, help="model keys (default: all available)")
    p.add_argument("--skip-langid", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = evaluate_all(args.models, args.skip_langid)
    print(pd.read_csv(COMPARISON_PATH).round(4).to_string(index=False))
    if "lr_vs_xlmr_disagreement_rate" in results:
        print("LR vs fine-tuned XLM-R disagreement rate:", results["lr_vs_xlmr_disagreement_rate"])


if __name__ == "__main__":
    main()
