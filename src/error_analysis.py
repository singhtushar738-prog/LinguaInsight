"""
Error analysis on the real test-set predictions written by ``src.evaluation``.

    python -m src.error_analysis        # writes summary + example tables to reports/

Nothing here is generated: every example is an actual test review together
with the actual model prediction. To help explain *why* a model may have
failed, each review is tagged with simple, transparent linguistic cues
(negation words, contrast words, code-mixing, length, 3-star "neutral"
labels). These cues are heuristics - they suggest a possible cause, they
do not prove it - and the error rate *with* vs *without* each cue is
reported so the effect can be checked quantitatively.
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from config import settings
from src.preprocessing import WORD_RE

# Negation and contrast markers per language (lower-case).
NEGATION_WORDS = {
    "en": {"not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "without", "cannot",
           "don't", "doesn't", "didn't", "isn't", "wasn't", "won't", "can't", "couldn't", "wouldn't"},
    "de": {"nicht", "kein", "keine", "keinen", "keiner", "nie", "niemals", "nichts", "ohne"},
    "es": {"no", "nunca", "nada", "ni", "sin", "jamás", "tampoco", "ningún", "ninguna"},
    "fr": {"ne", "pas", "jamais", "rien", "aucun", "aucune", "sans", "ni", "n'est", "n'a"},
    "it": {"non", "mai", "niente", "nulla", "nessuno", "nessuna", "senza", "né"},
    "pt": {"não", "nao", "nunca", "nada", "nem", "sem", "jamais", "nenhum", "nenhuma"},
    "hi": {"नहीं", "न", "मत", "बिना", "कभी"},
    "hi-en": {"nahi", "nhi", "nahin", "nai", "na", "mat", "bina", "not", "no", "never"},
}
CJK_NEGATION = {"ja": ["ない", "ません", "なかった", "ず"], "zh": ["不", "没", "沒", "别", "無", "无"]}
CONTRAST_WORDS = {
    "en": {"but", "however", "although", "though", "yet"},
    "de": {"aber", "jedoch", "obwohl", "trotzdem", "sondern"},
    "es": {"pero", "aunque", "sino"},
    "fr": {"mais", "cependant", "pourtant", "quoique"},
    "it": {"ma", "però", "tuttavia", "comunque"},
    "pt": {"mas", "porém", "contudo", "embora", "entretanto"},
    "hi": {"लेकिन", "पर", "परंतु", "किंतु", "मगर"},
    "hi-en": {"lekin", "par", "magar", "but", "however"},
}
CJK_CONTRAST = {"ja": ["けど", "しかし", "でも", "ただ", "が、"], "zh": ["但", "可是", "不过", "不過", "然而"]}

CUE_DESCRIPTIONS = {
    "negation": "Contains a negation word (e.g. not / nicht / nahi / 不). Negation can flip the meaning "
                "of positive words, which bag-of-words models handle poorly.",
    "contrast": "Contains a contrast word (but / aber / lekin / 但). Mixed opinions such as "
                "'good camera but poor battery' are hard to assign one label.",
    "code_mixed": "Hindi-English code-mixed text (romanised Hindi, non-standard spelling, slang).",
    "very_short": "Fewer than 6 words: little context for either model.",
    "long": "Longer than ~600 characters: the transformer sees it in several chunks and the "
            "opinion may change during the review.",
    "three_star_label": "Gold label 'neutral' derived from a 3-star rating. 3-star reviews often "
                        "contain both praise and complaints, so the label itself is ambiguous.",
}

POSSIBLE_CAUSES = {
    "Sarcasm / irony": "Positive words used with negative intent ('Great, it broke after a day'). "
                       "Neither model has an explicit notion of irony.",
    "Negation": "'not good', 'nicht schlecht', 'acha nahi hai'. TF-IDF bigrams capture only short "
                "negation scopes; the transformer models context but can still miss long-range negation.",
    "Mixed-language sentences": "Code-mixed tweets use romanised Hindi with inconsistent spelling "
                                "(nahi / nhi / nahin), which fragments the vocabulary.",
    "Slang and spelling mistakes": "Out-of-vocabulary tokens for TF-IDF; sub-word tokenisation helps "
                                   "the transformer but does not solve it.",
    "Ambiguous / mixed opinions": "Reviews that praise and criticise at the same time, especially "
                                  "3-star reviews mapped to 'neutral'.",
    "Domain-specific words": "Product vocabulary ('battery drains', 'runs small') whose polarity "
                             "depends on the domain; Yelp restaurant reviews (Italian) and "
                             "political tweets (Hinglish) differ from Amazon product reviews.",
    "Insufficient training examples": "Hindi has only ~800 training reviews and no neutral class, so "
                                      "the models have little evidence for that language.",
    "Label noise from rating mapping": "Star ratings are a proxy for sentiment; a 4-star review can "
                                       "still be mostly critical.",
}


# --------------------------------------------------------------------------
# Cue tagging
# --------------------------------------------------------------------------
def _has_any(text: str, words: set[str]) -> bool:
    tokens = {t.casefold() for t in WORD_RE.findall(text)}
    return bool(tokens & words)


def review_cues(text: str, language: str, original_label: str = "") -> list[str]:
    cues = []
    if language in CJK_NEGATION:
        if any(m in text for m in CJK_NEGATION[language]):
            cues.append("negation")
        if any(m in text for m in CJK_CONTRAST[language]):
            cues.append("contrast")
        n_words = len(text) // 2
    else:
        if _has_any(text, NEGATION_WORDS.get(language, NEGATION_WORDS["en"])):
            cues.append("negation")
        if _has_any(text, CONTRAST_WORDS.get(language, CONTRAST_WORDS["en"])):
            cues.append("contrast")
        n_words = len(WORD_RE.findall(text))
    if language == "hi-en":
        cues.append("code_mixed")
    if n_words < 6:
        cues.append("very_short")
    if len(text) > 600:
        cues.append("long")
    if re.fullmatch(r"3 stars?", str(original_label)):
        cues.append("three_star_label")
    return cues


def add_cues(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cues"] = [review_cues(t, l, o) for t, l, o in zip(df["text"], df["language"], df["original_label"])]
    df["language_name"] = df["language"].map(settings.language_name)
    return df


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------
def model_keys_in(preds: pd.DataFrame) -> list[str]:
    return [c[len("pred_"):] for c in preds.columns if c.startswith("pred_")]


def errors_by_language(preds: pd.DataFrame, key: str) -> pd.DataFrame:
    wrong = preds[f"pred_{key}"] != preds["sentiment"]
    out = preds.assign(error=wrong).groupby("language").agg(n=("error", "size"), errors=("error", "sum"))
    out["error_rate"] = out["errors"] / out["n"]
    out.index = out.index.map(settings.language_name)
    return out.sort_values("error_rate", ascending=False)


def error_types(preds: pd.DataFrame, key: str) -> pd.DataFrame:
    """Counts of (true label -> predicted label) confusions."""
    wrong = preds[preds[f"pred_{key}"] != preds["sentiment"]]
    out = wrong.groupby(["sentiment", f"pred_{key}"]).size().rename("count").reset_index()
    out.columns = ["true_label", "predicted_label", "count"]
    out["share_of_errors"] = out["count"] / max(len(wrong), 1)
    return out.sort_values("count", ascending=False)


def incorrect_predictions(preds: pd.DataFrame, key: str, predicted: str | None = None,
                          true: str | None = None, language: str | None = None) -> pd.DataFrame:
    """Misclassified reviews, most confident mistakes first.

    ``predicted="positive"`` returns *incorrect positive predictions*: reviews
    the model called positive whose gold label is negative or neutral.
    """
    df = preds[preds[f"pred_{key}"] != preds["sentiment"]]
    if predicted:
        df = df[df[f"pred_{key}"] == predicted]
    if true:
        df = df[df["sentiment"] == true]
    if language:
        df = df[df["language"] == language]
    return df.sort_values(f"conf_{key}", ascending=False)


def disagreements(preds: pd.DataFrame, key_a: str, key_b: str) -> pd.DataFrame:
    """Reviews on which two models predict different labels, and who was right."""
    df = preds[preds[f"pred_{key_a}"] != preds[f"pred_{key_b}"]].copy()
    a_ok = df[f"pred_{key_a}"] == df["sentiment"]
    b_ok = df[f"pred_{key_b}"] == df["sentiment"]
    df["correct_model"] = "neither"
    df.loc[a_ok, "correct_model"] = key_a
    df.loc[b_ok, "correct_model"] = key_b
    return df


def disagreement_summary(preds: pd.DataFrame, key_a: str, key_b: str) -> pd.DataFrame:
    d = disagreements(preds, key_a, key_b)
    table = pd.crosstab(d["language"].map(settings.language_name), d["correct_model"])
    total = preds.groupby(preds["language"].map(settings.language_name)).size()
    table["disagreements"] = table.sum(axis=1)
    table["disagreement_rate"] = table["disagreements"] / total.reindex(table.index)
    return table.sort_values("disagreement_rate", ascending=False)


def cue_statistics(preds: pd.DataFrame, key: str) -> pd.DataFrame:
    """Error rate of reviews with vs. without each cue."""
    if "cues" not in preds.columns:
        preds = add_cues(preds)
    wrong = preds[f"pred_{key}"] != preds["sentiment"]
    rows = []
    for cue in CUE_DESCRIPTIONS:
        has = preds["cues"].map(lambda c, cue=cue: cue in c)
        if has.sum() == 0:
            continue
        rows.append({"cue": cue, "reviews_with_cue": int(has.sum()),
                     "error_rate_with_cue": float(wrong[has].mean()),
                     "error_rate_without_cue": float(wrong[~has].mean()) if (~has).any() else float("nan"),
                     "description": CUE_DESCRIPTIONS[cue]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["difference"] = out["error_rate_with_cue"] - out["error_rate_without_cue"]
    return out


def main() -> None:
    from src.evaluation import load_predictions

    p = argparse.ArgumentParser(description="Error analysis on saved test predictions")
    p.add_argument("--examples-per-type", type=int, default=15)
    p.parse_args()
    preds = load_predictions()
    if preds.empty:
        raise SystemExit("No predictions found - run `python -m src.evaluation` first.")
    preds = add_cues(preds)
    keys = model_keys_in(preds)
    summary: dict = {}
    examples = []
    for key in keys:
        summary[key] = {
            "errors": int((preds[f"pred_{key}"] != preds["sentiment"]).sum()),
            "error_types": error_types(preds, key).round(4).to_dict(orient="records"),
            "errors_by_language": errors_by_language(preds, key).round(4).reset_index().to_dict(orient="records"),
            "cue_statistics": cue_statistics(preds, key).drop(columns="description").round(4).to_dict(orient="records"),
        }
        for lab in settings.LABELS:
            ex = incorrect_predictions(preds, key, predicted=lab).head(15)
            for r in ex.itertuples():
                examples.append({"model": key, "error_type": f"incorrect {lab} prediction", "language": r.language,
                                 "gold": r.sentiment, "predicted": getattr(r, f"pred_{key}"),
                                 "confidence": getattr(r, f"conf_{key}"), "cues": ";".join(r.cues),
                                 "text": r.text})
    if settings.MODEL_LR in keys and settings.MODEL_XLMR_FT in keys:
        summary["disagreement_lr_vs_xlmr"] = disagreement_summary(
            preds, settings.MODEL_LR, settings.MODEL_XLMR_FT).round(4).reset_index().to_dict(orient="records")
    out = settings.EVAL_RESULTS_DIR
    with open(out / "error_analysis_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    pd.DataFrame(examples).to_csv(out / "error_examples.csv", index=False)
    for key in keys:
        print(f"\n== {key}")
        print(cue_statistics(preds, key).drop(columns="description").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
