"""
Bulk (CSV) review analysis.

The functions here are UI-independent so they can be unit-tested and reused:

* ``read_csv_safely``  - robust CSV reading (encoding / delimiter detection,
  malformed lines skipped and reported instead of crashing).
* ``guess_text_column`` / ``guess_language_column`` / ``guess_label_column``.
* ``analyze_dataframe`` - language detection + predictions from one or more
  models; rows with missing/empty text are kept and marked as skipped.
* ``summarize_results`` / ``language_sentiment_table`` / ``score_against_labels``.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import settings
from src.language_detection import detect_language
from src.preprocessing import is_valid_text

TEXT_COLUMN_HINTS = ("review", "text", "comment", "content", "body", "feedback", "message", "sentence")
LANGUAGE_COLUMN_HINTS = ("language", "lang", "language_code", "locale")
LABEL_COLUMN_HINTS = ("sentiment", "label", "rating", "stars", "polarity", "score")
# UTF-16 is only tried when the file starts with a UTF-16 byte-order mark: almost any
# byte sequence "decodes" as UTF-16, which would silently turn latin-1 files into garbage.
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


@dataclass
class CsvReadResult:
    df: pd.DataFrame
    encoding: str
    delimiter: str
    warnings: list[str] = field(default_factory=list)


def read_csv_safely(data: bytes | str, max_rows: int = settings.BULK_MAX_ROWS) -> CsvReadResult:
    """Read an uploaded CSV without crashing on unusual files."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if not raw or not raw.strip():
        raise ValueError("The uploaded file is empty.")
    warnings: list[str] = []
    text, used_encoding = None, None
    encodings = ("utf-16",) + ENCODINGS if raw.startswith(UTF16_BOMS) else ENCODINGS
    for enc in encodings:
        try:
            text = raw.decode(enc)
            used_encoding = enc
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        raise ValueError("Could not decode the file. Please save it as UTF-8 CSV.")
    if used_encoding not in ("utf-8-sig", "utf-16"):
        warnings.append(f"File decoded as {used_encoding}; non-UTF-8 characters may display incorrectly.")

    sample = text[:20000]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","

    bad_lines: list[list[str]] = []
    # index_col=False stops pandas from silently treating extra fields as an index.
    df = pd.read_csv(io.StringIO(text), sep=delimiter, engine="python", dtype=str, index_col=False,
                     keep_default_na=True, on_bad_lines=lambda line: bad_lines.append(line) or None)
    if bad_lines:
        warnings.append(f"{len(bad_lines)} malformed line(s) had the wrong number of fields and were skipped.")
    df.columns = [str(c).strip() or f"column_{i}" for i, c in enumerate(df.columns)]
    if df.columns.duplicated().any():
        df.columns = pd.io.common.dedup_names(list(df.columns), is_potential_multiindex=False)
        warnings.append("Duplicate column names were renamed.")
    if df.empty:
        raise ValueError("The CSV file has a header but no data rows.")
    if len(df) > max_rows:
        warnings.append(f"Only the first {max_rows:,} of {len(df):,} rows are analysed.")
        df = df.head(max_rows)
    return CsvReadResult(df=df.reset_index(drop=True), encoding=used_encoding, delimiter=delimiter,
                         warnings=warnings)


def _guess_column(df: pd.DataFrame, hints: tuple[str, ...]) -> str | None:
    lowered = {c: c.lower() for c in df.columns}
    for hint in hints:
        for col, low in lowered.items():
            if low == hint:
                return col
    for hint in hints:
        for col, low in lowered.items():
            if hint in low:
                return col
    return None


def guess_text_column(df: pd.DataFrame) -> str:
    """Prefer a column named like review/text; otherwise the longest text column."""
    named = _guess_column(df, TEXT_COLUMN_HINTS)
    if named is not None:
        return named
    lengths = {c: df[c].dropna().astype(str).str.len().mean() for c in df.columns}
    lengths = {c: (v if pd.notna(v) else 0) for c, v in lengths.items()}
    return max(lengths, key=lengths.get)


def guess_language_column(df: pd.DataFrame) -> str | None:
    return _guess_column(df, LANGUAGE_COLUMN_HINTS)


def guess_label_column(df: pd.DataFrame) -> str | None:
    return _guess_column(df, LABEL_COLUMN_HINTS)


# --------------------------------------------------------------------------
# Label normalisation (for optional accuracy against a ground-truth column)
# --------------------------------------------------------------------------
_LABEL_WORDS = {
    "negative": "negative", "neg": "negative", "bad": "negative",
    "neutral": "neutral", "neu": "neutral", "mixed": "neutral",
    "positive": "positive", "pos": "positive", "good": "positive",
}


def normalize_label(value: object) -> str | None:
    """Map common label formats (text labels or 1-5 star ratings) to sentiment."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if s in _LABEL_WORDS:
        return _LABEL_WORDS[s]
    m = re.fullmatch(r"(\d(?:\.0)?)\s*(stars?)?", s)
    if m:
        stars = int(float(m.group(1)))
        return settings.RATING_TO_SENTIMENT.get(stars)
    return None


def normalize_language(value: object) -> str | None:
    """Accept language codes ("hi", "pt-BR") or English names ("Hindi")."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip().lower().replace("_", "-")
    if v in settings.DATASET_LANGUAGES:
        return v
    names = {n.lower(): c for c, n in settings.DATASET_LANGUAGES.items()}
    names["hinglish"] = "hi-en"
    if v in names:
        return names[v]
    base = v.split("-")[0]
    return base if base in settings.DATASET_LANGUAGES or base in settings.LANGDETECT_NAMES else v


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def analyze_dataframe(df: pd.DataFrame, text_column: str, model_keys: list[str],
                      language_mode: str = "auto", language_column: str | None = None,
                      fixed_language: str | None = None, progress_callback=None) -> pd.DataFrame:
    """Analyse every row of ``df``.

    language_mode: "auto" (detect per row), "column" (use ``language_column``,
    detect when that cell is empty) or "fixed" (``fixed_language`` for all rows).
    Returns the original columns plus status, language and prediction columns.
    """
    from src.prediction import predict_with_model, resolve_model_key

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the uploaded file.")
    if not model_keys:
        raise ValueError("Select at least one model.")
    out = df.copy()
    texts = out[text_column]
    valid = texts.map(is_valid_text).to_numpy()
    out["analysis_status"] = np.where(valid, "analysed", "skipped: empty or invalid text")
    out["review_length_chars"] = texts.fillna("").astype(str).str.len()

    n_steps = int(valid.sum()) * (1 + len(model_keys))
    done = 0

    def report(k: int) -> None:
        nonlocal done
        done += k
        if progress_callback and n_steps:
            progress_callback(min(done / n_steps, 1.0))

    # ---- language ------------------------------------------------------
    langs: list[str | None] = [None] * len(out)
    lang_source = [""] * len(out)
    lang_conf = [np.nan] * len(out)
    for i in np.flatnonzero(valid):
        text = str(texts.iat[i])
        given = None
        if language_mode == "fixed":
            given = fixed_language
        elif language_mode == "column" and language_column:
            given = normalize_language(out[language_column].iat[i])
        if given:
            langs[i], lang_source[i] = given, "provided"
        else:
            res = detect_language(text)
            langs[i], lang_source[i], lang_conf[i] = res.code, "detected", res.confidence
        report(1)
    out["language"] = langs
    out["language_name"] = [settings.language_name(c) if c else "Unknown" for c in langs]
    out["language_source"] = lang_source
    out["language_confidence"] = lang_conf

    # ---- predictions -----------------------------------------------------
    idx = np.flatnonzero(valid)
    valid_texts = [str(texts.iat[i]) for i in idx]
    valid_langs = [langs[i] for i in idx]
    for requested in model_keys:
        key, _ = resolve_model_key(requested)
        cols = {f"{key}_sentiment": [None] * len(out), f"{key}_confidence": [np.nan] * len(out)}
        cols.update({f"{key}_p_{lab}": [np.nan] * len(out) for lab in settings.LABELS})
        if len(idx):
            proba = predict_with_model(key, valid_texts, valid_langs,
                                       progress_callback=lambda d, n: None)
            for j, i in enumerate(idx):
                k = int(proba[j].argmax())
                cols[f"{key}_sentiment"][i] = settings.LABELS[k]
                cols[f"{key}_confidence"][i] = round(float(proba[j, k]), 4)
                for c, lab in enumerate(settings.LABELS):
                    cols[f"{key}_p_{lab}"][i] = round(float(proba[j, c]), 4)
        for c, v in cols.items():
            out[c] = v
        report(len(idx))
    return out


def prediction_columns(results: pd.DataFrame) -> list[str]:
    """Model keys present in an analysed results table."""
    return [c[: -len("_sentiment")] for c in results.columns if c.endswith("_sentiment")]


def summarize_results(results: pd.DataFrame, model_key: str) -> dict:
    col = f"{model_key}_sentiment"
    analysed = results[results["analysis_status"] == "analysed"]
    counts = analysed[col].value_counts()
    return {
        "total_rows": len(results),
        "analysed": len(analysed),
        "skipped": int((results["analysis_status"] != "analysed").sum()),
        "positive": int(counts.get("positive", 0)),
        "negative": int(counts.get("negative", 0)),
        "neutral": int(counts.get("neutral", 0)),
        "languages": int(analysed["language"].nunique()),
        "avg_length_chars": float(analysed["review_length_chars"].mean()) if len(analysed) else 0.0,
        "avg_confidence": float(analysed[f"{model_key}_confidence"].mean()) if len(analysed) else 0.0,
    }


def language_sentiment_table(results: pd.DataFrame, model_key: str) -> pd.DataFrame:
    analysed = results[results["analysis_status"] == "analysed"]
    table = pd.crosstab(analysed["language_name"], analysed[f"{model_key}_sentiment"])
    for lab in settings.LABELS:
        if lab not in table.columns:
            table[lab] = 0
    table = table[settings.LABELS]
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)


def score_against_labels(results: pd.DataFrame, label_column: str, model_key: str) -> dict | None:
    """Accuracy / macro-F1 when the upload contains a recognisable label column."""
    from sklearn.metrics import accuracy_score, f1_score

    gold = results[label_column].map(normalize_label)
    pred = results[f"{model_key}_sentiment"]
    mask = gold.notna() & pred.notna()
    if mask.sum() == 0:
        return None
    return {"n_scored": int(mask.sum()),
            "accuracy": float(accuracy_score(gold[mask], pred[mask])),
            "macro_f1": float(f1_score(gold[mask], pred[mask], average="macro"))}
