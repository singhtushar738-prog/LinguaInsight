"""
Exploratory data analysis (EDA) of the processed multilingual dataset.

    python -m src.data_exploration      # prints statistics, saves figures to reports/figures

All numbers are computed from ``data/processed/reviews_all.csv`` and the
cleaning log ``data/processed/dataset_summary.json`` written by the loader.
"""
from __future__ import annotations

import json

import pandas as pd

from config import settings
from src.preprocessing import WORD_RE, contains_cjk, top_terms


def add_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Character and word counts. For Chinese/Japanese one character ~ one token."""
    df = df.copy()
    df["length_chars"] = df["text"].str.len()
    df["length_words"] = [len(t) if contains_cjk(t) and lang in ("ja", "zh") else len(WORD_RE.findall(t))
                          for t, lang in zip(df["text"], df["language"])]
    df["language_name"] = df["language"].map(settings.language_name)
    return df


def overview(df: pd.DataFrame) -> dict:
    counts = df["sentiment"].value_counts()
    return {
        "total_reviews": len(df),
        "languages": int(df["language"].nunique()),
        "positive": int(counts.get("positive", 0)),
        "negative": int(counts.get("negative", 0)),
        "neutral": int(counts.get("neutral", 0)),
        "sources": int(df["source"].nunique()),
        "split_sizes": df["split"].value_counts().to_dict(),
    }


def reviews_per_language(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(df["language"], df["sentiment"]).reindex(columns=settings.LABELS, fill_value=0)
    table["total"] = table.sum(axis=1)
    table["source"] = df.groupby("language")["source"].first()
    table.index = [f"{settings.language_name(c)} ({c})" for c in table.index]
    return table.sort_values("total", ascending=False)


def sentiment_share_by_language(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(df["language"].map(settings.language_name), df["sentiment"], normalize="index")
    return table.reindex(columns=settings.LABELS, fill_value=0).round(3)


def split_by_language(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(df["language"].map(settings.language_name), df["split"])
    return table.reindex(columns=["train", "val", "test"], fill_value=0)


def cleaning_report(summary: dict) -> pd.DataFrame:
    """Missing / duplicate / invalid rows removed per language (from the loader log)."""
    rows = []
    for lang, s in summary.get("per_language", {}).items():
        rows.append({
            "language": settings.language_name(lang), "source": s["source"], "raw_records": s["raw_records"],
            "missing_or_empty_removed": s["removed_missing_or_empty"],
            "invalid_label_removed": s["removed_invalid_label"],
            "duplicates_removed": s["removed_duplicates"],
            "conflicting_label_duplicates_removed": s["removed_conflicting_duplicates"],
            "clean_records": s["clean_records"], "sampled_records": s["sampled_records"],
        })
    return pd.DataFrame(rows)


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Missing/empty values per column in the processed dataset (should be zero for text/label)."""
    empty = df.apply(lambda col: col.isna() | col.astype(str).str.strip().eq(""))
    return pd.DataFrame({"missing_or_empty": empty.sum(), "share": empty.mean().round(4)})


def duplicate_check(df: pd.DataFrame) -> dict:
    key = df["text"].str.casefold().str.replace(r"\s+", " ", regex=True)
    return {"duplicate_texts_in_processed_data": int(key.duplicated().sum()),
            "duplicate_review_ids": int(df["review_id"].duplicated().sum())}


def length_statistics(df: pd.DataFrame) -> pd.DataFrame:
    df = add_length_columns(df) if "length_chars" not in df else df
    stats = df.groupby("language_name").agg(
        reviews=("text", "size"), mean_chars=("length_chars", "mean"), median_chars=("length_chars", "median"),
        max_chars=("length_chars", "max"), mean_words=("length_words", "mean"),
        median_words=("length_words", "median"))
    return stats.round(1).sort_values("reviews", ascending=False)


def top_terms_table(df: pd.DataFrame, language: str, n: int = 15) -> pd.DataFrame:
    """Most frequent terms per sentiment class (NLTK stop words removed where a list exists)."""
    cols = {}
    sub = df[df["language"] == language]
    for lab in settings.LABELS:
        terms = top_terms(sub.loc[sub["sentiment"] == lab, "text"], language, n=n)
        cols[lab] = [f"{w} ({c})" for w, c in terms] + [""] * (n - len(terms))
    return pd.DataFrame(cols)


def main() -> None:
    from src import visualization as viz
    from src.data_loader import load_processed, load_summary

    df = add_length_columns(load_processed())
    summary = load_summary()
    out = settings.FIGURES_DIR
    print(json.dumps(overview(df), indent=2))
    print(reviews_per_language(df).to_string())
    print(length_statistics(df).to_string())
    print(cleaning_report(summary).to_string(index=False))
    print(duplicate_check(df))

    counts = df["language_name"].value_counts()
    viz.bar_matplotlib(counts, "Reviews per language").savefig(out / "reviews_per_language.png", dpi=140)
    table = pd.crosstab(df["language"], df["sentiment"])
    viz.stacked_sentiment_matplotlib(table, "Sentiment distribution by language").savefig(
        out / "sentiment_by_language.png", dpi=140)
    viz.bar_matplotlib(df["sentiment"].value_counts().reindex(settings.LABELS), "Overall sentiment distribution",
                       color="#72B7B2").savefig(out / "sentiment_distribution.png", dpi=140)
    viz.histogram_matplotlib(df["length_chars"], "Review length distribution", "Characters").savefig(
        out / "review_length_distribution.png", dpi=140)

    eda = {"overview": overview(df), "duplicates": duplicate_check(df),
           "length_statistics": length_statistics(df).reset_index().to_dict(orient="records"),
           "sentiment_share_by_language": sentiment_share_by_language(df).to_dict(orient="index")}
    with open(settings.REPORTS_DIR / "eda_summary.json", "w", encoding="utf-8") as fh:
        json.dump(eda, fh, indent=2, ensure_ascii=False)
    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
