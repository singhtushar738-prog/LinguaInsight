"""
Dataset collection and preparation.

Run from the project root:

    python -m src.data_loader            # download (once) + build processed dataset

All sources are genuine, publicly available corpora. Files are downloaded once
into ``data/raw/<source>/`` and the combined, cleaned, de-duplicated and split
dataset is written to ``data/processed/``.

Sources
-------
* MARC - Multilingual Amazon Reviews Corpus (Keung et al., 2020), en/de/es/fr/ja/zh,
  1-5 star ratings. Hugging Face mirror ``mteb/amazon_reviews_multi``.
* B2W-Reviews01 - Brazilian Portuguese e-commerce reviews (Real et al., 2019),
  1-5 star ratings. GitHub ``americanas-tech/b2w-reviews01``.
* IndicSentiment (AI4Bharat, IndicXTREME 2023) - Hindi product/service reviews
  with positive / negative / neutral labels (human translated from English).
* Italian Yelp reviews - ``yeniguno/italian-yelp-reviews-dataset``; Yelp
  reviews machine-translated into Italian with 0/1/2 sentiment labels.
* SentiMix Hinglish (SemEval-2020 Task 9, Patwa et al., 2020) - code-mixed
  Hindi-English tweets with sentence sentiment and token language tags.
  Mirror ``RTT1/SentiMix``. These are social-media posts, not reviews.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from config import settings
from src.preprocessing import basic_clean, is_valid_text

logger = logging.getLogger(__name__)

HF = "https://huggingface.co/datasets"

# Source registry: where each raw file comes from and where it is stored.
SOURCES = {
    "marc": {
        "name": "MARC - Multilingual Amazon Reviews Corpus",
        "url": f"{HF}/mteb/amazon_reviews_multi",
        "license": "Amazon MARC licence (non-commercial research use)",
        "files": {f"{lang}_{split}.jsonl": f"{HF}/mteb/amazon_reviews_multi/resolve/main/{lang}/{split}.jsonl"
                  for lang in ("en", "de", "es", "fr", "ja", "zh") for split in ("validation", "test")},
    },
    "b2w": {
        "name": "B2W-Reviews01 (Brazilian Portuguese e-commerce reviews)",
        "url": "https://github.com/americanas-tech/b2w-reviews01",
        "license": "CC BY-NC-SA 4.0",
        "files": {"B2W-Reviews01.csv": "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/"
                                       "4639429ec698d7821fc99a0bc665fa213d9fcd5a/B2W-Reviews01.csv"},
    },
    "indicsentiment": {
        "name": "AI4Bharat IndicSentiment (Hindi)",
        "url": f"{HF}/ai4bharat/IndicSentiment",
        "license": "CC0 1.0",
        "files": {f"hi_{split}.json": f"{HF}/ai4bharat/IndicSentiment/resolve/main/data/{split}/hi.json"
                  for split in ("validation", "test")},
    },
    "yelp_it": {
        "name": "Italian Yelp reviews (machine-translated)",
        "url": f"{HF}/yeniguno/italian-yelp-reviews-dataset",
        "license": "Yelp Open Dataset terms (derived)",
        "files": {f"{split}.parquet": f"{HF}/yeniguno/italian-yelp-reviews-dataset/resolve/main/data/"
                                      f"{split}-00000-of-00001.parquet"
                  for split in ("train", "validation", "test")},
    },
    "sentimix": {
        "name": "SentiMix Hinglish - SemEval-2020 Task 9",
        "url": f"{HF}/RTT1/SentiMix",
        "license": "Research use (SemEval-2020 shared task data)",
        "files": {"train_14k_split_conll.txt": f"{HF}/RTT1/SentiMix/resolve/main/train_14k_split_conll.txt",
                  "dev_3k_split_conll.txt": f"{HF}/RTT1/SentiMix/resolve/main/dev_3k_split_conll.txt"},
    },
}

LANGUAGE_SOURCE = {"en": "marc", "de": "marc", "es": "marc", "fr": "marc", "ja": "marc", "zh": "marc",
                   "pt": "b2w", "hi": "indicsentiment", "it": "yelp_it", "hi-en": "sentimix"}

OUTPUT_COLUMNS = ["review_id", "language", "text", "sentiment", "original_label",
                  "label_type", "source", "source_split", "split"]


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download_file(url: str, dest: Path, timeout: int = 120) -> Path:
    """Download ``url`` to ``dest`` once (skipped if the file already exists)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return dest


def download_all() -> None:
    for key, spec in SOURCES.items():
        for fname, url in spec["files"].items():
            download_file(url, settings.RAW_DATA_DIR / key / fname)


def raw_path(source: str, fname: str) -> Path:
    return settings.RAW_DATA_DIR / source / fname


# --------------------------------------------------------------------------
# Source parsers -> common schema
# --------------------------------------------------------------------------
def _frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def load_marc(language: str) -> pd.DataFrame:
    """MARC review = title + body; label 0..4 corresponds to 1..5 stars."""
    records = []
    for split in ("validation", "test"):
        df = pd.read_json(raw_path("marc", f"{language}_{split}.jsonl"), lines=True)
        for row in df.itertuples(index=False):
            stars = int(row.label) + 1
            records.append({"review_id": row.id, "language": language, "text": row.text,
                            "original_label": f"{stars} stars", "rating": stars,
                            "label_type": "star rating (mapped)", "source_split": split})
    out = _frame(records)
    out["sentiment"] = out["rating"].map(settings.RATING_TO_SENTIMENT)
    out["source"] = "marc"
    return out.drop(columns="rating")


def load_b2w() -> pd.DataFrame:
    """B2W review = review_title + review_text; overall_rating is 1..5 stars."""
    df = pd.read_csv(raw_path("b2w", "B2W-Reviews01.csv"), low_memory=False,
                     usecols=["review_title", "review_text", "overall_rating"])
    df = df.dropna(subset=["overall_rating"])
    df["overall_rating"] = pd.to_numeric(df["overall_rating"], errors="coerce")
    df = df[df["overall_rating"].isin([1, 2, 3, 4, 5])]
    title = df["review_title"].fillna("").astype(str).str.strip()
    body = df["review_text"].fillna("").astype(str).str.strip()
    title_in_body = [b.startswith(t) for t, b in zip(title, body)]
    text = np.where(title.eq("") | np.array(title_in_body), body, title + ". " + body)
    stars = df["overall_rating"].astype(int)
    return pd.DataFrame({
        "review_id": [f"b2w_{i}" for i in df.index], "language": "pt", "text": text,
        "sentiment": stars.map(settings.RATING_TO_SENTIMENT).values,
        "original_label": (stars.astype(str) + " stars").values,
        "label_type": "star rating (mapped)", "source": "b2w", "source_split": "full",
    })


def load_indicsentiment() -> pd.DataFrame:
    """Hindi reviews ("INDIC REVIEW") with native Positive/Negative/Neutral labels."""
    frames = []
    for split in ("validation", "test"):
        df = pd.read_json(raw_path("indicsentiment", f"hi_{split}.json"), lines=True)
        df = df.reset_index(drop=True)
        frames.append(pd.DataFrame({
            "review_id": [f"indic_hi_{split}_{i}" for i in df.index], "language": "hi",
            "text": df["INDIC REVIEW"], "original_label": df["LABEL"],
            "sentiment": df["LABEL"].astype(str).str.strip().str.lower(),
            "label_type": "native sentiment label", "source": "indicsentiment", "source_split": split,
        }))
    return pd.concat(frames, ignore_index=True)


def load_yelp_it() -> pd.DataFrame:
    """Italian (machine-translated) Yelp reviews; label 0=neg, 1=neutral, 2=pos."""
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    frames = []
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(raw_path("yelp_it", f"{split}.parquet"))
        frames.append(pd.DataFrame({
            "review_id": [f"yelp_it_{split}_{i}" for i in df["id"]], "language": "it",
            "text": df["translated_text"], "original_label": df["label"].astype(str),
            "sentiment": df["label"].map(label_map),
            "label_type": "native sentiment label", "source": "yelp_it", "source_split": split,
        }))
    return pd.concat(frames, ignore_index=True)


def parse_sentimix_conll(path: Path) -> list[dict]:
    """Parse the SentiMix CoNLL format.

    Each tweet starts with ``meta<TAB>uid<TAB>sentiment`` followed by one
    ``token<TAB>lang_tag`` line per token (tags: Eng, Hin, O) and a blank line.
    """
    tweets, current = [], None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "meta" and len(parts) >= 2:
                if current:
                    tweets.append(current)
                current = {"uid": parts[1], "sentiment": parts[2] if len(parts) > 2 else None,
                           "tokens": [], "tags": []}
            elif current is not None and len(parts) == 2 and parts[0]:
                current["tokens"].append(parts[0])
                current["tags"].append(parts[1])
    if current:
        tweets.append(current)
    return tweets


def detokenize_tweet(tokens: list[str]) -> str:
    """Re-assemble a tokenised tweet into readable text."""
    import re
    text = " ".join(tokens)
    text = re.sub(r"https?\s*//\s*(?:t\s*\.?\s*co|\S+)(?:\s*/\s*\w*)?", "http", text)  # t.co links
    text = re.sub(r"([@#])\s+(?=\w)", r"\1", text)                     # "@ user" -> "@user"
    text = re.sub(r"(\w)\s+_\s+(\w)", r"\1_\2", text)                  # user _ name
    text = re.sub(r"\s+([,.!?;:…%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text


def load_sentimix(path_name: str = "train_14k_split_conll.txt") -> pd.DataFrame:
    tweets = parse_sentimix_conll(raw_path("sentimix", path_name))
    return pd.DataFrame({
        "review_id": [f"sentimix_{t['uid']}" for t in tweets], "language": "hi-en",
        "text": [detokenize_tweet(t["tokens"]) for t in tweets],
        "original_label": [t["sentiment"] for t in tweets],
        "sentiment": [str(t["sentiment"]).lower() for t in tweets],
        "label_type": "native sentiment label", "source": "sentimix",
        "source_split": path_name.split("_")[0],
    })


LOADERS = {"marc": load_marc, "b2w": load_b2w, "indicsentiment": load_indicsentiment,
           "yelp_it": load_yelp_it, "sentimix": load_sentimix}


def load_language(language: str) -> pd.DataFrame:
    source = LANGUAGE_SOURCE[language]
    return LOADERS[source](language) if source == "marc" else LOADERS[source]()


# --------------------------------------------------------------------------
# Cleaning, de-duplication and sampling
# --------------------------------------------------------------------------
def clean_reviews(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Normalise text, drop missing/empty/invalid rows and duplicates.

    ``stats`` is updated in place with the number of removed rows per reason.
    """
    n0 = len(df)
    df = df.copy()
    df["text"] = df["text"].map(basic_clean)
    valid_text = df["text"].map(lambda t: is_valid_text(t, settings.MIN_REVIEW_CHARS))
    valid_label = df["sentiment"].isin(settings.LABELS)
    stats["missing_or_empty_text"] += int((~valid_text).sum())
    stats["invalid_label"] += int((valid_text & ~valid_label).sum())
    df = df[valid_text & valid_label]

    # Exact duplicates after case/whitespace normalisation. If the same text
    # appears with *different* labels, all copies are dropped (label noise).
    key = df["text"].str.casefold().str.replace(r"\s+", " ", regex=True)
    labels_per_key = df.groupby(key)["sentiment"].transform("nunique")
    conflicting = labels_per_key > 1
    stats["conflicting_duplicate_labels"] += int(conflicting.sum())
    df, key = df[~conflicting], key[~conflicting]
    dup = key.duplicated(keep="first")
    stats["duplicate_text"] += int(dup.sum())
    df = df[~dup]
    stats["kept"] += len(df)
    stats["raw"] += n0
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample ``n`` rows keeping the original label proportions."""
    if len(df) <= n:
        return df
    sample, _ = train_test_split(df, train_size=n, stratify=df["sentiment"], random_state=seed)
    return sample


def split_dataset(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """70/15/15 split stratified by language *and* sentiment."""
    strata = df["language"] + "|" + df["sentiment"]
    test_size = settings.TEST_FRACTION
    val_size = settings.VAL_FRACTION / (1 - test_size)
    train_val, test = train_test_split(df, test_size=test_size, stratify=strata, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size,
                                  stratify=strata.loc[train_val.index], random_state=seed)
    df = df.copy()
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


# --------------------------------------------------------------------------
# Hinglish lexicon (used by the code-mixed language detector)
# --------------------------------------------------------------------------
def build_hinglish_lexicon(exclude_uids: set[str], min_count: int = 8,
                           min_ratio: float = 0.9) -> pd.DataFrame:
    """Data-driven romanised-Hindi / English word lists from SentiMix token tags.

    Only tweets that are NOT part of the project dataset are used, so the
    lexicon never "sees" evaluation tweets. A word is assigned the tag ``hi``
    (or ``en``) when it occurs at least ``min_count`` times and carries the
    ``Hin`` (or ``Eng``) annotation in >= ``min_ratio`` of its occurrences.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for t in parse_sentimix_conll(raw_path("sentimix", "train_14k_split_conll.txt")):
        if f"sentimix_{t['uid']}" in exclude_uids:
            continue
        for tok, tag in zip(t["tokens"], t["tags"]):
            w = tok.casefold()
            if w.isalpha() and w.isascii() and len(w) >= 2:
                counts[w][tag] += 1
    rows = []
    for w, c in counts.items():
        total = c["Hin"] + c["Eng"]
        if total < min_count:
            continue
        hindi_ratio = c["Hin"] / total
        tag = "hi" if hindi_ratio >= min_ratio else "en" if (1 - hindi_ratio) >= min_ratio else None
        if tag:
            rows.append({"word": w, "tag": tag, "hin_count": c["Hin"], "eng_count": c["Eng"],
                         "hindi_ratio": round(hindi_ratio, 3)})
    return pd.DataFrame(rows).sort_values(["tag", "hin_count"], ascending=[False, False])


def evaluate_hinglish_lexicon(lexicon: pd.DataFrame) -> dict:
    """Token-level accuracy of the lexicon on the held-out SentiMix dev file."""
    lookup = dict(zip(lexicon["word"], lexicon["tag"]))
    gold_map = {"Hin": "hi", "Eng": "en"}
    covered = correct = total = 0
    for t in parse_sentimix_conll(raw_path("sentimix", "dev_3k_split_conll.txt")):
        for tok, tag in zip(t["tokens"], t["tags"]):
            w = tok.casefold()
            if tag not in gold_map or not (w.isalpha() and w.isascii()):
                continue
            total += 1
            if w in lookup:
                covered += 1
                correct += lookup[w] == gold_map[tag]
    return {"tokens_evaluated": total, "coverage": round(covered / total, 4) if total else 0.0,
            "accuracy_on_covered_tokens": round(correct / covered, 4) if covered else 0.0}


def download_nltk_resources() -> None:
    """Stop-word lists (used for exploration and lexicon disambiguation only)."""
    try:
        import nltk
        nltk.download("stopwords", download_dir=str(settings.NLTK_DATA_DIR), quiet=True)
    except Exception as exc:  # network problems must not stop dataset creation
        logger.warning("Could not download NLTK stopwords: %s", exc)


# --------------------------------------------------------------------------
# Build pipeline
# --------------------------------------------------------------------------
def build_dataset(samples_per_language: int = settings.SAMPLES_PER_LANGUAGE,
                  seed: int = settings.RANDOM_SEED) -> pd.DataFrame:
    download_all()
    download_nltk_resources()
    frames, per_language_stats = [], {}
    for lang in settings.DATASET_LANGUAGES:
        stats: Counter = Counter()
        raw = load_language(lang)
        raw_label_counts = raw["sentiment"].value_counts().to_dict()
        clean = clean_reviews(raw, stats)
        sampled = stratified_sample(clean, samples_per_language, seed)
        per_language_stats[lang] = {
            "source": LANGUAGE_SOURCE[lang],
            "raw_records": int(stats["raw"]),
            "raw_label_counts": {k: int(v) for k, v in raw_label_counts.items()},
            "removed_missing_or_empty": int(stats["missing_or_empty_text"]),
            "removed_invalid_label": int(stats["invalid_label"]),
            "removed_conflicting_duplicates": int(stats["conflicting_duplicate_labels"]),
            "removed_duplicates": int(stats["duplicate_text"]),
            "clean_records": int(stats["kept"]),
            "sampled_records": len(sampled),
        }
        logger.info("%-6s raw=%d clean=%d sampled=%d", lang, stats["raw"], stats["kept"], len(sampled))
        frames.append(sampled)

    df = pd.concat(frames, ignore_index=True)
    # Cross-language duplicates (e.g. the same English sentence in two corpora).
    key = df["text"].str.casefold()
    cross_dup = key.duplicated(keep="first")
    df = df[~cross_dup].reset_index(drop=True)

    df = split_dataset(df, seed)
    df = df[OUTPUT_COLUMNS].sort_values(["split", "language", "review_id"]).reset_index(drop=True)

    out = settings.PROCESSED_DATA_DIR
    df.to_csv(out / "reviews_all.csv", index=False)
    for split in ("train", "val", "test"):
        df[df["split"] == split].to_csv(out / f"{split}.csv", index=False)

    write_sample_csv(df, seed=seed)
    lexicon = build_hinglish_lexicon(set(df.loc[df["language"] == "hi-en", "review_id"]))
    lexicon.to_csv(out / "hinglish_lexicon.csv", index=False)
    lexicon_eval = evaluate_hinglish_lexicon(lexicon)

    summary = {
        "total_records": len(df),
        "random_seed": seed,
        "samples_per_language_cap": samples_per_language,
        "removed_cross_language_duplicates": int(cross_dup.sum()),
        "split_sizes": df["split"].value_counts().to_dict(),
        "label_counts": df["sentiment"].value_counts().to_dict(),
        "rating_to_sentiment_mapping": {str(k): v for k, v in settings.RATING_TO_SENTIMENT.items()},
        "per_language": per_language_stats,
        "sources": {k: {kk: vv for kk, vv in v.items() if kk != "files"} for k, v in SOURCES.items()},
        "hinglish_lexicon_size": lexicon["tag"].value_counts().to_dict(),
        "hinglish_lexicon_eval_sentimix_dev": lexicon_eval,
    }
    with open(out / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d records to %s", len(df), out)
    return df


def write_sample_csv(df: pd.DataFrame, per_language: int = 25, seed: int = settings.RANDOM_SEED) -> Path:
    """Write ``assets/sample_reviews.csv``: real test-split reviews for the bulk-analysis demo."""
    test = df[df["split"] == "test"]
    sample = test.groupby("language").sample(n=per_language, random_state=seed)
    sample = sample.rename(columns={"text": "review_text"})[["review_id", "review_text", "language", "sentiment",
                                                              "original_label", "source"]]
    path = settings.PROJECT_ROOT / "assets" / "sample_reviews.csv"
    path.parent.mkdir(exist_ok=True)
    sample.sample(frac=1, random_state=seed).to_csv(path, index=False)
    return path


def load_processed(split: str | None = None) -> pd.DataFrame:
    """Load the processed dataset (``split`` = train / val / test / None for all)."""
    name = f"{split}.csv" if split else "reviews_all.csv"
    path = settings.PROCESSED_DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.data_loader` first.")
    return pd.read_csv(path, dtype={"original_label": str}, keep_default_na=False)


def load_summary() -> dict:
    path = settings.PROCESSED_DATA_DIR / "dataset_summary.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare the multilingual review dataset")
    parser.add_argument("--samples-per-language", type=int, default=settings.SAMPLES_PER_LANGUAGE)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build_dataset(args.samples_per_language, args.seed)
    print(df.groupby(["language", "sentiment"]).size().unstack(fill_value=0))
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
