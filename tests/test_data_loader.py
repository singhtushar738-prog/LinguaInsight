"""Tests for dataset parsing and cleaning helpers."""
from collections import Counter

import pandas as pd

from src.data_loader import clean_reviews, detokenize_tweet, parse_sentimix_conll, split_dataset


def test_parse_sentimix_conll(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("meta\t1\tpositive\nbahut\tHin\nacha\tHin\n\nmeta\t2\tnegative\nbad\tEng\n", encoding="utf-8")
    tweets = parse_sentimix_conll(f)
    assert [t["uid"] for t in tweets] == ["1", "2"]
    assert tweets[0]["tokens"] == ["bahut", "acha"] and tweets[0]["tags"] == ["Hin", "Hin"]


def test_detokenize_tweet():
    assert detokenize_tweet("@ RahulGandhi ji , sahi hai !".split()) == "@RahulGandhi ji, sahi hai!"
    assert detokenize_tweet("dekho https // t co / Ab12".split()) == "dekho http"


def test_clean_reviews_removes_empty_duplicates_and_conflicts():
    df = pd.DataFrame({
        "text": ["Good product", "good  product", "", None, "Same text", "same text", "Fine"],
        "sentiment": ["positive", "positive", "negative", "neutral", "positive", "negative", "neutral"],
    })
    stats = Counter()
    out = clean_reviews(df, stats)
    assert out["text"].tolist() == ["Good product", "Fine"]
    assert stats["missing_or_empty_text"] == 2
    assert stats["duplicate_text"] == 1
    assert stats["conflicting_duplicate_labels"] == 2


def test_split_is_stratified_and_complete():
    df = pd.DataFrame({"language": ["en"] * 100 + ["hi"] * 100,
                       "sentiment": (["positive"] * 50 + ["negative"] * 50) * 2,
                       "text": [f"t{i}" for i in range(200)]})
    out = split_dataset(df, seed=0)
    sizes = out["split"].value_counts()
    assert sizes["test"] == 30 and sizes["train"] + sizes["val"] == 170 and abs(sizes["val"] - 30) <= 1
    test = out[out["split"] == "test"]
    assert test.groupby(["language", "sentiment"]).size().min() >= 7
