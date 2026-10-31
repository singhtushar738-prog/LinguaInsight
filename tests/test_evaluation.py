"""Tests for metric computation and dataset integrity."""
import pandas as pd
import pytest

from config import settings
from src.evaluation import compute_metrics


def test_compute_metrics_known_values():
    y_true = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
    y_pred = ["positive", "negative", "negative", "negative", "neutral", "positive"]
    m = compute_metrics(y_true, y_pred, labels=settings.LABELS)
    assert m["accuracy"] == pytest.approx(4 / 6)
    # F1: negative 0.8, neutral 2/3, positive 0.5 -> macro = 0.6556
    assert m["f1_macro"] == pytest.approx((0.8 + 2 / 3 + 0.5) / 3)
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 1, 1], [1, 0, 1]]


def test_macro_uses_present_classes_only_by_default():
    m = compute_metrics(["positive", "negative"], ["positive", "neutral"])
    assert m["labels_used_for_macro"] == ["negative", "positive"]
    assert m["f1_macro"] == pytest.approx(0.5)


def test_processed_splits_do_not_overlap():
    path = settings.PROCESSED_DATA_DIR / "reviews_all.csv"
    if not path.exists():
        pytest.skip("processed dataset not built")
    df = pd.read_csv(path, keep_default_na=False)
    ids = {s: set(df.loc[df["split"] == s, "review_id"]) for s in ("train", "val", "test")}
    assert not ids["train"] & ids["test"] and not ids["train"] & ids["val"] and not ids["val"] & ids["test"]
    texts = {s: set(df.loc[df["split"] == s, "text"].str.casefold()) for s in ("train", "test")}
    assert not texts["train"] & texts["test"], "identical review text in train and test (leakage)"
    assert set(df["sentiment"]) <= set(settings.LABELS)
