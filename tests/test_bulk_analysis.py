"""Tests for robust CSV reading and bulk analysis."""
import numpy as np
import pandas as pd
import pytest

from config import settings
from src import bulk_analysis as ba
from src import prediction


def test_reads_semicolon_and_latin1():
    data = "id;comentario;nota\n1;Muy bueno, me encantó;5\n2;Pésimo;1\n".encode("latin-1")
    res = ba.read_csv_safely(data)
    assert res.delimiter == ";" and list(res.df.columns) == ["id", "comentario", "nota"]
    assert res.df.loc[1, "comentario"] == "Pésimo"


def test_malformed_lines_are_skipped_with_warning():
    data = b"text,rating\ngood,5\nbad,1,extra,fields\nok,3\n"
    res = ba.read_csv_safely(data)
    assert len(res.df) == 2
    assert any("malformed" in w for w in res.warnings)


def test_utf16_with_bom():
    res = ba.read_csv_safely("text\nबहुत अच्छा\n".encode("utf-16"))
    assert res.encoding == "utf-16" and res.df.loc[0, "text"] == "बहुत अच्छा"


def test_empty_file_raises():
    with pytest.raises(ValueError):
        ba.read_csv_safely(b"")
    with pytest.raises(ValueError):
        ba.read_csv_safely(b"text,label\n")


def test_column_guessing():
    df = pd.DataFrame({"id": ["1"], "customer_review": ["long text here"], "lang": ["en"], "stars": ["4"]})
    assert ba.guess_text_column(df) == "customer_review"
    assert ba.guess_language_column(df) == "lang"
    assert ba.guess_label_column(df) == "stars"


@pytest.mark.parametrize("value,expected", [("Positive", "positive"), ("neg", "negative"), ("3", "neutral"),
                                            ("5 stars", "positive"), ("2.0", "negative"), ("??", None), (None, None)])
def test_normalize_label(value, expected):
    assert ba.normalize_label(value) == expected


class _FakeModel:
    def predict_proba(self, texts, languages=None, progress_callback=None):
        return np.array([[0.8, 0.1, 0.1] if "bad" in t else [0.1, 0.1, 0.8] for t in texts])


def test_analyze_dataframe_handles_invalid_rows(monkeypatch):
    monkeypatch.setattr(prediction, "load_model", lambda key: _FakeModel())
    df = pd.DataFrame({"review": ["very good product", None, "   ", "bad bad quality"],
                       "unexpected column": [1, 2, 3, 4], "label": ["positive", "neutral", "neutral", "negative"]})
    out = ba.analyze_dataframe(df, "review", [settings.MODEL_LR])
    assert list(out["analysis_status"]) == ["analysed", "skipped: empty or invalid text",
                                            "skipped: empty or invalid text", "analysed"]
    assert out["unexpected column"].tolist() == [1, 2, 3, 4]
    preds = out[f"{settings.MODEL_LR}_sentiment"]
    assert preds[0] == "positive" and preds[3] == "negative" and preds[1:3].isna().all()
    s = ba.summarize_results(out, settings.MODEL_LR)
    assert (s["analysed"], s["skipped"], s["positive"], s["negative"]) == (2, 2, 1, 1)
    score = ba.score_against_labels(out, "label", settings.MODEL_LR)
    assert score["n_scored"] == 2 and score["accuracy"] == 1.0


def test_fixed_language_mode(monkeypatch):
    monkeypatch.setattr(prediction, "load_model", lambda key: _FakeModel())
    df = pd.DataFrame({"text": ["bom produto", "ruim"]})
    out = ba.analyze_dataframe(df, "text", [settings.MODEL_LR], language_mode="fixed", fixed_language="pt")
    assert out["language"].tolist() == ["pt", "pt"] and set(out["language_source"]) == {"provided"}


def test_missing_text_column_raises():
    with pytest.raises(ValueError):
        ba.analyze_dataframe(pd.DataFrame({"a": ["x"]}), "review", [settings.MODEL_LR])
