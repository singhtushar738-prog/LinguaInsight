"""Tests for the multilingual preprocessing pipelines."""
from src.preprocessing import (basic_clean, is_valid_text, preprocess_for_classical,
                               preprocess_for_transformer, segment_cjk, tokenize)


def test_hindi_tokens_keep_vowel_signs():
    # Python's plain \w would split these words at the vowel signs (matras).
    tokens = tokenize("यह फ़ोन बहुत अच्छा है", "hi")
    assert tokens == ["यह", "फ़ोन", "बहुत", "अच्छा", "है"]


def test_latin_diacritics_and_apostrophes():
    assert tokenize("très déçu, l'appareil ne marche pas", "fr") == ["très", "déçu", "l'appareil", "ne", "marche", "pas"]


def test_cjk_is_segmented_into_characters():
    assert segment_cjk("很好用") == "很 好 用"
    assert preprocess_for_classical("この商品は良い", "ja") == "こ の 商 品 は 良 い"


def test_classical_normalisation():
    out = preprocess_for_classical("Sooooo GOOD!!! see https://x.com/a @bob for 1,299", "en")
    assert out == "soo good!! see httpurl @user for 0"


def test_negations_are_not_removed():
    # Stop-word removal would delete "not" and flip the meaning.
    assert "not" in tokenize(preprocess_for_classical("This is not good", "en"))
    assert "nahi" in tokenize(preprocess_for_classical("accha nahi hai", "hi-en"))


def test_fullwidth_characters_are_folded():
    assert preprocess_for_classical("ＧＯＯＤ！", "en") == "good!"


def test_transformer_preprocessing_preserves_structure():
    out = preprocess_for_transformer("  Great   phone!! 😀 see https://a.b/c @x ")
    assert out == "Great phone!! 😀 see http @user"


def test_basic_clean_handles_missing_and_html():
    assert basic_clean(None) == ""
    assert basic_clean(float("nan")) == ""
    assert basic_clean("Good<br/>product &amp; fast") == "Good product & fast"


def test_is_valid_text():
    assert not is_valid_text("")
    assert not is_valid_text("   ")
    assert not is_valid_text("!!! ???")
    assert not is_valid_text(None)
    assert is_valid_text("ok")
    assert is_valid_text("好")
