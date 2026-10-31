"""Tests for language detection, including the Hinglish heuristic."""
import pytest

from src.language_detection import detect_language, script_profile, tag_tokens


@pytest.mark.parametrize("text,code", [
    ("This product is great, I love it and would buy it again.", "en"),
    ("Ce produit est vraiment nul, je suis très déçu.", "fr"),
    ("Das Produkt ist gut, aber die Lieferung war langsam.", "de"),
    ("Llegó a tiempo y funciona perfectamente, muy recomendable.", "es"),
    ("Il servizio era lento e il cibo freddo, non tornerò.", "it"),
    ("यह फ़ोन बहुत अच्छा है लेकिन बैटरी जल्दी खत्म हो जाती है।", "hi"),
    ("この商品はとても使いやすいです", "ja"),
])
def test_monolingual_detection(text, code):
    assert detect_language(text).code == code


def test_romanised_hinglish_is_detected_as_code_mixed():
    r = detect_language("Product bahut accha hai, quality is amazing!")
    assert r.code == "hi-en"
    assert r.is_code_mixed
    tags = dict(r.token_tags)
    assert tags["bahut"] == "hi-rom" and tags["is"] == "en"


def test_script_mixed_hindi_english():
    r = detect_language("फ़ोन की quality बहुत अच्छी है but battery weak")
    assert r.code == "hi-en" and r.is_code_mixed
    assert "Devanagari" in r.scripts and "Latin" in r.scripts


def test_italian_with_ambiguous_words_is_not_hinglish():
    # "hai", "se", "non" are also romanised-Hindi words in the lexicon.
    assert detect_language("Non hai capito niente, se vuoi te lo spiego di nuovo domani.").code == "it"


def test_empty_text_fails_gracefully():
    r = detect_language("   ")
    assert r.code is None and not r.is_reliable
    assert "failed" in r.messages[0].lower()


def test_short_text_is_flagged_unreliable():
    r = detect_language("ok")
    assert not r.is_reliable
    assert any("short" in m for m in r.messages)


def test_unsupported_language_is_flagged():
    r = detect_language("Это отличный товар, рекомендую всем покупателям.")
    assert r.code == "ru"
    assert not r.is_supported


def test_script_profile_and_token_tags():
    assert script_profile("abc अब") == {"Latin": 0.6, "Devanagari": 0.4}
    assert ("अच्छा", "hi") in tag_tokens("बहुत अच्छा")
