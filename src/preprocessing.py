"""
Multilingual text preprocessing.

Two *separate* pipelines are provided:

1. ``preprocess_for_classical``  - used before TF-IDF. It normalises Unicode,
   lower-cases (case-folds), replaces URLs / user mentions / numbers with
   placeholder tokens, reduces character elongation ("sooooo" -> "soo") and,
   for Chinese/Japanese (written without spaces), inserts spaces between
   characters so that word n-grams become character n-grams.

2. ``preprocess_for_transformer`` - used before XLM-RoBERTa. It only performs
   light, meaning-preserving normalisation (Unicode NFC, URL/mention
   placeholders, whitespace). Casing, punctuation, emojis and word order are
   kept because XLM-R's own SentencePiece tokenizer was trained on raw text.

Design decisions (important for the viva):
* Stop words are NOT removed for modelling. Generic stop-word lists contain
  negations ("not", "nicht", "pas", "nahi", "não") that flip sentiment.
  Language-specific stop-word lists (NLTK) are only used for exploratory
  "top words" statistics, and only for languages that have such a list.
* Reviews are NOT translated to English; the original text is always kept.
* Python's built-in ``\\w`` regex class does not match Indic vowel signs
  (Unicode category Mc/Mn), so a naive tokenizer would split the Hindi word
  "अच्छा" into fragments. ``WORD_PATTERN`` below explicitly includes
  combining marks.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable

# --------------------------------------------------------------------------
# Regular expressions
# --------------------------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\bhttp\b)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MENTION_RE = re.compile(r"@\w+")
HTML_TAG_RE = re.compile(r"<[^>]{1,100}>")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
# Three or more repetitions of the same character -> two repetitions.
ELONGATION_RE = re.compile(r"(.)\1{2,}")
WHITESPACE_RE = re.compile(r"\s+")


def _ranges(*pairs: tuple[int, int]) -> str:
    """Build a regex character-class body from (first, last) code points."""
    return "".join(f"{re.escape(chr(a))}-{re.escape(chr(b))}" for a, b in pairs)


# ASCII control characters, zero-width space (U+200B) and BOM (U+FEFF).
CONTROL_RE = re.compile("[" + _ranges((0x00, 0x08), (0x0B, 0x0C), (0x0E, 0x1F), (0x7F, 0x7F),
                                      (0x200B, 0x200B), (0xFEFF, 0xFEFF)) + "]")

# Combining marks (Unicode categories Mn/Mc) used by Latin diacritics and the
# main Indic scripts. Python's \w does not include them.
_COMBINING = _ranges(
    (0x0300, 0x036F),                                    # Latin combining diacritics
    (0x0900, 0x0903), (0x093A, 0x094F), (0x0951, 0x0957), (0x0962, 0x0963),  # Devanagari
    (0x0981, 0x0983), (0x09BC, 0x09D7),                  # Bengali
    (0x0A01, 0x0A03), (0x0A3C, 0x0A51),                  # Gurmukhi (Punjabi)
    (0x0A81, 0x0A83), (0x0ABC, 0x0ACD),                  # Gujarati
    (0x0B01, 0x0B03), (0x0B3C, 0x0B57),                  # Odia
    (0x0B82, 0x0B82), (0x0BBE, 0x0BCD),                  # Tamil
    (0x0C00, 0x0C04), (0x0C3E, 0x0C56),                  # Telugu
    (0x0C81, 0x0C83), (0x0CBC, 0x0CD6),                  # Kannada
    (0x0D00, 0x0D03), (0x0D3B, 0x0D57),                  # Malayalam
    (0x0E31, 0x0E31), (0x0E34, 0x0E3A), (0x0E47, 0x0E4E),  # Thai
)
# A "word" = letters/digits/underscore plus combining marks, optionally with
# internal apostrophes or hyphens (e.g. "don't", "e-mail", "l'acqua").
_APOSTROPHES = "'" + chr(0x2019)
WORD_PATTERN = rf"[\w{_COMBINING}]+(?:[{_APOSTROPHES}\-][\w{_COMBINING}]+)*"
WORD_RE = re.compile(WORD_PATTERN)

# Characters of scripts that are written without spaces between words.
_CJK_RANGES = _ranges(
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),   # Half-width Katakana
)
CJK_CHAR_RE = re.compile(f"([{_CJK_RANGES}])")
SPACELESS_LANGUAGES = {"zh", "ja", "zh-cn", "zh-tw", "th"}

# Placeholder tokens
URL_TOKEN, USER_TOKEN, EMAIL_TOKEN, NUM_TOKEN = "httpurl", "@user", "emailaddr", "0"


# --------------------------------------------------------------------------
# Shared basic cleaning
# --------------------------------------------------------------------------
def is_valid_text(text: object, min_chars: int = 1) -> bool:
    """True when ``text`` is a non-empty string with at least one letter/digit."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    return any(ch.isalnum() for ch in stripped)


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Unicode normalisation + removal of invisible control characters.

    NFC composes characters (e.g. "e" + combining acute -> "é") so the same
    word is always represented by the same code points. NFKC would also map
    full-width characters (common in Japanese/Chinese reviews) to ASCII.
    """
    text = unicodedata.normalize(form, text)
    text = CONTROL_RE.sub(" ", text)
    return text


def basic_clean(text: object) -> str:
    """Cleaning applied to every stored review (dataset construction).

    Keeps the review readable: only Unicode normalisation, HTML tag removal and
    whitespace collapsing. Returns "" for missing values.
    """
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text, "NFC")
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_urls_and_mentions(text: str) -> str:
    text = URL_RE.sub(f" {URL_TOKEN} ", text)
    text = EMAIL_RE.sub(f" {EMAIL_TOKEN} ", text)
    text = MENTION_RE.sub(f" {USER_TOKEN} ", text)
    return text


# --------------------------------------------------------------------------
# Language-aware helpers
# --------------------------------------------------------------------------
def segment_cjk(text: str) -> str:
    """Insert spaces around every Chinese/Japanese character.

    Chinese and Japanese do not separate words with spaces, so a whitespace or
    regex tokenizer would treat an entire sentence as one "word". Treating
    each character as a token (and their bigrams via ngram_range) is a simple,
    dictionary-free segmentation strategy commonly used for TF-IDF baselines.
    """
    return WHITESPACE_RE.sub(" ", CJK_CHAR_RE.sub(r" \1 ", text)).strip()


def contains_cjk(text: str) -> bool:
    return bool(CJK_CHAR_RE.search(text))


def tokenize(text: str, language: str | None = None) -> list[str]:
    """Language-aware tokenizer (used by TF-IDF and the explainability views).

    * Space-less scripts (Chinese/Japanese) -> one token per character.
    * Everything else -> Unicode-aware word regex that keeps Indic vowel signs
      and Latin diacritics attached to their word.
    """
    if language in SPACELESS_LANGUAGES or (language is None and contains_cjk(text)):
        text = segment_cjk(text)
    return WORD_RE.findall(text)


# --------------------------------------------------------------------------
# Pipeline 1: classical ML (TF-IDF)
# --------------------------------------------------------------------------
def preprocess_for_classical(text: object, language: str | None = None) -> str:
    """Normalise a review for the TF-IDF models.

    Steps: NFKC normalisation -> URL/e-mail/mention placeholders -> case-fold
    -> number placeholder -> elongation reduction -> CJK segmentation ->
    whitespace collapse. Punctuation is kept at this stage: the word analyser
    ignores it, while the character analyser can use "!" or "?" patterns.
    """
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text, "NFKC")      # also folds full-width forms
    text = HTML_TAG_RE.sub(" ", text)
    text = normalize_urls_and_mentions(text)
    text = text.casefold()                      # better than lower() for ß, etc.
    text = NUMBER_RE.sub(NUM_TOKEN, text)
    text = ELONGATION_RE.sub(r"\1\1", text)
    if language in SPACELESS_LANGUAGES or (language is None and contains_cjk(text)):
        text = segment_cjk(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def classical_tokenizer(text: str) -> list[str]:
    """Tokenizer passed to scikit-learn's TfidfVectorizer (text already cleaned)."""
    return WORD_RE.findall(text)


# --------------------------------------------------------------------------
# Pipeline 2: transformer (XLM-RoBERTa)
# --------------------------------------------------------------------------
def preprocess_for_transformer(text: object) -> str:
    """Minimal normalisation that preserves natural language structure.

    The text keeps its casing, punctuation, emojis and diacritics. Only URLs,
    e-mails and @mentions are replaced (they carry no sentiment and waste
    tokens), and whitespace is collapsed. Sub-word tokenization is done later
    by the model's own tokenizer.
    """
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text, "NFC")
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" http ", text)
    text = EMAIL_RE.sub(" email ", text)
    text = MENTION_RE.sub(" @user ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------
# Stop words (exploration only)
# --------------------------------------------------------------------------
NLTK_STOPWORD_LANGUAGES = {
    "en": "english", "de": "german", "es": "spanish", "fr": "french",
    "it": "italian", "pt": "portuguese", "hi": "hinglish",
}


@lru_cache(maxsize=None)
def get_stopwords(language: str) -> frozenset[str]:
    """NLTK stop-word list for a language, or an empty set when unavailable.

    Used only for exploratory word-frequency charts. If the NLTK corpus has
    not been downloaded the function degrades gracefully to an empty set.
    """
    name = NLTK_STOPWORD_LANGUAGES.get(language)
    if name is None or name == "hinglish":
        return frozenset()
    try:
        from config import settings  # noqa: F401  (sets NLTK_DATA)
        from nltk.corpus import stopwords
        return frozenset(stopwords.words(name))
    except (LookupError, OSError):
        return frozenset()


def top_terms(texts: Iterable[str], language: str | None, n: int = 20,
              remove_stopwords: bool = True) -> list[tuple[str, int]]:
    """Most frequent tokens for a collection of texts (exploration helper)."""
    from collections import Counter

    stop = get_stopwords(language) if (remove_stopwords and language) else frozenset()
    counter: Counter[str] = Counter()
    placeholders = {URL_TOKEN, USER_TOKEN, EMAIL_TOKEN, NUM_TOKEN, "user"}
    for t in texts:
        for tok in tokenize(preprocess_for_classical(t, language), language):
            if tok in stop or tok in placeholders or len(tok) < 2 and not contains_cjk(tok):
                continue
            counter[tok] += 1
    return counter.most_common(n)
