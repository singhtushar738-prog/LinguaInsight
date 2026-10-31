"""
Language detection for single reviews and batches.

Three complementary signals are combined:

1. ``langdetect`` (a port of Google's language-detection library, naive Bayes
   over character n-grams, 55 languages). It returns candidate languages with
   probabilities. A fixed seed makes it deterministic.
2. A Unicode *script profile* (Latin, Devanagari, CJK, ...). Mixed scripts
   (e.g. Devanagari + Latin) are a strong sign of code-mixing.
3. A token-level *Hinglish tagger*. Romanised Hindi ("bahut accha hai") is
   written in Latin script, which langdetect does not model, so it typically
   returns a random Latin-script language. Each Latin token is looked up in
   a romanised-Hindi / English lexicon learned from the token-level language
   tags of the SemEval-2020 SentiMix corpus (see ``data_loader``).

Limitations (stated honestly in the UI):
* Ordinary language identification assigns ONE label per text. Code-mixed
  detection here is a heuristic and only covers Hindi-English.
* Very short texts (a few words) are often misidentified.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from config import settings
from src.preprocessing import WORD_RE, get_stopwords, is_valid_text

try:
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0          # deterministic results
    LANGDETECT_AVAILABLE = True
except ImportError:                   # pragma: no cover - dependency is in requirements.txt
    LANGDETECT_AVAILABLE = False

    class LangDetectException(Exception):  # type: ignore[no-redef]
        pass

SHORT_TEXT_WORDS = 3
SHORT_TEXT_LETTERS = 10
LOW_CONFIDENCE = 0.70
LATIN_EUROPEAN = {"es", "fr", "it", "pt", "de"}


@dataclass
class LanguageResult:
    code: str | None                   # e.g. "en", "hi", "hi-en"; None when detection failed
    name: str
    confidence: float                  # langdetect probability (or heuristic score)
    method: str                        # how the decision was made
    candidates: list[tuple[str, float]] = field(default_factory=list)
    scripts: dict[str, float] = field(default_factory=dict)
    is_reliable: bool = True
    is_supported: bool = True          # language present in the project training data
    is_code_mixed: bool = False
    mixed_components: list[str] = field(default_factory=list)
    token_tags: list[tuple[str, str]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "language_code": self.code, "language_name": self.name,
            "language_confidence": round(self.confidence, 4), "language_method": self.method,
            "language_reliable": self.is_reliable, "language_supported": self.is_supported,
            "code_mixed": self.is_code_mixed,
        }


# --------------------------------------------------------------------------
# Script profile
# --------------------------------------------------------------------------
def char_script(ch: str) -> str | None:
    """Coarse Unicode script of a letter (None for non-letters)."""
    if not ch.isalpha() and unicodedata.category(ch) not in ("Mn", "Mc"):
        return None
    cp = ord(ch)
    if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF:
        return "Latin"
    if 0x0900 <= cp <= 0x097F:
        return "Devanagari"
    if 0x3040 <= cp <= 0x30FF or 0xFF66 <= cp <= 0xFF9F:
        return "Kana"
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
        return "Han"
    if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
        return "Hangul"
    if 0x0600 <= cp <= 0x06FF:
        return "Arabic"
    if 0x0400 <= cp <= 0x04FF:
        return "Cyrillic"
    if 0x0980 <= cp <= 0x0DFF:
        return "Other Indic"
    if 0x0370 <= cp <= 0x03FF:
        return "Greek"
    return "Other"


def script_profile(text: str) -> dict[str, float]:
    """Fraction of letters belonging to each script (sorted, largest first)."""
    counts = Counter(s for s in map(char_script, text) if s)
    total = sum(counts.values())
    if not total:
        return {}
    return {k: round(v / total, 3) for k, v in counts.most_common()}


# --------------------------------------------------------------------------
# Hinglish lexicon
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_lexicon() -> dict[str, str]:
    """word -> "hi" / "en" (romanised Hindi or English), learned from SentiMix."""
    path = settings.PROCESSED_DATA_DIR / "hinglish_lexicon.csv"
    lexicon: dict[str, str] = {}
    if path.exists():
        df = pd.read_csv(path, keep_default_na=False)
        lexicon = dict(zip(df["word"].astype(str), df["tag"].astype(str)))
    # English function words from NLTK. (NLTK's "hinglish" list is NOT used: it is a
    # union of romanised Hindi with English and other words such as "good", "que".)
    for w in get_stopwords("en"):
        lexicon.setdefault(w, "en")
    return lexicon


@lru_cache(maxsize=1)
def _european_function_words() -> frozenset[str]:
    """Function words of es/fr/it/pt/de, e.g. "se" or "hai" (Italian).

    Romanised-Hindi words that are also common words in these languages are
    treated as *ambiguous* and count only a quarter when scoring Hinglish.
    """
    words: set[str] = set()
    for lang in LATIN_EUROPEAN:
        words |= get_stopwords(lang)
    return frozenset(words)


def tag_tokens(text: str) -> list[tuple[str, str]]:
    """Token-level language tags: hi (Devanagari), hi-rom, en, other."""
    lexicon = load_lexicon()
    tags = []
    for tok in WORD_RE.findall(text):
        scripts = {char_script(c) for c in tok} - {None}
        if "Devanagari" in scripts:
            tags.append((tok, "hi"))
        elif scripts == {"Latin"} and tok.isascii():
            tag = lexicon.get(tok.casefold())
            tags.append((tok, "hi-rom" if tag == "hi" else "en" if tag == "en" else "other"))
        elif scripts:
            tags.append((tok, "other"))
    return tags


def hinglish_score(token_tags: list[tuple[str, str]]) -> dict[str, float]:
    """Score how "romanised Hindi" the Latin tokens of a text look.

    Unambiguous Hindi words ("bahut", "accha") count 1.0; words that are also
    European function words ("se", "hai", "non") count 0.25.
    """
    ambiguous = _european_function_words()
    stats = {"score": 0.0, "hi_hits": 0, "strong_hi_hits": 0, "en_hits": 0, "latin_tokens": 0}
    for tok, tag in token_tags:
        if tag in ("hi-rom", "en", "other"):
            stats["latin_tokens"] += 1
        if tag == "hi-rom":
            stats["hi_hits"] += 1
            if tok.casefold() in ambiguous:
                stats["score"] += 0.25
            else:
                stats["score"] += 1.0
                stats["strong_hi_hits"] += 1
        elif tag == "en":
            stats["en_hits"] += 1
    return stats


# --------------------------------------------------------------------------
# Main API
# --------------------------------------------------------------------------
def _langdetect_candidates(text: str) -> list[tuple[str, float]]:
    if not LANGDETECT_AVAILABLE:
        return []
    try:
        return [(settings.LANGDETECT_TO_DATASET.get(c.lang, c.lang), float(c.prob))
                for c in detect_langs(text)]
    except LangDetectException:
        return []


def detect_language(text: object) -> LanguageResult:
    """Detect the (likely) language of one review."""
    if not is_valid_text(text):
        return LanguageResult(code=None, name="Unknown", confidence=0.0, method="none",
                              is_reliable=False, is_supported=False,
                              messages=["Language detection failed: the text is empty or has no letters."])
    text = str(text)
    scripts = script_profile(text)
    candidates = _langdetect_candidates(text)
    token_tags = tag_tokens(text)
    letters = sum(1 for c in text if char_script(c))
    # Chinese/Japanese have no spaces: approximate one word per two characters.
    cjk_letters = sum(1 for c in text if char_script(c) in ("Han", "Kana"))
    n_words = len(WORD_RE.findall(text)) + cjk_letters // 2
    messages: list[str] = []

    code, conf, method = (candidates[0][0], candidates[0][1], "langdetect") if candidates else (None, 0.0, "none")

    # --- Code-mixed Hindi-English --------------------------------------
    is_mixed, components = False, []
    hs = hinglish_score(token_tags)
    latin, hi_hits = hs["latin_tokens"], hs["hi_hits"]
    dev_share, latin_share = scripts.get("Devanagari", 0.0), scripts.get("Latin", 0.0)
    if dev_share >= 0.15 and latin_share >= 0.15:
        # Hindi and English in their own scripts within one review.
        is_mixed, components, method = True, ["hi", "en"], "script mix (Devanagari + Latin)"
        code, conf = "hi-en", min(dev_share, latin_share) * 2
    elif (latin and latin_share >= 0.8 and hs["score"] >= 1.0 and hs["strong_hi_hits"] >= 1
          and hi_hits / latin >= 0.1):
        # Thresholds were tuned on the validation split (not the test split).
        # langdetect is very reliable for confident European-language predictions; only
        # override it when the text is clearly dominated by unambiguous Hindi words.
        clearly_hindi = hs["strong_hi_hits"] >= 3 and hi_hits / latin >= 0.25
        trusted_european = code in LATIN_EUROPEAN and conf >= 0.9 and not clearly_hindi
        if not trusted_european:
            is_mixed = hs["en_hits"] > 0
            components = ["hi (romanised)", "en"] if is_mixed else ["hi (romanised)"]
            method = "Hinglish lexicon (SentiMix token tags)"
            code, conf = "hi-en", min(1.0, hi_hits / latin + 0.3)

    if code is None:
        messages.append("Language detection failed. Please select the language manually.")
        return LanguageResult(code=None, name="Unknown", confidence=0.0, method="none",
                              candidates=candidates, scripts=scripts, is_reliable=False,
                              is_supported=False, token_tags=token_tags, messages=messages)

    # --- Reliability and support ----------------------------------------
    is_reliable = True
    if n_words < SHORT_TEXT_WORDS or letters < SHORT_TEXT_LETTERS:
        is_reliable = False
        messages.append("The text is very short, so language detection may be unreliable.")
    if method == "langdetect" and conf < LOW_CONFIDENCE:
        is_reliable = False
        messages.append(f"Low detection confidence ({conf:.0%}); the text may be mixed or ambiguous.")
    is_supported = code in settings.DATASET_LANGUAGES
    if not is_supported:
        messages.append(
            f"{settings.language_name(code)} is not part of the project training data. "
            "XLM-RoBERTa (pre-trained on 100 languages) can still be used, but results are "
            "not evaluated for this language and the TF-IDF models are unlikely to work well.")
    if is_mixed:
        messages.append("Code-mixed text detected heuristically; standard language identifiers "
                        "assign only one language per text.")

    return LanguageResult(code=code, name=settings.language_name(code), confidence=float(conf),
                          method=method, candidates=candidates[:3], scripts=scripts,
                          is_reliable=is_reliable, is_supported=is_supported,
                          is_code_mixed=is_mixed, mixed_components=components,
                          token_tags=token_tags, messages=messages)


def detect_languages(texts: list[object]) -> pd.DataFrame:
    """Vectorised helper for bulk analysis (one row per input text)."""
    return pd.DataFrame([detect_language(t).as_dict() for t in texts])
