"""Headless smoke tests: every Streamlit page must render without exceptions."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["app.py"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=300).run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _classical_available() -> bool:
    from config import settings
    return settings.CLASSICAL_LR_PATH.exists()


@pytest.mark.skipif(not _classical_available(), reason="classical model not trained")
def test_single_review_interaction():
    from config import settings
    at = AppTest.from_file(str(ROOT / "pages/1_Single_Review_Analysis.py"), default_timeout=300).run()
    at.text_area(key="single_text").set_value("Ce produit est excellent, je le recommande !")
    at.multiselect[0].set_value([settings.MODEL_LR])
    at.button[-1].click().run()     # "Analyse sentiment"
    assert not at.exception, [e.message for e in at.exception]
    assert any(m.label == "Detected language" and m.value == "French" for m in at.metric)


@pytest.mark.skipif(not _classical_available(), reason="classical model not trained")
def test_bulk_sample_interaction():
    from config import settings
    at = AppTest.from_file(str(ROOT / "pages/2_Bulk_Review_Analysis.py"), default_timeout=600).run()
    at.radio[0].set_value("Use the sample file (real test-set reviews)").run()
    at.multiselect[0].set_value([settings.MODEL_LR])
    at.button[0].click().run()      # "Analyse reviews"
    assert not at.exception, [e.message for e in at.exception]
    assert "bulk_results" in at.session_state
    assert len(at.session_state["bulk_results"]) == 250
