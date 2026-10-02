"""Headless smoke test of the Streamlit app against the cached prices."""

from pathlib import Path

import pytest

from argento_lite.config import CACHE_DIR, PROJECT_ROOT
from argento_lite.data.prices import META_FILE, PRICES_FILE

APP = PROJECT_ROOT / "app" / "streamlit_app.py"

pytestmark = pytest.mark.skipif(
    not (Path(CACHE_DIR) / PRICES_FILE).exists() or not (Path(CACHE_DIR) / META_FILE).exists(),
    reason="price cache missing; run `make fetch`",
)


def test_app_renders_all_tabs_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == [
        "Overview", "Frontier", "Risk ladder", "Concentration cap (my addition)"]
    assert any("not investment advice" in c.value for c in at.caption)


def test_overview_shows_live_headline():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Thinking out loud" in text
    assert "What I found" in text
    assert "**" not in text
    labels = [m.label for m in at.metric]
    assert "Risk levels that change" in labels and "Biggest change in return" in labels
    # Dollar amounts in the markdown must stay escaped or Streamlit renders them as math.
    assert r"\$50,000" in text


def _ladder_table(at):
    return next(df.value for df in at.dataframe if "Composition" in df.value.columns)


def test_changing_amount_updates_ladder():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at_50k = _ladder_table(at).copy()
    assert len(at_50k) == 10
    at.sidebar.selectbox[0].select(10000).run()
    assert not at.exception, at.exception
    assert not _ladder_table(at).equals(at_50k)


def test_look_inside_tab_renders_and_cap_slider_works():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    slider = at.sidebar.slider[0]
    assert (slider.min, slider.max) == (0.05, 0.15) and slider.value == 0.10
    comparison = next(df.value for df in at.dataframe if "Concentration-aware" in df.value.columns)
    assert len(comparison) == 10
    at.sidebar.slider[0].set_value(0.05).run()
    assert not at.exception, at.exception
