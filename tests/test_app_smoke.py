"""The app's pages load, demos render offline, bad input gets a friendly message."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from ui import state  # noqa: E402

TIMEOUT = 90


@pytest.fixture
def offline(monkeypatch):
    """Fail loudly if anything tries the network."""
    from src.ingest import fetcher, resolver

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(resolver, "_query_tic_by_id", boom)
    monkeypatch.setattr(resolver, "_query_tic_by_name", boom)
    monkeypatch.setattr(fetcher, "_search_products", boom)


def _texts(at) -> str:
    parts = [m.value for m in at.markdown] + [e.value for e in at.error] + \
        [s.value for s in at.subheader] + [t.value for t in at.title]
    return "\n".join(str(p) for p in parts)


def test_entry_point_and_home_load(offline):
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT).run()
    assert not at.exception
    home = AppTest.from_file("pages/1_Home.py", default_timeout=TIMEOUT).run()
    assert not home.exception
    assert "Planet Hunter" in _texts(home)


def test_examples_are_committed_and_complete():
    ex = state.load_examples()
    groups = {e["group"] for e in ex["examples"]}
    assert {"Confirmed planets", "Eclipsing binaries", "Variable stars",
            "Unconfirmed candidates", "Tricky cases"} <= groups
    for e in ex["examples"]:
        assert state.load_demo(e["tic_id"]) is not None, e["display_name"]


@pytest.mark.parametrize("tic,verdict", [(420814525, "Planet"), (300871545, "Eclipsing binary"),
                                         (441584047, "Variable star")])
def test_demo_renders_verdict_offline(offline, tic, verdict):
    at = AppTest.from_file("pages/2_Analyze_a_Star.py", default_timeout=TIMEOUT).run()
    at.selectbox(key="example_sel").set_value(tic).run()
    assert not at.exception
    text = _texts(at)
    assert verdict in text
    demo = state.load_demo(tic)
    assert demo["plain_summary"] in text
    assert "Why do we think so?" in text


def test_demo_eb_leads_with_the_physics_reason(offline):
    at = AppTest.from_file("pages/2_Analyze_a_Star.py", default_timeout=TIMEOUT).run()
    at.selectbox(key="example_sel").set_value(300871545).run()
    bullets = [m.value for m in at.markdown if str(m.value).startswith("- ")]
    assert bullets and bullets[0].startswith("- A clear second eclipse was found halfway")


def test_bad_star_name_shows_friendly_error(monkeypatch):
    from src.errors import TargetNotFoundError
    from src.ingest import resolver

    def not_found(*a, **k):
        raise TargetNotFoundError("'zzzz-not-a-star': not found")

    monkeypatch.setattr(resolver, "resolve_target", not_found)
    at = AppTest.from_file("pages/2_Analyze_a_Star.py", default_timeout=TIMEOUT).run()
    at.text_input[0].set_value("zzzz-not-a-star")
    next(b for b in at.button if "Find" in str(b.label)).click().run()
    assert not at.exception
    errs = " ".join(e.value for e in at.error)
    assert "We couldn't find that star" in errs
    assert "Traceback" not in errs                 # technical details stay collapsed
