"""The single-page app loads, and every example star renders all four tabs offline."""

import html
import json

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from config import CONFIG  # noqa: E402

TIMEOUT = 120
EXAMPLES = json.loads((CONFIG.paths.examples / "examples.json").read_text(encoding="utf-8"))
TICS = [e["tic_id"] for e in EXAMPLES["examples"]]
TABS = ["📈 Light Curves", "🔍 BLS Detection", "🌀 Phase-Folded", "🤖 Results"]


@pytest.fixture
def offline(monkeypatch):
    """Fail loudly if anything tries the network or the live pipeline."""
    from src import pipeline
    from src.ingest import fetcher, resolver

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(resolver, "_query_tic_by_id", boom)
    monkeypatch.setattr(resolver, "_query_tic_by_name", boom)
    monkeypatch.setattr(fetcher, "_search_products", boom)
    monkeypatch.setattr(pipeline, "analyze_target", boom)


def _texts(at) -> str:
    parts = ([m.value for m in at.markdown] + [s.value for s in at.subheader]
             + [c.value for c in at.caption] + [i.value for i in at.info])
    return "\n".join(str(p) for p in parts)


def _analyze(tic: int) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT).run()
    at.selectbox(key="star").set_value(tic).run()
    at.button(key="analyze").click().run()
    return at


def test_app_loads_with_the_six_stars_and_a_prompt(offline):
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT).run()
    assert not at.exception
    assert "EXOPLANET DETECTION &amp; FALSE POSITIVE RECYCLER" in _texts(at)
    assert at.selectbox(key="star").options and len(at.selectbox(key="star").options) == 6
    assert at.checkbox(key="live").value is False
    assert any("Pick a star in the sidebar" in i.value for i in at.info)
    assert not at.tabs


def test_examples_have_the_plot_views_and_stay_small():
    demo = CONFIG.paths.examples / "demo"
    assert len(TICS) == 6
    assert sum(p.stat().st_size for p in demo.glob("*.json")) < 10e6
    for tic in TICS:
        rep = json.loads((demo / f"{tic}.json").read_text(encoding="utf-8"))
        pdata = rep["plot_data"]
        for key in ("lc_norm", "lc_flat", "periodogram", "ls", "views"):
            assert pdata.get(key), (tic, key)
        assert pdata["views"]["final"]["bin_flux"], tic
        assert len(pdata["views"]["top_peaks"]) == 3, tic


@pytest.mark.parametrize("tic", TICS)
def test_each_star_renders_all_four_tabs_offline(offline, tic):
    at = _analyze(tic)
    assert not at.exception
    assert [t.label for t in at.tabs] == TABS
    for tab in at.tabs:
        assert tab.get("plotly_chart"), f"{tab.label} has no chart"
    rep = json.loads((CONFIG.paths.examples / "demo" / f"{tic}.json").read_text(encoding="utf-8"))
    assert html.escape(rep["plain_summary"]) in _texts(at)          # shown in the verdict card


@pytest.mark.parametrize("tic", TICS)
def test_results_tab_shows_the_confusion_matrix(offline, tic):
    at = _analyze(tic)
    results = at.tabs[3]
    assert "Confusion matrix" in "\n".join(m.value for m in results.markdown)
    assert any(c.proto.id and "confusion" in str(c.proto) for c in results.get("plotly_chart")) \
        or len(results.get("plotly_chart")) >= 3


def test_demo_eb_leads_with_the_physics_reason(offline):
    at = _analyze(300871545)
    why = next(m.value for m in at.tabs[3].markdown if str(m.value).startswith("- "))
    assert why.startswith("- A clear second eclipse was found halfway through the orbit")


def test_eb_and_variable_extras(offline):
    eb = _analyze(300871545)
    assert len(eb.tabs[2].get("plotly_chart")) == 3          # fold + 2 eclipse comparisons
    var = _analyze(441584047)
    assert len(var.tabs[1].get("plotly_chart")) == 2         # BLS + Lomb-Scargle


def test_live_failure_shows_a_friendly_error(monkeypatch):
    from src import pipeline
    from src.errors import DataDownloadError

    def timeout(*a, **k):
        raise DataDownloadError("MAST timed out")

    monkeypatch.setattr(pipeline, "analyze_target", timeout)
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT).run()
    at.checkbox(key="live").check().run()
    at.button(key="analyze").click().run()
    assert not at.exception
    errs = " ".join(e.value for e in at.error)
    assert "did not respond in time" in errs
    assert "Traceback" not in errs                    # details stay in the collapsed expander
    assert at.expander and at.expander[0].label == "Technical details"
    assert not at.tabs


def test_catalog_period_check(offline):
    eb = _analyze(300871545)
    text = "\n".join(m.value for m in eb.tabs[3].markdown)
    assert "Catalog period: <b>4.8171 d</b>" in text and "✅ match" in text
    tricky = _analyze(329864959)
    text = "\n".join(m.value for m in tricky.tabs[3].markdown)
    assert "we found half the catalog period" in text
    warn = " ".join(w.value for w in tricky.tabs[3].warning)
    assert "Our pipeline measures 4.52 d, NASA's catalog says 9.05 d." in warn
    assert "Known limitation" in warn
    side = " ".join(m.value for m in tricky.sidebar.markdown)
    assert "Our pipeline measures 4.52 d, NASA's catalog says 9.05 d." in side


def test_router_reasons_use_full_class_names():
    for tic in TICS:
        rep = json.loads((CONFIG.paths.examples / "demo" / f"{tic}.json").read_text("utf-8"))
        for r in rep["routing"]["reasons"]:
            assert " Eb " not in f" {r} " and "Eb." not in r and "Eb 1" not in r, r
