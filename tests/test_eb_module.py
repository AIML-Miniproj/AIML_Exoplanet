"""EB true-period rule on synthetic binaries."""

import pytest

from src.recycler.eb_module import true_period
from tests.test_vetting import run


def test_similar_eclipses_detected_at_half_period_are_doubled(eb_factory):
    sig, f, _ = run(eb_factory(period=3.0, depth1=0.10, depth2=0.07, n_sectors=2))
    assert sig.period == pytest.approx(1.5, rel=0.01)
    tp = true_period(sig.period, f)
    assert tp.period_d == pytest.approx(3.0, rel=0.01) and tp.rule == "odd_even_differ"


def test_visible_secondary_keeps_detected_period(eb_factory):
    sig, f, _ = run(eb_factory(period=2.4, depth1=0.12, depth2=0.02, n_sectors=2))
    assert sig.period == pytest.approx(2.4, rel=0.01)
    tp = true_period(sig.period, f)
    assert tp.period_d == pytest.approx(2.4, rel=0.01) and tp.rule == "secondary_seen"


def test_no_evidence_either_way_is_flagged_ambiguous():
    tp = true_period(1.2, {"odd_even_depth_sigma": 0.5, "secondary_snr": 0.8})
    assert tp.period_d == pytest.approx(2.4) and tp.ambiguous
