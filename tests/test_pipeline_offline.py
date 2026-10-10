"""End-to-end smoke test of the pipeline on a synthetic planet, no network."""

import pytest

from src.pipeline import analyze_target


def test_pipeline_recovers_synthetic_planet(planet_factory):
    coll = planet_factory(period=3.0, depth=0.006, duration=0.12, n_sectors=2)
    stages = []
    report = analyze_target("synthetic planet", collection=coll,
                            progress_cb=lambda s, f: stages.append((s, f)))
    assert report.status == "ok"
    assert report.signals[0]["period"] == pytest.approx(3.0, rel=0.01)
    assert report.signals[0]["depth_ppm"] == pytest.approx(6000, rel=0.3)
    assert stages[0][1] == 0.0 and stages[-1][1] == 1.0
    assert report.signals[0]["detected"]
    assert report.features["odd_even_depth_sigma"] < 3
    assert report.triage["status"] in ("ok", "no_model")
    assert {"lc_norm", "lc_flat", "periodogram", "fold"} <= set(report.plot_data)
    assert report.to_json()   # serialisable


@pytest.mark.parametrize("kind,period", [("multiperiodic", 0.13), ("sinusoid", 0.526)])
def test_short_period_pulsator_does_not_break_the_pipeline(variable_factory, kind, period):
    """Regression: on a fast pulsator BLS used to fit a "dip" lasting most of the
    cycle, the masked re-detrend then hid every point and the run crashed."""
    coll = variable_factory(kind=kind, period=period, amp=0.02, n_sectors=2)
    report = analyze_target("synthetic pulsator", collection=coll)
    sig = report.signals[0]
    assert sig["duration"] <= 0.2 * sig["period"] * 1.02     # astropy rounds durations ~1%
    assert report.features["duty_cycle"] <= 0.2 * 1.02
    assert report.preprocessing["n_cadences"] > 1000
