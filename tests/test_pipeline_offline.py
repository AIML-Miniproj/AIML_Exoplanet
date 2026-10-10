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
