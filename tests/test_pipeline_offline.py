"""End-to-end smoke test of the pipeline on a synthetic planet, no network."""

import numpy as np
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


def _planet_on_fast_variable(var_amp, seed=4):
    from tests.conftest import T_START, build_collection, trapezoid

    def model(t):
        return (1 + var_amp * np.sin(2 * np.pi * t / 1.19)
                + 0.4 * var_amp * np.sin(2 * np.pi * t / 0.61 + 1.0)
                - trapezoid(t, 3.31, T_START + 0.7, 0.005, 0.11, 0.015))

    return build_collection(model, n_sectors=2, noise_ppm=600, seed=seed)


def test_prewhitening_uncovers_a_transit_under_fast_variability():
    """Transit (0.5%, 3.31 d) on a star varying by ~1% every ~1.2 days."""
    report = analyze_target("synthetic active star", collection=_planet_on_fast_variable(0.006))
    removed = report.preprocessing["prewhitened"]
    assert removed and removed[0]["period_d"] == pytest.approx(1.19, rel=0.02)
    pw = report.extra["prewhitening"]
    assert pw["used"] and pw["sde"] - pw["sde_without"] >= 1.0
    assert report.signals[0]["period"] == pytest.approx(3.31, rel=0.002)
    assert report.signals[0]["depth_ppm"] == pytest.approx(5000, rel=0.35)   # untouched curve


def test_prewhitening_leaves_a_quiet_planet_alone(planet_factory):
    coll = planet_factory(period=3.0, depth=0.006, duration=0.12, n_sectors=2)
    report = analyze_target("quiet planet", collection=coll)
    assert report.preprocessing["prewhitened"] == []


def test_prewhitening_does_not_remove_a_contact_binary():
    """Near-sinusoidal binary (two almost equal minima per orbit): its "variability" is
    the signal itself, so the prewhitened search must not replace the normal one."""
    from tests.conftest import build_collection

    p_orb = 2.55

    def model(t):
        x = 2 * np.pi * t / p_orb
        return 1 - 0.008 * np.cos(x) ** 2 + 0.0004 * np.cos(x)

    report = analyze_target("synthetic contact binary",
                            collection=build_collection(model, n_sectors=2, noise_ppm=400))
    pw = report.extra.get("prewhitening")
    assert pw is None or not pw["used"]
    ratio = report.signals[0]["period"] / p_orb
    assert min(abs(ratio - 0.5), abs(ratio - 1.0)) < 0.01
