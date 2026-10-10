"""Router and recycler modules (no network)."""

import math
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from config import CONFIG
from src.ingest.resolver import TargetInfo
from src.recycler import eb_module, planet_module, systematics_log, variable_module
from src.recycler.router import NEEDS_REVIEW, confidence_label, route

SUN_LIKE = TargetInfo(tic_id=1, teff=6050, teff_err=100, rad=1.19, rad_err=0.05, mass=1.13,
                      mass_err=0.15, logg=4.3, rho=0.68)


def probs(**kw):
    p = {"PLANET": 0.0, "EB": 0.0, "VARIABLE": 0.0, "NOISE": 0.0}
    p.update(kw)
    return p


# ── Router ───────────────────────────────────────────────────

def test_confident_planet_goes_to_planet_module():
    d = route(probs(PLANET=0.9, EB=0.05, NOISE=0.05), {"implied_rp_rearth": 15})
    assert d.primary_class == "PLANET" and d.confidence_label == "High"
    assert d.modules == ["planet"] and not d.needs_review


def test_unsure_planet_is_kept_for_review_not_discarded():
    d = route(probs(NOISE=0.45, PLANET=0.40, EB=0.1, VARIABLE=0.05), {})
    assert d.primary_class == NEEDS_REVIEW and d.needs_review
    assert d.modules == ["noise", "planet"]
    assert any("kept for review" in r for r in d.reasons)
    assert d.confidence_label == "Low"


def test_too_big_for_a_planet_is_overridden():
    d = route(probs(PLANET=0.7, EB=0.2, NOISE=0.1), {"implied_rp_rearth": 40})  # ~3.6 R_jup
    assert d.primary_class != "PLANET" and "PLANET" not in d.modules
    assert any("too big to be a planet" in r for r in d.reasons)


def test_clear_second_eclipse_forces_eb():
    feats = {"secondary_snr": 12, "secondary_depth_ppm": 3000, "refl_amp_ppm": 100}
    d = route(probs(PLANET=0.6, EB=0.3, NOISE=0.1), feats)
    assert d.primary_class == "EB" and d.modules == ["eb"]
    assert d.reasons[0].startswith("A clear second eclipse was found halfway through the orbit")


def test_second_dip_explained_by_reflection_keeps_planet():
    feats = {"secondary_snr": 8, "secondary_depth_ppm": 150, "refl_amp_ppm": 90}
    d = route(probs(PLANET=0.85, EB=0.1, NOISE=0.05), feats)
    assert d.primary_class == "PLANET"
    assert any("reflected" in r for r in d.reasons)


def test_review_mentions_a_detected_repeating_dip():
    d = route(probs(VARIABLE=0.39, NOISE=0.26, PLANET=0.2, EB=0.15), {},
              signal={"period": 3.31, "detected": True})
    assert d.needs_review and any("repeating dip was found every 3.3 days" in r
                                  for r in d.reasons)
    d = route(probs(VARIABLE=0.39, NOISE=0.26, PLANET=0.2, EB=0.15), {},
              signal={"period": 3.31, "detected": False})
    assert not any("repeating dip" in r for r in d.reasons)


def test_no_model_means_review():
    assert route(None, {}).primary_class == NEEDS_REVIEW


def test_confidence_labels_follow_config():
    assert [confidence_label(p) for p in (0.9, 0.8, 0.6, 0.5, 0.3)] == \
        ["High", "High", "Medium", "Medium", "Low"]
    cfg = replace(CONFIG, conf_high=0.95)
    assert confidence_label(0.9, cfg) == "Medium"


# ── Planet module ────────────────────────────────────────────

def test_planet_module_hot_jupiter():
    sig = {"period": 3.5247, "depth": 0.0138, "depth_err": 0.0001}
    t = time.perf_counter()
    mc, missing = planet_module.monte_carlo(sig, SUN_LIKE, n=1000)
    assert time.perf_counter() - t < 1.0 and not missing
    lo, med, hi = mc["rp_rjup"]
    assert med == pytest.approx(1.36, rel=0.05) and lo < med < hi
    res = planet_module.characterize(sig, SUN_LIKE, None)
    assert res.details["size_category"] == "Jupiter-size"
    assert "Jupiter" in res.plain_summary and "3.5 days" in res.plain_summary
    labels = [k.label for k in res.key_facts]
    assert {"Size", "Orbit period", "Distance from star", "Temperature"} <= set(labels)


def test_size_categories_and_missing_errors():
    assert [planet_module.size_category(r) for r in (1.0, 1.6, 3.0, 11.0, 40.0)] == \
        ["Earth-size", "super-Earth", "Neptune-size", "Jupiter-size",
         "larger than any known planet"]
    star = TargetInfo(tic_id=2, teff=5000, rad=0.8, mass=0.8)            # no errors
    _, missing = planet_module.monte_carlo({"period": 2, "depth": 1e-3}, star, n=100)
    assert {"dip depth", "star radius", "star mass", "star temperature"} == set(missing)


def test_known_planet_lookup_uses_cached_toi_table(tmp_path):
    raw = tmp_path / "labels" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({"toi": [5972.01], "tid": [420814525], "tfopwg_disp": ["KP"],
                  "pl_orbper": [3.524755]}).to_parquet(raw / "toi.parquet")
    cfg = replace(CONFIG, paths=replace(CONFIG.paths, labels=tmp_path / "labels"))
    planet_module._toi_table.cache_clear()
    m = planet_module.known_planet(420814525, 3.5248, cfg)
    assert m["toi"] == "TOI 5972.01" and m["status"] == "a known planet"
    assert planet_module.known_planet(420814525, 1.7624, cfg)["ratio"] == 0.5   # found at P/2
    assert planet_module.known_planet(1, 3.5, cfg) is None
    planet_module._toi_table.cache_clear()


# ── EB module ────────────────────────────────────────────────

def test_eb_eclipses_measured_at_true_period(eb_factory):
    from tests.conftest import T_START

    t0 = T_START + 0.7
    lc = eb_factory(period=3.0, depth1=0.10, depth2=0.04, phase2=0.5, duration=0.15,
                    n_sectors=2, t0=t0).stitch()
    t, f = lc.time.value, lc.flux.value
    # start from the secondary's epoch: the module must make the deeper eclipse primary
    ecl = eb_module.measure_eclipses(t, f, 3.0, t0 + 1.5, 0.15)
    assert ecl["primary_depth"] == pytest.approx(0.10, rel=0.25)
    assert ecl["secondary_depth"] == pytest.approx(0.04, rel=0.3)
    assert ecl["secondary_phase"] == pytest.approx(0.5, abs=0.02)
    assert abs(ecl["ecosw"]) < 0.05
    assert ecl["temp_ratio"] == pytest.approx((0.04 / 0.10) ** 0.25, rel=0.15)


def test_eb_morphology_rules():
    detached = {"depth_ppm": 100_000, "ellip_amp_ppm": 500, "refl_amp_ppm": 0, "t14_hr": 4}
    assert eb_module.morphology(detached, 5.0)[0] == "detached"
    semi = dict(detached, ellip_amp_ppm=12_000)
    assert eb_module.morphology(semi, 5.0)[0] == "semi-detached"
    contact = dict(detached, ellip_amp_ppm=40_000)
    assert eb_module.morphology(contact, 0.4)[0] == "contact"
    assert [eb_module.catalog_morph_class(m) for m in (0.2, 0.6, 0.85)] == \
        ["detached", "semi-detached", "contact"]


def test_eb_summary_mentions_ambiguous_period():
    sig = {"period": 2.0, "t0": 0.0, "duration": 0.1}
    res = eb_module.characterize(sig, {"odd_even_depth_sigma": 0.5, "secondary_snr": 0.5,
                                       "depth_ppm": 50_000, "t14_hr": 2.5})
    assert "every 2.0 days or 4.0 days; this data can't tell which" in res.plain_summary
    period_fact = next(k for k in res.key_facts if k.label == "Orbital period")
    assert period_fact.value == "2.0000 or 4.0000"


# ── Variable module ──────────────────────────────────────────

@pytest.mark.parametrize("feats,expected", [
    (dict(var_period_d=0.63, amp_p2p_ppm=280_000, ls_power_1=0.8, ls_log10_fap=-50), "RR_LYRAE"),
    (dict(var_period_d=5.4, amp_p2p_ppm=200_000, ls_power_1=0.9, ls_log10_fap=-50), "CEPHEID"),
    (dict(var_period_d=0.08, amp_p2p_ppm=8_000, ls_power_1=0.6, ls_log10_fap=-50),
     "DELTA_SCUTI/GAMMA_DOR"),
    (dict(var_period_d=1.1, amp_p2p_ppm=5_000, ls_power_1=0.7, ls_log10_fap=-50, teff=7200),
     "DELTA_SCUTI/GAMMA_DOR"),
    (dict(var_period_d=6.4, amp_p2p_ppm=14_000, ls_power_1=0.4, ls_log10_fap=-50, teff=5200),
     "ROTATION/SPOTTED"),
    (dict(var_period_d=2.0, amp_p2p_ppm=900, ls_power_1=0.01, ls_log10_fap=-0.5), "OTHER"),
])
def test_variable_rules(feats, expected):
    assert variable_module.classify(feats)[0] == expected


def test_variable_summary_is_plain():
    res = variable_module.characterize(dict(var_period_d=0.63, amp_p2p_ppm=280_000,
                                            ls_power_1=0.8, ls_log10_fap=-50))
    assert res.plain_summary.startswith("Not a planet, but still interesting")
    assert "0.63" not in res.plain_summary and "15 hours" in res.plain_summary


# ── Systematics log ──────────────────────────────────────────

def test_only_noise_and_review_are_logged(tmp_path):
    cfg = replace(CONFIG, paths=replace(CONFIG.paths, logs=tmp_path))
    base = {"tic_id": 5, "target": "TIC 5", "signals": [{"period": 1.0, "sde": 5.0}],
            "sectors": [{"sector": 3}], "triage": {"model_version": "test"}}
    assert systematics_log.append(dict(base, routing={"primary_class": "PLANET"}), cfg) is None
    systematics_log.append(dict(base, routing={"primary_class": "NOISE", "reasons": ["x"]}), cfg)
    systematics_log.append(dict(base, routing={"primary_class": NEEDS_REVIEW}), cfg)
    df = pd.read_parquet(tmp_path / "recycler_log.parquet")
    assert list(df["primary_class"]) == [NEEDS_REVIEW]          # same star + model: latest
    systematics_log.append(dict(base, tic_id=6, routing={"primary_class": "NOISE"}), cfg)
    systematics_log.append(dict(base, triage={"model_version": "v2"},
                                routing={"primary_class": "NOISE"}), cfg)
    df = pd.read_parquet(tmp_path / "recycler_log.parquet")
    assert len(df) == 3
    assert set(zip(df.tic_id, df.model_version)) == {(5, "test"), (6, "test"), (5, "v2")}


def test_noise_module_summary():
    res = systematics_log.characterize({"period": 7.1, "sde": 5.2, "depth_ppm": 300,
                                        "detected": False}, {}, ["unsure"])
    assert "noise" in res.plain_summary and len(res.key_facts) == 4


# ── End to end ───────────────────────────────────────────────

def test_pipeline_fills_summary_and_key_facts(planet_factory):
    from src.pipeline import analyze_target
    rep = analyze_target("synthetic", collection=planet_factory(period=3.0, depth=0.006,
                                                                duration=0.12, n_sectors=2),
                         target_info=SUN_LIKE)
    assert rep.routing and rep.plain_summary
    assert rep.key_facts and all(len(k) == 4 for k in rep.key_facts)
    assert set(rep.modules) == set(rep.routing["modules"])
    assert math.isfinite(rep.physical["rp_rearth"])
    assert rep.to_json()


def _alternating_eclipses(width_b, depth_b=0.05, period=1.5, days=54.0, seed=0):
    """Eclipses every ``period``: even ones 0.10 d wide and 5% deep, odd ones
    ``width_b`` wide and ``depth_b`` deep (2-minute cadence, 0.1% noise)."""
    from tests.conftest import trapezoid

    rng = np.random.default_rng(seed)
    t = np.arange(0, days, 2 / 1440)
    a = trapezoid(t, 2 * period, 0.3, 0.05, 0.10, 0.3)
    b = trapezoid(t, 2 * period, 0.3 + period, depth_b, width_b, 0.3)
    return t, 1 - a - b + rng.normal(0, 1e-3, t.size)


def test_alternate_eclipse_check_finds_different_widths():
    t, f = _alternating_eclipses(width_b=0.16)                   # same depth, wider
    chk = eb_module.odd_even_at_double(t, f, 1.5, 0.3, 0.12)
    assert chk["sigma_width"] >= 3 and chk["n_odd"] >= 15 and chk["n_even"] >= 15
    # measured for information only: the period decision does not use it
    sig = {"period": 1.5, "t0": 0.3, "duration": 0.12}
    tp, info = eb_module.true_period_with_data(sig, {"odd_even_depth_sigma": 0.5,
                                                     "secondary_snr": 0.5}, t, f)
    assert info["alternate_eclipses"]["sigma_width"] >= 3
    assert tp.period_d == pytest.approx(1.5) and tp.ambiguous


def test_period_rule_uses_the_modules_own_secondary(eb_factory):
    """Feature says no second eclipse, but the module's own measurement on all the
    data sees one: the detected period is the true one (not ambiguous)."""
    from tests.conftest import T_START

    t0 = T_START + 0.7
    lc = eb_factory(period=3.0, depth1=0.10, depth2=0.01, phase2=0.5, duration=0.15,
                    n_sectors=2, t0=t0).stitch()
    sig = {"period": 3.0, "t0": t0, "duration": 0.15}
    feats = {"odd_even_depth_sigma": 0.5, "secondary_snr": 0.5}
    assert eb_module.true_period(3.0, feats).ambiguous
    tp, info = eb_module.true_period_with_data(sig, feats, lc.time.value, lc.flux.value)
    assert info["secondary_source"] == "module" and info["secondary_snr"] >= 3
    assert tp.rule == "secondary_seen" and tp.period_d == pytest.approx(3.0)


def test_alternate_eclipse_check_quiet_for_identical_eclipses():
    t, f = _alternating_eclipses(width_b=0.10)
    chk = eb_module.odd_even_at_double(t, f, 1.5, 0.3, 0.12)
    assert all(chk[k] < 3 for k in ("sigma_depth", "sigma_width", "sigma_shape"))
    tp, _ = eb_module.true_period_with_data({"period": 1.5, "t0": 0.3, "duration": 0.12},
                                            {"odd_even_depth_sigma": 0.5,
                                             "secondary_snr": 0.5}, t, f)
    assert tp.ambiguous and tp.period_d == pytest.approx(1.5)


def test_measured_morphology_uses_fitted_width():
    ecl = {"primary_depth": 0.10, "primary_width_d": 0.15, "out_of_eclipse_amp": 0.001}
    assert eb_module.morphology_measured(ecl, 5.0)[0] == "detached"
    assert eb_module.morphology_measured(dict(ecl, primary_width_d=1.0), 5.0)[0] == "contact"
    assert (eb_module.morphology_measured(dict(ecl, out_of_eclipse_amp=0.012), 5.0)[0]
            == "semi-detached")
    assert eb_module.morphology_measured({}, 5.0) is None
