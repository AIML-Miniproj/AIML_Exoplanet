"""Vetting features separate planets, eclipsing binaries, variables and noise."""

import math

import numpy as np
import pytest

from config import CONFIG
from src.detect.bls import search
from src.ingest.resolver import TargetInfo
from src.preprocess.preprocessor import bin_lightcurve, preprocess, redetrend_masked
from src.vetting.features import FEATURE_NAMES, MODEL_FEATURES, extract_features

SUN = TargetInfo(tic_id=1, teff=5800, rad=1.0, mass=1.0, logg=4.44, rho=1.0, tmag=9.0,
                 contratio=0.01)


def run(coll, star=SUN):
    pre = preprocess(coll)
    lc = bin_lightcurve(pre.lc_flat, CONFIG.search_bin_minutes)
    sig = search(lc).signal
    pre = redetrend_masked(pre, [sig.ephemeris])
    feats, details = extract_features(pre, sig, star)
    return sig, feats, details


@pytest.fixture(scope="module")
def planet():
    from tests.conftest import make_planet_lc
    return run(make_planet_lc(period=3.3, depth=0.004, duration=0.12, n_sectors=3, noise=400))


@pytest.fixture(scope="module")
def eb_secondary():
    """EB with a clearly weaker secondary: detected at the true period."""
    from tests.conftest import make_eb_lc
    return run(make_eb_lc(period=2.4, depth1=0.12, depth2=0.02, n_sectors=2,
                          ellipsoidal_amp=0.003))


@pytest.fixture(scope="module")
def eb_similar():
    """EB with two similar eclipses: the search locks onto half the true period."""
    from tests.conftest import make_eb_lc
    return run(make_eb_lc(period=3.0, depth1=0.10, depth2=0.07, n_sectors=2))


def test_all_features_present(planet):
    _, feats, _ = planet
    assert list(feats) == FEATURE_NAMES
    assert set(MODEL_FEATURES) <= set(FEATURE_NAMES)
    assert "n_sectors" not in MODEL_FEATURES


def test_odd_even_separates_eb_from_planet(planet, eb_similar):
    sig_eb, f_eb, _ = eb_similar
    assert sig_eb.period == pytest.approx(1.5, rel=0.01)       # half the true period
    assert f_eb["odd_even_depth_sigma"] > 5
    assert f_eb["odd_even_depth_ratio"] < 0.85
    _, f_pl, _ = planet
    assert f_pl["odd_even_depth_sigma"] < 3
    assert f_pl["odd_even_depth_ratio"] > 0.9


def test_secondary_snr_high_for_eb(planet, eb_secondary):
    sig, f_eb, _ = eb_secondary
    assert sig.period == pytest.approx(2.4, rel=0.01)
    assert f_eb["secondary_snr"] > 7
    assert f_eb["secondary_best_phase"] == pytest.approx(0.5, abs=0.02)
    assert f_eb["ellip_amp_ppm"] > 1000 and f_eb["ellip_snr"] > 5
    assert planet[1]["secondary_snr"] < 3


def test_shape_separates_v_from_u(planet, eb_secondary):
    assert eb_secondary[1]["ingress_to_total_ratio"] > 0.35
    assert planet[1]["ingress_to_total_ratio"] < 0.25
    assert planet[1]["trapezoid_chi2_vs_box_chi2"] <= 1.0


def test_planet_physics_features(planet):
    _, f, _ = planet
    # depth 0.004 on a Sun-sized star -> ~6.9 Earth radii
    assert f["implied_rp_rearth"] == pytest.approx(6.9, rel=0.15)
    assert f["implied_rp_gt_2rjup"] == 0.0
    assert 0.5 < f["duration_ratio_expected"] < 1.5
    assert f["n_sectors"] == 3 and f["n_sectors_with_transit"] == 3
    assert f["sector_depth_cv"] < 0.2


def test_ls_recovers_variable_period(variable_factory):
    for kind, period in (("sinusoid", 1.7), ("spotted", 4.3), ("multiperiodic", 0.21)):
        _, f, _ = run(variable_factory(kind=kind, period=period, amp=0.01, n_sectors=2))
        assert f["ls_period_1"] == pytest.approx(period, rel=0.01), kind
        assert f["ls_log10_fap"] < -10
    _, f, _ = run(variable_factory(kind="sinusoid", period=1.7, amp=0.01, n_sectors=2))
    assert f["acf_period"] == pytest.approx(1.7, rel=0.05)
    assert f["fourier_R21"] < 0.1
    assert f["ls_period_sector_std"] < 0.01


def test_noise_features_are_finite_or_nan(noise_factory):
    sig, f, _ = run(noise_factory(n_sectors=2, n_spikes=4), star=TargetInfo(tic_id=2))
    assert not sig.detected()
    assert math.isnan(f["teff"]) and math.isnan(f["implied_rp_rearth"])
    assert np.isfinite(f["rms_ppm"]) and np.isfinite(f["sde"])
