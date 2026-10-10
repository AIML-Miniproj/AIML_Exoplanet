"""Periodic-dip search."""

from dataclasses import replace

import lightkurve as lk
import numpy as np
import pytest

from config import CONFIG
from src.detect.bls import distinct_peaks, mask_signal, multi_search, search
from src.detect.tls import tls_available
from src.preprocess.preprocessor import bin_lightcurve, preprocess
from tests.conftest import T_START, make_planet_lc


def _flat(coll):
    return bin_lightcurve(preprocess(coll).lc_flat, CONFIG.search_bin_minutes)


def test_recovers_planet_period_and_depth(planet_factory):
    lc = _flat(planet_factory(period=3.3, depth=0.004, duration=0.12, n_sectors=2, noise=400))
    res = search(lc)
    sig = res.signal
    assert sig.period == pytest.approx(3.3, rel=0.001)
    assert sig.depth == pytest.approx(0.004, rel=0.2)
    assert sig.sde > CONFIG.sde_min and sig.detected()
    assert sig.stats_2p, "2P must always be evaluated"
    assert len(sig.stats["per_transit_count"]) >= 15
    # top peaks are distinct, not harmonics of each other
    ps = [p.period for p in sig.top_peaks]
    for i, a in enumerate(ps):
        for b in ps[i + 1:]:
            assert all(abs(a / (b * r) - 1) > 0.009 for r in (1, 0.5, 2, 1 / 3, 3))


def test_noise_is_not_detected(noise_factory):
    res = search(_flat(noise_factory(n_sectors=2, red=0, n_spikes=0)))
    assert not res.signal.detected()


def test_coarse_grid_is_refined_across_a_long_gap():
    """Two sectors ~2 years apart need a huge grid; the coarse+refine path must still
    land on the exact period."""
    period, t0 = 4.1234, T_START + 1.0
    a = make_planet_lc(period=period, t0=t0, depth=0.005, n_sectors=1, first_sector=1, seed=1)
    b = make_planet_lc(period=period, t0=t0, depth=0.005, n_sectors=1, first_sector=27, seed=2)
    cfg = replace(CONFIG, max_grid_size=20_000)
    res = search(_flat(lk.LightCurveCollection([a[0], b[0]])), cfg)
    assert res.grid_size_full > res.grid_size_used
    assert res.signal.period == pytest.approx(period, rel=2e-4)


def test_multi_signal_search_finds_two_planets(planet_factory):
    from tests.conftest import build_collection, trapezoid

    def model(t):
        return (1 - trapezoid(t, 2.7, T_START + 0.5, 0.006, 0.1, 0.1)
                - trapezoid(t, 5.9, T_START + 1.3, 0.003, 0.15, 0.1))

    lc = _flat(build_collection(model, n_sectors=3, noise_ppm=300))
    cfg = replace(CONFIG, max_signals=3)
    results = multi_search(lc, cfg)
    periods = sorted(r.signal.period for r in results[:2])
    assert periods[0] == pytest.approx(2.7, rel=0.002)
    assert periods[1] == pytest.approx(5.9, rel=0.002)
    assert len(results) <= 3


def test_mask_signal_removes_transits(planet_factory):
    lc = _flat(planet_factory(n_sectors=1))
    sig = search(lc).signal
    masked = mask_signal(lc, sig)
    assert len(masked) < len(lc)
    assert np.min(masked.flux.value) > 1 - 0.5 * sig.depth


def test_distinct_peaks_skips_aliases():
    periods = np.linspace(1, 10, 9001)
    power = np.exp(-((periods - 3) / 0.01) ** 2) + 0.8 * np.exp(-((periods - 6) / 0.01) ** 2) \
        + 0.5 * np.exp(-((periods - 4.4) / 0.01) ** 2)
    idx = distinct_peaks(periods, power, 2)
    assert [round(periods[i], 2) for i in idx] == [3.0, 4.4]


@pytest.mark.skipif(not tls_available(), reason="transitleastsquares not installed")
def test_tls_matches_bls(planet_factory):
    from src.detect.tls import search_tls

    lc = _flat(planet_factory(period=3.3, n_sectors=1))
    assert search_tls(lc).period == pytest.approx(3.3, rel=0.01)
