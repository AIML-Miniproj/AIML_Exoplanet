"""Preprocessing: per-sector cleaning, upper-only clipping, stitching and detrending."""

import numpy as np
import pytest

from config import CONFIG
from src.preprocess.preprocessor import (bin_lightcurve, preprocess, redetrend_masked,
                                         transit_mask, upper_outlier_mask)


def _phase_dist(t, period, t0):
    return np.abs((t - t0 + 0.5 * period) % period - 0.5 * period)


def test_eb_eclipse_depth_survives(eb_factory):
    """Regression for the old symmetric clip, which deleted deep eclipses."""
    period, t0, depth1 = 2.4, 1325.3 + 0.21 * 2.4, 0.30
    coll = eb_factory(period=period, t0=t0, depth1=depth1, depth2=0.05, noise=200)
    pre = preprocess(coll)
    for lc in (pre.lc_norm, pre.lc_flat):
        t, f = lc.time.value, lc.flux.value
        core = _phase_dist(t, period, t0) < 0.004    # bottom of the V
        assert core.sum() > 5
        assert 1 - np.median(f[core]) == pytest.approx(depth1, rel=0.1)
    assert pre.stats["n_clipped"] < 0.002 * pre.stats["n_cadences"]


def test_gaps_do_not_break_detrending(noise_factory):
    coll = noise_factory(n_sectors=3, n_spikes=0)
    pre = preprocess(coll)
    f = pre.lc_flat.flux.value
    assert np.all(np.isfinite(f))
    assert len(pre.lc_flat) >= 0.98 * len(pre.lc_norm)
    assert np.median(f) == pytest.approx(1.0, abs=1e-3)
    assert set(pre.sectors) == {1, 2, 3}


def test_lc_norm_keeps_sinusoid_amplitude(variable_factory):
    period, amp = 1.7, 0.01
    pre = preprocess(variable_factory(kind="sinusoid", period=period, amp=amp, noise=200))
    def fitted_amp(lc):
        t, f = lc.time.value, lc.flux.value
        X = np.column_stack([np.ones_like(t), np.sin(2 * np.pi * t / period),
                             np.cos(2 * np.pi * t / period)])
        c, *_ = np.linalg.lstsq(X, f, rcond=None)
        return np.hypot(c[1], c[2])

    assert fitted_amp(pre.lc_norm) == pytest.approx(amp, rel=0.05)
    # ...while the flattened curve has most of it removed
    assert fitted_amp(pre.lc_flat) < 0.5 * amp


def test_each_sector_normalised_to_one(planet_factory):
    pre = preprocess(planet_factory(n_sectors=3))
    sec = pre.lc_norm["sector"].value
    for s in (1, 2, 3):
        assert np.median(pre.lc_norm.flux.value[sec == s]) == pytest.approx(1.0, abs=1e-3)


def test_upper_only_clip_and_quality_mask(noise_factory):
    coll = noise_factory(n_sectors=1, n_spikes=0, red=0)
    lc = coll[0]
    f = lc.flux.value.copy()
    med = np.median(f)
    f[100] = med * 1.05     # upward spike: removed
    f[200] = med * 0.95     # downward dip: kept
    lc.flux = f * lc.flux.unit
    n_flagged = int(np.sum(lc["quality"].value > 0))
    pre = preprocess([lc])
    t = pre.lc_norm.time.value
    assert lc.time.value[100] not in t
    assert lc.time.value[200] in t
    assert pre.stats["n_quality_removed"] == n_flagged


def test_upper_outlier_mask_ignores_dips():
    t = np.linspace(0, 5, 3000)
    f = 1 + np.random.default_rng(0).normal(0, 1e-3, t.size)
    f[1000:1010] -= 0.5
    f[2000] += 0.05
    m = upper_outlier_mask(t, f, 5.0, 0.02)
    assert m[2000] and not m[1000:1010].any()


def test_detrend_window_is_three_times_longest_duration():
    assert CONFIG.detrend_window() >= 3 * CONFIG.search_durations().max()
    assert CONFIG.search_durations().max() < CONFIG.period_min_d


def test_masked_redetrend_protects_long_transit(planet_factory):
    # A long transit on a varying star: the first pass eats into the dip, the masked
    # second pass should recover the depth more fully.
    period, t0, depth, dur = 4.0, 1326.0, 0.004, 0.3
    coll = planet_factory(period=period, t0=t0, depth=depth, duration=dur, noise=100)
    for lc in coll:
        t = lc.time.value
        lc.flux = lc.flux * (1 + 0.01 * np.sin(2 * np.pi * t / 6.0))
    pre = preprocess(coll)
    pre2 = redetrend_masked(pre, [(period, t0, dur)])

    def depth_of(p):
        t, f = p.lc_flat.time.value, p.lc_flat.flux.value
        core = _phase_dist(t, period, t0) < 0.3 * dur
        return 1 - np.median(f[core])

    assert abs(depth_of(pre2) - depth) < abs(depth_of(pre) - depth) + 1e-5
    assert depth_of(pre2) == pytest.approx(depth, rel=0.15)
    assert pre2.stats["redetrended"]


def test_transit_mask_and_binning(planet_factory):
    pre = preprocess(planet_factory(cadence=2.0, n_sectors=1))
    m = transit_mask(pre.lc_flat.time.value, 3.0, 1326.41, 0.12)
    assert 0 < m.mean() < 0.1
    binned = bin_lightcurve(pre.lc_flat, 10)
    assert len(binned) < len(pre.lc_flat) / 4
    assert np.all(np.isfinite(binned.flux.value))
    assert bin_lightcurve(binned, 10) is binned
