"""Flare module on synthetic light curves (2-minute cadence)."""

import numpy as np
import pytest

from src.ingest.resolver import TargetInfo
from src.recycler import flare_module
from tests.conftest import T_START, build_collection


def flare_shape(t: np.ndarray, t_peak: float, amp: float, rise_d: float, decay_d: float):
    """Fast linear rise, exponential decay: the classic flare profile."""
    x = t - t_peak
    rise = np.where((x > -rise_d) & (x <= 0), amp * (1 + x / rise_d), 0.0)
    decay = np.where(x > 0, amp * np.exp(-np.clip(x, 0, None) / decay_d), 0.0)
    return rise + decay


def flaring_lc(peaks, amp=0.02, noise=400.0, n_spikes=0, seed=1):
    def model(t):
        f = np.ones_like(t)
        for tp in peaks:
            f += flare_shape(t, tp, amp, 4 / 1440, 15 / 1440)
        return f
    return build_collection(model, n_sectors=1, cadence_min=2.0, noise_ppm=noise,
                            n_spikes=n_spikes, seed=seed)


M_DWARF = TargetInfo(tic_id=1, teff=3300.0, rad=0.3)


def test_finds_injected_flares():
    peaks = [T_START + d for d in (2.13, 5.71, 9.42, 17.08, 21.66)]
    coll = flaring_lc(peaks)
    res = flare_module.characterize(coll, M_DWARF)
    d = res.details
    assert d["checked"] and d["is_flare_star"]
    found = np.array([f["t_peak"] for f in d["flares"]])
    t = coll[0].time.value
    # every injected flare whose peak is covered by data is recovered within a few cadences
    covered = [tp for tp in peaks if np.min(np.abs(t - tp)) < 3 / 1440]
    assert len(covered) >= 3
    for tp in covered:
        assert np.any(np.abs(found - tp) < 10 / 1440), f"flare at {tp:.3f} not found: {found}"
    assert len(found) <= len(peaks)
    big = max(d["flares"], key=lambda f: f["amplitude"])
    assert big["amplitude"] == pytest.approx(0.02, rel=0.3)
    assert 10 < big["equiv_duration_s"] < 60 * 60
    assert all(f["log10_energy_erg"] and 29 < f["log10_energy_erg"] < 36 for f in d["flares"])


def test_single_point_spikes_are_not_flares():
    res = flare_module.characterize(flaring_lc([], n_spikes=40), M_DWARF)
    assert res.details["n_flares"] == 0
    assert not res.details["is_flare_star"]


def test_quiet_star_has_no_flares():
    res = flare_module.characterize(flaring_lc([], noise=300.0, seed=5), M_DWARF)
    assert res.details["n_flares"] <= 1


def test_pulsations_are_not_called_flares():
    # sawtooth pulsator (fast rise, slow fade) with a big amplitude, like an RR Lyrae star
    p = 0.55

    def model(t):
        ph = ((t - T_START) / p) % 1
        return 1 + 0.3 * np.where(ph < 0.15, ph / 0.15, (1 - ph) / 0.85)

    coll = build_collection(model, n_sectors=1, cadence_min=2.0, noise_ppm=500)
    res = flare_module.characterize(coll, M_DWARF, {"ls_period_1": p})
    assert not res.details["is_flare_star"]


def test_coarse_cadence_is_not_checked():
    coll = build_collection(lambda t: np.ones_like(t), n_sectors=1, cadence_min=30.0)
    res = flare_module.characterize(coll, M_DWARF)
    assert res.details["checked"] is False


def test_tess_band_luminosity_is_sensible():
    sun = flare_module.tess_band_luminosity(5772, 1.0)
    assert 0.2 * 3.828e33 < sun < 0.5 * 3.828e33        # a fair share of the Sun's output
    assert flare_module.tess_band_luminosity(3300, 0.3) < 0.01 * sun
    assert np.isnan(flare_module.tess_band_luminosity(np.nan, 1.0))
