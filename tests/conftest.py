"""Synthetic TESS-like light curves for offline tests.

Every factory returns a ``lightkurve.LightCurveCollection`` with one light curve
per 27-day sector. Each sector has two ~13-day orbits separated by a ~1-day
downlink gap, a few short random gaps, its own raw flux level, and a
``quality`` column with a handful of flagged cadences. Time is in BTJD.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import lightkurve as lk
import numpy as np
import pytest
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SECTOR_LEN_D = 27.4
DOWNLINK_GAP_D = 1.0
SECTOR_GAP_D = 0.6
T_START = 1325.3            # BTJD, near the start of sector 1
QUALITY_FLAG = 32           # "desaturation event", in lightkurve's default bitmask


def pytest_collection_modifyitems(config, items):
    """On CI (env CI=true) network tests are reported as skipped, never run."""
    if os.environ.get("CI", "").lower() != "true":
        return
    skip = pytest.mark.skip(reason="network test: skipped in CI")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


# ── Building blocks ──────────────────────────────────────────

def sector_times(n_sectors: int, cadence_min: float, rng: np.random.Generator,
                 first_sector: int = 1) -> list[tuple[int, np.ndarray]]:
    """Time stamps for consecutive sectors, with downlink and random gaps."""
    dt = cadence_min / 1440.0
    out = []
    for k in range(n_sectors):
        start = T_START + (first_sector - 1 + k) * SECTOR_LEN_D
        stop = start + SECTOR_LEN_D - SECTOR_GAP_D
        t = np.arange(start, stop, dt)
        mid = 0.5 * (start + stop)
        keep = np.abs(t - mid) > DOWNLINK_GAP_D / 2
        for _ in range(3):                       # short gaps (e.g. momentum dumps)
            g0 = rng.uniform(start, stop)
            keep &= ~((t > g0) & (t < g0 + rng.uniform(0.05, 0.3)))
        out.append((first_sector + k, t[keep]))
    return out


def red_noise(n: int, sigma: float, corr_len: int, rng: np.random.Generator) -> np.ndarray:
    """AR(1) correlated noise with standard deviation ``sigma``."""
    if sigma <= 0 or n == 0:
        return np.zeros(n)
    phi = np.exp(-1.0 / max(corr_len, 1))
    e = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def build_collection(model, n_sectors: int = 2, cadence_min: float = 10.0,
                     noise_ppm: float = 300.0, red_ppm: float = 0.0, red_corr: int = 30,
                     n_spikes: int = 0, seed: int = 0, author: str = "SYNTHETIC",
                     first_sector: int = 1) -> lk.LightCurveCollection:
    """Turn a relative-flux model ``model(t) -> array`` into a sector collection."""
    rng = np.random.default_rng(seed)
    lcs = []
    for sector, t in sector_times(n_sectors, cadence_min, rng, first_sector):
        rel = model(t)
        rel = rel + rng.normal(0, noise_ppm * 1e-6, t.size)
        rel = rel + red_noise(t.size, red_ppm * 1e-6, red_corr, rng)
        if n_spikes:
            idx = rng.choice(t.size, n_spikes, replace=False)
            rel[idx] += rng.choice([-1, 1], n_spikes) * rng.uniform(0.01, 0.05, n_spikes)
        level = rng.uniform(5e3, 5e4)            # raw e-/s differs per sector
        quality = np.zeros(t.size, dtype=np.int32)
        quality[rng.choice(t.size, max(t.size // 500, 1), replace=False)] = QUALITY_FLAG
        lc = lk.LightCurve(time=Time(t, format="btjd", scale="tdb"),
                           flux=rel * level, flux_err=np.full(t.size, noise_ppm * 1e-6 * level),
                           quality=quality)
        lc.meta.update({"SECTOR": sector, "AUTHOR": author, "EXPTIME": cadence_min * 60,
                        "CROWDSAP": 1.0, "MISSION": "TESS", "TARGETID": 0})
        lcs.append(lc)
    return lk.LightCurveCollection(lcs)


def trapezoid(t: np.ndarray, period: float, t0: float, depth: float, duration: float,
              ingress_frac: float) -> np.ndarray:
    """Fractional dimming (0..depth) of a trapezoid dip; ingress_frac=0.5 gives a V."""
    dt = np.abs((t - t0 + 0.5 * period) % period - 0.5 * period)
    half = duration / 2
    tau = max(ingress_frac * duration, 1e-6)
    dim = np.where(dt <= half - tau, depth, depth * (half - dt) / tau)
    return np.clip(dim, 0, depth) * (dt <= half)


# ── Public factories ─────────────────────────────────────────

def make_planet_lc(period: float = 3.0, depth: float = 0.005, duration: float = 0.12,
                   n_sectors: int = 2, cadence: float = 10.0, noise: float = 300.0,
                   t0: float | None = None, ingress_frac: float = 0.1,
                   seed: int = 0, **kw) -> lk.LightCurveCollection:
    """U-shaped (flat-bottomed trapezoid) transit. ``cadence`` in minutes, ``noise`` in ppm."""
    t0 = T_START + 0.37 * period if t0 is None else t0

    def model(t):
        return 1.0 - trapezoid(t, period, t0, depth, duration, ingress_frac)

    return build_collection(model, n_sectors, cadence, noise, seed=seed, **kw)


def make_eb_lc(period: float = 2.4, depth1: float = 0.15, depth2: float = 0.05,
               phase2: float = 0.5, odd_even_diff: float = 0.0, duration: float = 0.18,
               ellipsoidal_amp: float = 0.0, reflection_amp: float = 0.0,
               n_sectors: int = 2, cadence: float = 10.0, noise: float = 300.0,
               t0: float | None = None, seed: int = 1, **kw) -> lk.LightCurveCollection:
    """V-shaped primary and a secondary eclipse at ``phase2``.

    ``odd_even_diff`` makes alternate primaries differ by that fraction of depth1.
    Ellipsoidal (cos 4*pi*phase) and reflection (cos 2*pi*phase) terms are optional.
    """
    t0 = T_START + 0.21 * period if t0 is None else t0

    def model(t):
        cycle = np.floor((t - t0) / period + 0.5)
        odd = (cycle % 2).astype(bool)
        d1 = np.where(odd, depth1 * (1 - odd_even_diff / 2), depth1 * (1 + odd_even_diff / 2))
        prim = trapezoid(t, period, t0, 1.0, duration, 0.5) * d1
        sec = trapezoid(t, period, t0 + phase2 * period, depth2, duration, 0.5)
        phase = ((t - t0) / period) % 1.0
        ell = -ellipsoidal_amp * np.cos(4 * np.pi * phase)
        refl = -reflection_amp * np.cos(2 * np.pi * phase)
        return 1.0 - prim - sec + ell + refl

    return build_collection(model, n_sectors, cadence, noise, seed=seed, **kw)


def make_variable_lc(kind: str = "sinusoid", period: float = 1.7, amp: float = 0.01,
                     n_sectors: int = 2, cadence: float = 10.0, noise: float = 300.0,
                     seed: int = 2, **kw) -> lk.LightCurveCollection:
    """Variable star. ``amp`` is the semi-amplitude of the main mode (relative flux).

    - sinusoid: one pure sine
    - multiperiodic: several incommensurate modes (pulsator-like)
    - spotted: rotation with two harmonics, slowly evolving amplitude and phase
    """
    rng = np.random.default_rng(seed + 100)

    if kind == "sinusoid":
        def model(t):
            return 1.0 + amp * np.sin(2 * np.pi * t / period)
    elif kind == "multiperiodic":
        ratios = np.array([1.0, 0.773, 0.618, 1.309])
        amps = amp * np.array([1.0, 0.6, 0.4, 0.25])
        phases = rng.uniform(0, 2 * np.pi, 4)

        def model(t):
            return 1.0 + sum(a * np.sin(2 * np.pi * t / (period * r) + p)
                             for a, r, p in zip(amps, ratios, phases))
    elif kind == "spotted":
        evol = 15.0
        p0 = rng.uniform(0, 2 * np.pi)

        def model(t):
            env = 1 + 0.3 * np.sin(2 * np.pi * t / evol)
            drift = 0.3 * np.sin(2 * np.pi * t / (2.3 * evol))
            ph = 2 * np.pi * t / period + p0 + drift
            return 1.0 + amp * env * (np.sin(ph) + 0.35 * np.sin(2 * ph + 0.8))
    else:
        raise ValueError(f"unknown kind {kind!r}")

    return build_collection(model, n_sectors, cadence, noise, seed=seed, **kw)


def make_noise_lc(n_sectors: int = 2, cadence: float = 10.0, noise: float = 400.0,
                  red: float = 300.0, red_corr: int = 40, n_spikes: int = 20,
                  seed: int = 3, **kw) -> lk.LightCurveCollection:
    """No astrophysical signal: white + red noise, gaps and outlier spikes."""
    return build_collection(lambda t: np.ones_like(t), n_sectors, cadence, noise,
                            red_ppm=red, red_corr=red_corr, n_spikes=n_spikes,
                            seed=seed, **kw)


# ── Fixtures exposing the factories ──────────────────────────

@pytest.fixture
def planet_factory():
    return make_planet_lc


@pytest.fixture
def eb_factory():
    return make_eb_lc


@pytest.fixture
def variable_factory():
    return make_variable_lc


@pytest.fixture
def noise_factory():
    return make_noise_lc
