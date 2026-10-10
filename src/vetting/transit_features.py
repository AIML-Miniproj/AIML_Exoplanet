"""Features describing the dip itself: depth, shape, odd/even and secondary eclipses."""

from __future__ import annotations

import logging
from typing import Any

import lightkurve as lk
import numpy as np
from scipy.optimize import least_squares

from src import physics
from src.detect.bls import Signal
from src.vetting.common import (NAN, bin_xy, box_depth, per_transit_depths, phase,
                                phase_time, red_noise_factor, safe_ratio, signed_log10)

log = logging.getLogger(__name__)

R_2JUP_REARTH = 2 * physics.R_JUP_M / physics.R_EARTH_M


def trapezoid_model(x: np.ndarray, depth: float, t14: float, frac: float,
                    tc: float) -> np.ndarray:
    """Relative flux of a trapezoid dip. ``frac`` = ingress time / total duration."""
    dt = np.abs(x - tc)
    half = t14 / 2
    tau = max(frac * t14, 1e-6)
    dim = np.where(dt <= half - tau, depth, depth * (half - dt) / tau)
    return 1.0 - np.clip(dim, 0, depth) * (dt <= half)


def fit_trapezoid(pt: np.ndarray, flux: np.ndarray, duration: float, depth0: float,
                  cadence: float) -> dict[str, Any]:
    """Fit trapezoid and box shapes to the folded dip; compare their chi-squared."""
    out = {"ok": False, "frac": NAN, "t14": NAN, "depth": NAN, "chi2_trap": NAN,
           "chi2_box": NAN, "x": [], "y": [], "model": []}
    win = 1.5 * duration
    sel = np.abs(pt) < win
    width = max(duration / 20, cadence)
    edges = np.arange(-win, win + width, width)
    x, y, e, n = bin_xy(pt[sel], flux[sel], edges)
    ok = (n >= 2) & np.isfinite(y) & (e > 0)
    if ok.sum() < 8 or not np.isfinite(depth0) or depth0 <= 0:
        return out
    x, y, e = x[ok], y[ok], np.maximum(e[ok], np.median(e[ok]) * 0.3)

    def resid(p, frac_fixed=None):
        d, t14, frac, tc = p if frac_fixed is None else (p[0], p[1], frac_fixed, p[2])
        return (y - trapezoid_model(x, d, t14, frac, tc)) / e

    lo_t, hi_t = 0.3 * duration, 2.8 * duration
    t14_0 = float(np.clip(duration, lo_t * 1.01, hi_t * 0.99))
    try:
        r = least_squares(resid, [depth0, t14_0, 0.15, 0.0],
                          bounds=([0, lo_t, 0.001, -duration / 2],
                                  [1, hi_t, 0.5, duration / 2]))
        rb = least_squares(lambda p: resid(p, 0.001), [depth0, t14_0, 0.0],
                           bounds=([0, lo_t, -duration / 2], [1, hi_t, duration / 2]))
    except ValueError as err:
        log.debug("trapezoid fit failed: %s", err)
        return out
    d, t14, frac, tc = r.x
    out.update(ok=True, frac=float(frac), t14=float(t14), depth=float(d), tc=float(tc),
               chi2_trap=float(np.sum(r.fun**2)), chi2_box=float(np.sum(rb.fun**2)),
               x=x.tolist(), y=y.tolist(),
               model=trapezoid_model(x, d, t14, frac, tc).tolist())
    return out


def secondary_search(t: np.ndarray, flux: np.ndarray, period: float, t0: float,
                     duration: float, beta: float = 1.0) -> dict[str, float]:
    """Depth and significance of a second dip at phase 0.5 and at the best phase 0.3-0.7."""
    ph = phase(t, period, t0)
    dphase = duration / period
    primary = np.abs(phase_time(t, period, t0)) < duration

    def at(p0: float) -> tuple[float, float]:
        d_ph = np.abs((ph - p0 + 0.5) % 1 - 0.5)
        inside = d_ph < dphase / 2
        out_m = ~primary & (d_ph > dphase)
        depth, err, _ = box_depth(flux, inside, out_m, beta)
        return depth, err

    d05, e05 = at(0.5)
    best = (NAN, NAN, NAN)
    step = max(dphase / 4, 0.002)
    for p0 in np.arange(0.3, 0.7 + 1e-9, step):
        d, e = at(p0)
        snr = d / e if e and np.isfinite(e) and e > 0 else NAN
        if np.isfinite(snr) and (not np.isfinite(best[1]) or snr > best[1]):
            best = (float(p0), float(snr), float(d))
    return {
        "secondary_depth_ppm": d05 * 1e6 if np.isfinite(d05) else NAN,
        "secondary_snr": safe_ratio(d05, e05),
        "secondary_best_phase": best[0],
        "secondary_best_snr": best[1],
        "secondary_best_depth_ppm": best[2] * 1e6 if np.isfinite(best[2]) else NAN,
    }


def _odd_even(stats: dict[str, Any], beta: float = 1.0) -> tuple[float, float]:
    """Odd/even depth difference from the BLS statistics, errors inflated by ``beta``."""
    do, eo = stats.get("depth_odd", [NAN, NAN])
    de, ee = stats.get("depth_even", [NAN, NAN])
    err = beta * np.hypot(eo, ee)
    sig = abs(do - de) / err if err > 0 else NAN
    hi = max(abs(do), abs(de))
    ratio = min(do, de) / hi if hi > 0 else NAN
    return float(sig), float(ratio)


def odd_even_from_transits(t: np.ndarray, flux: np.ndarray, period: float, t0: float,
                           duration: float) -> tuple[float, float] | None:
    """Odd/even comparison using the scatter of individual transit depths.

    This accounts for correlated noise and transit-to-transit systematics, which the
    white-noise BLS errors ignore. Needs at least two odd and two even transits.
    """
    epochs, depths = per_transit_depths(t, flux, period, t0, duration)
    odd, even = depths[epochs % 2 == 1], depths[epochs % 2 == 0]
    if odd.size < 2 or even.size < 2:
        return None
    err = np.hypot(np.std(odd, ddof=1) / np.sqrt(odd.size),
                   np.std(even, ddof=1) / np.sqrt(even.size))
    mo, me = float(np.mean(odd)), float(np.mean(even))
    hi = max(abs(mo), abs(me))
    sig = abs(mo - me) / err if err > 0 else NAN
    return float(sig), float(min(mo, me) / hi) if hi > 0 else NAN


def transit_features(lc_flat: lk.LightCurve, signal: Signal, star=None,
                     search_cadence_d: float | None = None,
                     ) -> tuple[dict[str, float], dict[str, Any]]:
    """Return (features, details). Details hold arrays for plots."""
    t = np.asarray(lc_flat.time.value, float)
    f = np.asarray(lc_flat.flux.value, float)
    P, t0, dur = signal.period, signal.t0, signal.duration
    pt = phase_time(t, P, t0)
    cadence = float(np.median(np.diff(t))) if t.size > 1 else 0.0014

    feats: dict[str, float] = {
        "period_d": P,
        "duration_hr": dur * 24,
        "depth_ppm": signal.depth * 1e6,
        "depth_snr": signal.depth_snr,
        "sde": signal.sde,
        "single_event_fraction": signal.single_event_fraction,
    }

    out_of_dip = np.abs(pt) > 1.5 * dur
    beta = red_noise_factor(t, f, dur, out_of_dip)
    feats["red_noise_beta"] = beta

    counts = np.asarray(signal.stats.get("per_transit_count", []), float)
    expected = dur / (search_cadence_d or cadence)
    feats["n_transits_observed"] = float(np.sum(counts >= max(2, 0.5 * expected)))

    oe = odd_even_from_transits(t, f, P, t0, dur)
    feats["odd_even_depth_sigma"], feats["odd_even_depth_ratio"] = (
        oe if oe is not None else _odd_even(signal.stats, beta))
    if signal.stats_2p:
        feats["odd_even_depth_sigma_2p"], _ = _odd_even(signal.stats_2p, beta)
    else:
        feats["odd_even_depth_sigma_2p"] = NAN

    feats.update(secondary_search(t, f, P, t0, dur, beta))

    trap = fit_trapezoid(pt, f, dur, signal.depth, cadence)
    feats["ingress_to_total_ratio"] = trap["frac"]
    feats["t14_hr"] = trap["t14"] * 24 if trap["ok"] else NAN
    feats["trapezoid_chi2_vs_box_chi2"] = safe_ratio(trap["chi2_trap"], trap["chi2_box"])

    # Asymmetry: compare the first and second halves of the dip.
    half = (trap["t14"] if trap["ok"] else dur) / 2
    left, right = (pt > -half) & (pt < 0), (pt > 0) & (pt < half)
    out = (np.abs(pt) > 1.5 * dur) & (np.abs(pt) < 4 * dur)
    if left.sum() > 2 and right.sum() > 2 and out.sum() > 5 and signal.depth > 0:
        base = np.median(f[out])
        feats["transit_asymmetry"] = float(((base - f[left].mean()) - (base - f[right].mean()))
                                           / signal.depth)
    else:
        feats["transit_asymmetry"] = NAN

    feats["harmonic_dll"] = signed_log10(signal.stats.get("harmonic_delta_log_likelihood", NAN))

    rad = getattr(star, "rad", NAN) if star is not None else NAN
    rho = getattr(star, "rho", NAN) if star is not None else NAN
    if np.isfinite(rad) and signal.depth > 0:
        rp = physics.planet_radius_rearth(signal.depth, rad)
        feats["implied_rp_rearth"] = rp
        feats["implied_rp_gt_2rjup"] = float(rp > R_2JUP_REARTH)
    else:
        feats["implied_rp_rearth"] = NAN
        feats["implied_rp_gt_2rjup"] = NAN
    obs_dur = trap["t14"] if trap["ok"] else dur
    feats["duration_ratio_expected"] = safe_ratio(obs_dur, physics.expected_duration_d(P, rho))

    details = {"trapezoid": {k: trap[k] for k in ("x", "y", "model", "frac", "t14", "depth")},
               "red_noise_beta": beta}
    return feats, details
