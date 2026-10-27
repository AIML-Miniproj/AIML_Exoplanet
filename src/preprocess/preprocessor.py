"""Clean, normalise, stitch and detrend multi-sector light curves.

Two products come out:
- ``lc_norm``: every sector divided by its own median and stitched. Stellar
  variability is kept; the variable-star features use this.
- ``lc_flat``: ``lc_norm`` with slow trends removed sector by sector (sliding
  biweight). Only short dips survive; the periodic-dip search uses this.

Outliers are removed on the *upper* side only, so dips and eclipses are never
clipped, however deep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

import lightkurve as lk
import numpy as np
from astropy.time import Time
from lightkurve.utils import TessQualityFlags
from scipy.ndimage import median_filter

from config import CONFIG, Config

log = logging.getLogger(__name__)


@dataclass
class Preprocessed:
    lc_norm: lk.LightCurve
    lc_flat: lk.LightCurve
    trend: np.ndarray
    window_d: float
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def sectors(self) -> np.ndarray:
        return np.unique(self.lc_norm["sector"].value)


def _make_lc(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
             sector: np.ndarray) -> lk.LightCurve:
    lc = lk.LightCurve(time=Time(time, format="btjd", scale="tdb"), flux=flux,
                       flux_err=flux_err)
    lc["sector"] = sector.astype(np.int32)
    return lc


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def robust_std(x: np.ndarray) -> float:
    """Standard deviation estimated from the median absolute deviation."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def upper_outlier_mask(time: np.ndarray, flux: np.ndarray, sigma: float,
                       window_d: float, long_window_d: float = 0.5) -> np.ndarray:
    """True for points that sit far *above* both a short and a long running median.

    The short median makes spikes stand out on top of stellar variability; the
    long one stops the flat points just outside a deep, sharp eclipse (where the
    short median already dips) from being mistaken for spikes.
    """
    if time.size < 5:
        return np.zeros(flux.size, bool)
    cadence = max(float(np.median(np.diff(time))), 1e-6)
    out = np.ones(flux.size, bool)
    # Both windows must agree ('&='): a point is clipped only if it is far above the
    # short AND the long running median. Points below the median are never clipped.
    for w in (window_d, long_window_d):
        n = _odd(max(5, int(round(w / cadence))))
        resid = flux - median_filter(flux, size=n, mode="nearest")
        s = robust_std(resid)
        if not np.isfinite(s) or s == 0:
            return np.zeros(flux.size, bool)
        out &= resid > sigma * s
    return out


def clean_sector(lc: lk.LightCurve, cfg: Config = CONFIG) -> tuple[dict[str, np.ndarray], dict]:
    """Quality mask, NaN removal, per-sector median normalisation, upper-only clip."""
    sector = int(lc.meta.get("SECTOR", -1))
    t = np.asarray(lc.time.value, float)
    f = np.asarray(lc.flux.value, float)
    e = np.asarray(lc.flux_err.value, float) if lc.flux_err is not None else np.full_like(f, np.nan)

    # 1) Drop missing values and readings NASA flagged as bad (momentum dumps,
    # scattered light, ...).
    good = np.isfinite(t) & np.isfinite(f)
    n_quality = 0
    if "quality" in lc.colnames:
        q = np.asarray(lc["quality"].value, dtype=np.int64)
        bad_q = (q & TessQualityFlags.DEFAULT_BITMASK) != 0
        n_quality = int(np.sum(bad_q & good))
        good &= ~bad_q
    t, f, e = t[good], f[good], e[good]
    order = np.argsort(t)
    t, f, e = t[order], f[order], e[order]

    # 2) Divide by the sector's median so every sector sits at 1.0 (raw flux levels
    # differ between sectors because the aperture and pointing differ).
    med = np.nanmedian(f) if f.size else np.nan
    if not np.isfinite(med) or med <= 0:
        return ({"time": t[:0], "flux": f[:0], "flux_err": e[:0]},
                {"sector": sector, "skipped": True})
    f, e = f / med, e / med

    # 3) Clip upward spikes only (cosmic rays, glitches). Flares are clipped too; the
    # flare module looks for them in the raw sectors instead.
    clip = upper_outlier_mask(t, f, cfg.clip_sigma_upper, cfg.clip_window_d)
    crowd = lc.meta.get("CROWDSAP")
    stats = {"sector": sector, "author": lc.meta.get("AUTHOR"),
             "n_in": int(lc.flux.size), "n_quality": n_quality,
             "n_clipped": int(clip.sum()), "n_out": int((~clip).sum()),
             "crowdsap": float(crowd) if crowd is not None else None}
    # Missing errors: estimate the noise from the point-to-point scatter
    # (std of successive differences / sqrt(2)).
    if not np.all(np.isfinite(e)):
        e = np.where(np.isfinite(e), e, robust_std(np.diff(f)) / np.sqrt(2))
    return {"time": t[~clip], "flux": f[~clip], "flux_err": e[~clip]}, stats


def detrend(time: np.ndarray, flux: np.ndarray, window_d: float, cfg: Config = CONFIG,
            mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sliding robust trend for one sector; returns (flat, trend).

    Points where ``mask`` is True are left out of the trend fit, and the trend is
    interpolated across them (used to protect transits on the second pass).
    """
    from wotan import flatten

    # Biweight (wotan): a robust sliding-window average that ignores outliers, so a
    # short dip barely pulls the trend down. Result: flat = flux / trend.
    fit = np.ones(time.size, bool) if mask is None else ~mask
    if fit.sum() < 10:
        return np.full_like(flux, np.nan), np.full_like(flux, np.nan)
    _, trend_fit = flatten(time[fit], flux[fit], method=cfg.detrend_method,
                           window_length=window_d, break_tolerance=cfg.break_tolerance_d,
                           return_trend=True)
    ok = np.isfinite(trend_fit)
    if ok.sum() < 2:
        return np.full_like(flux, np.nan), np.full_like(flux, np.nan)
    trend = np.interp(time, time[fit][ok], trend_fit[ok])
    if mask is None:
        trend = np.where(np.isfinite(trend_fit), trend_fit, trend)
    return flux / trend, trend


def _flatten_all(lc_norm: lk.LightCurve, window_d: float, cfg: Config,
                 mask: np.ndarray | None = None) -> tuple[lk.LightCurve, np.ndarray]:
    t = lc_norm.time.value
    f = lc_norm.flux.value
    e = lc_norm.flux_err.value
    sec = lc_norm["sector"].value
    flat = np.full_like(f, np.nan)
    trend = np.full_like(f, np.nan)
    for s in np.unique(sec):
        m = sec == s
        flat[m], trend[m] = detrend(t[m], f[m], window_d, cfg,
                                    None if mask is None else mask[m])
    ok = np.isfinite(flat)
    lc_flat = _make_lc(t[ok], flat[ok], e[ok] / trend[ok], sec[ok])
    return lc_flat, trend


def preprocess(collection: lk.LightCurveCollection | list[lk.LightCurve],
               cfg: Config = CONFIG) -> Preprocessed:
    """Clean every sector, stitch into ``lc_norm`` and detrend into ``lc_flat``."""
    parts, stats = [], []
    for lc in collection:
        arr, st = clean_sector(lc, cfg)
        stats.append(st)
        if arr["time"].size:
            arr["sector"] = np.full(arr["time"].size, st["sector"])
            parts.append(arr)
    if not parts:
        from src.errors import AnalysisError
        raise AnalysisError("no usable cadences after cleaning")

    # Stitch: join all sectors and sort by time.
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(cat["time"])
    lc_norm = _make_lc(cat["time"][order], cat["flux"][order], cat["flux_err"][order],
                       cat["sector"][order])

    # Detrend sector by sector (each has its own systematics) with a window at least
    # 3x the longest searched dip, so the dips themselves survive.
    window = cfg.detrend_window()
    lc_flat, trend = _flatten_all(lc_norm, window, cfg)
    summary = {
        "sectors": stats,
        "n_cadences": len(lc_norm),
        "n_quality_removed": sum(s.get("n_quality", 0) for s in stats),
        "n_clipped": sum(s.get("n_clipped", 0) for s in stats),
        "detrend_window_d": window,
    }
    log.info("Preprocessed %d sectors: %d cadences, %d quality-flagged, %d upper outliers, "
             "detrend window %.2f d", len(parts), len(lc_norm), summary["n_quality_removed"],
             summary["n_clipped"], window)
    return Preprocessed(lc_norm=lc_norm, lc_flat=lc_flat, trend=trend, window_d=window,
                        stats=summary)


def transit_mask(time: np.ndarray, period: float, t0: float, duration: float,
                 factor: float = 1.5) -> np.ndarray:
    """True within ``factor * duration / 2`` of every predicted mid-transit time."""
    phase_t = np.abs((time - t0 + 0.5 * period) % period - 0.5 * period)
    return phase_t < factor * duration / 2


def redetrend_masked(pre: Preprocessed, ephemerides: list[tuple[float, float, float]],
                     cfg: Config = CONFIG, factor: float = 1.5) -> Preprocessed:
    """Second detrending pass with the detected dips left out of the trend fit.

    Safety net: if the dips would hide more than ``cfg.max_masked_fraction`` of
    the points, the trend cannot be fitted on what is left, so the first pass is
    kept unchanged.
    """
    t = pre.lc_norm.time.value
    mask = np.zeros(t.size, bool)
    for period, t0, duration in ephemerides:
        mask |= transit_mask(t, period, t0, duration, factor)
    if mask.mean() > cfg.max_masked_fraction:
        log.warning("Masked re-detrend skipped: the dips would hide %.0f%% of the points",
                    100 * mask.mean())
        return replace(pre, stats=dict(pre.stats, redetrended=False,
                                       redetrend_skipped_masked_frac=float(mask.mean())))
    lc_flat, trend = _flatten_all(pre.lc_norm, pre.window_d, cfg, mask)
    stats = dict(pre.stats, redetrended=True, n_masked=int(mask.sum()))
    return replace(pre, lc_flat=lc_flat, trend=trend, stats=stats)


def bin_lightcurve(lc: lk.LightCurve, minutes: float | None) -> lk.LightCurve:
    """Bin to ``minutes`` for the search step only. No-op if already that coarse."""
    if not minutes or len(lc) < 2:
        return lc
    cadence_min = np.median(np.diff(lc.time.value)) * 1440
    if cadence_min >= 0.9 * minutes:
        return lc
    t = lc.time.value
    f = lc.flux.value
    e = lc.flux_err.value
    width = minutes / 1440.0
    # Put each point in a time bin and average per bin with bincount (O(N));
    # errors combine in quadrature.
    idx = np.floor((t - t[0]) / width).astype(np.int64)
    uniq, inv, counts = np.unique(idx, return_inverse=True, return_counts=True)
    tb = np.bincount(inv, t) / counts
    fb = np.bincount(inv, f) / counts
    eb = np.sqrt(np.bincount(inv, e**2)) / counts
    sec = lc["sector"].value[np.searchsorted(idx, uniq)] if "sector" in lc.colnames else \
        np.zeros(uniq.size)
    return _make_lc(tb, fb, eb, sec)


PREWHITEN_MIN_PERIOD_D = 1 / 48          # 30 minutes


def prewhiten(lc: lk.LightCurve, cfg: Config = CONFIG) -> tuple[lk.LightCurve, list[dict]]:
    """Subtract up to ``cfg.prewhiten_max_terms`` strong, fast sinusoids (dip search only).

    Some stars vary on roughly the same timescale as the detrending window
    (about a day). The window can't follow that, so the leftover wiggles can
    outscore a shallow transit in the dip search. Shortening the window would
    also start to eat real transits, so instead the strongest sinusoids are
    measured and subtracted, one at a time:

    - only periods shorter than ``prewhiten_max_period_factor`` x the detrend window,
    - only while one sinusoid explains at least ``prewhiten_min_power`` of the
      variance (Lomb-Scargle power),
    - fitted without the lowest points (more than 3 sigma below the median), so a
      transit or eclipse does not steer the fit.

    Returns the cleaned copy and what was removed. The input is not changed; all
    depths and vetting features are measured on the original light curve.
    """
    from astropy.timeseries import LombScargle

    if not cfg.prewhiten or len(lc) < 100:
        return lc, []
    t = np.asarray(lc.time.value, float)
    f = np.asarray(lc.flux.value, float)
    ok = np.isfinite(t) & np.isfinite(f)
    resid = f - np.nanmedian(f[ok])
    pmax = cfg.prewhiten_max_period_factor * cfg.detrend_window()
    baseline = float(t[ok].max() - t[ok].min())
    if pmax <= PREWHITEN_MIN_PERIOD_D or baseline < 2 * pmax:
        return lc, []
    freq = np.linspace(1 / pmax, 1 / PREWHITEN_MIN_PERIOD_D,
                       min(int(5 * baseline / PREWHITEN_MIN_PERIOD_D), 200_000))
    removed: list[dict] = []
    for _ in range(cfg.prewhiten_max_terms):
        sd = 1.4826 * np.nanmedian(np.abs(resid[ok] - np.nanmedian(resid[ok])))
        fit = ok & (resid > np.nanmedian(resid[ok]) - 3 * sd)
        ls = LombScargle(t[fit], resid[fit])
        power = ls.power(freq, method="fast")
        k = int(np.nanargmax(power))
        if power[k] < cfg.prewhiten_min_power:
            break
        model = ls.model(t, freq[k])
        resid = resid - (model - ls.offset())
        amp = float(np.ptp(model) / 2)
        removed.append({"period_d": float(1 / freq[k]), "power": float(power[k]),
                        "amplitude_ppm": amp * 1e6})
    if not removed:
        return lc, []
    log.info("Prewhitened %d sinusoid(s) before the dip search: %s", len(removed),
             ", ".join(f"P={r['period_d']:.3f} d ({r['amplitude_ppm']:.0f} ppm)"
                       for r in removed))
    out = lc.copy()
    out.flux = (resid + np.nanmedian(f[ok])) * lc.flux.unit
    return out, removed
