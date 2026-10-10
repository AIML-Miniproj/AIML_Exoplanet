"""Periodic-dip search with Box Least Squares (BLS).

The search slides a box-shaped dip across the light curve for many trial
periods and durations, and scores how much better a dip fits than a flat line.

Grid: the period grid is uniform in frequency (``autoperiod``). With sectors
years apart, a full-resolution grid has tens of millions of periods, so the
search runs on a coarser grid capped at ``cfg.max_grid_size`` and then refines
the strongest peaks at full resolution in narrow windows.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import lightkurve as lk
import numpy as np
from astropy.timeseries import BoxLeastSquares

from config import CONFIG, Config

log = logging.getLogger(__name__)

ALIAS_RATIOS = (1.0, 0.5, 2.0, 1 / 3, 3.0, 2 / 3, 1.5)


@dataclass
class Peak:
    period: float
    power: float
    t0: float
    duration: float
    depth: float
    grid_power: float = float("nan")   # power on the search grid (used for SDE)


@dataclass
class Signal:
    """The best periodic dip found, with the statistics used for vetting."""

    period: float
    t0: float
    duration: float
    depth: float
    depth_err: float
    power: float
    sde: float
    stats: dict[str, Any] = field(default_factory=dict)
    stats_2p: dict[str, Any] = field(default_factory=dict)
    top_peaks: list[Peak] = field(default_factory=list)
    method: str = "bls"

    @property
    def depth_snr(self) -> float:
        return self.depth / self.depth_err if self.depth_err > 0 else 0.0

    @property
    def ephemeris(self) -> tuple[float, float, float]:
        return self.period, self.t0, self.duration

    @property
    def single_event_fraction(self) -> float:
        """Share of the total evidence that comes from the single strongest transit.

        Near 1 means one isolated dip (or glitch) drives the detection, which is not
        a repeating signal.
        """
        ll = np.asarray(self.stats.get("per_transit_log_likelihood", []), float)
        ll = ll[np.isfinite(ll) & (ll > 0)]
        return float(ll.max() / ll.sum()) if ll.size else float("nan")

    def detected(self, cfg: Config = CONFIG) -> bool:
        sef = self.single_event_fraction
        return (self.sde >= cfg.sde_min and self.depth > 0
                and not (np.isfinite(sef) and sef > cfg.max_single_event_fraction))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["depth_ppm"] = self.depth * 1e6
        d["depth_snr"] = self.depth_snr
        d["single_event_fraction"] = self.single_event_fraction
        return d


@dataclass
class SearchResult:
    periods: np.ndarray
    power: np.ndarray
    signal: Signal
    grid_size_full: int
    grid_size_used: int


def _arrays(lc: lk.LightCurve) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(lc.time.value, float)
    f = np.asarray(lc.flux.value, float)
    e = np.asarray(lc.flux_err.value, float) if lc.flux_err is not None else None
    ok = np.isfinite(t) & np.isfinite(f)
    if e is not None:
        ok &= np.isfinite(e) & (e > 0)
        return t[ok], f[ok], e[ok]
    return t[ok], f[ok], np.full(ok.sum(), np.std(f[ok]))


def period_grid(model: BoxLeastSquares, durations: np.ndarray, pmin: float, pmax: float,
                cfg: Config = CONFIG) -> tuple[np.ndarray, int]:
    """Frequency-uniform grid, coarsened to at most ``cfg.max_grid_size`` periods."""
    full = model.autoperiod(durations, minimum_period=pmin, maximum_period=pmax,
                            frequency_factor=cfg.frequency_factor)
    n_full = len(full)
    if n_full <= cfg.max_grid_size:
        return np.asarray(full), n_full
    log.warning("Full period grid has %d periods; searching %d and refining the best peaks",
                n_full, cfg.max_grid_size)
    freqs = np.linspace(1 / pmax, 1 / pmin, cfg.max_grid_size)
    return np.sort(1 / freqs), n_full


def _is_alias(p: float, accepted: list[float], tol: float = 0.01) -> bool:
    return any(abs(p / (q * r) - 1) < tol for q in accepted for r in ALIAS_RATIOS)


def distinct_peaks(periods: np.ndarray, power: np.ndarray, n: int,
                   tol: float = 0.01) -> list[int]:
    """Indices of the ``n`` strongest local maxima, skipping harmonics of stronger ones."""
    if len(power) < 3:
        return [int(np.argmax(power))]
    is_max = np.r_[power[0] > power[1],
                   (power[1:-1] >= power[:-2]) & (power[1:-1] >= power[2:]),
                   power[-1] > power[-2]]
    order = np.flatnonzero(is_max)[np.argsort(power[is_max])[::-1]]
    chosen: list[int] = []
    for i in order:
        if not _is_alias(periods[i], [periods[j] for j in chosen], tol):
            chosen.append(int(i))
        if len(chosen) >= n:
            break
    return chosen


def _refine(model: BoxLeastSquares, durations: np.ndarray, period: float,
            coarse_df: float, fine_df: float, pmin: float, pmax: float):
    """Re-search a narrow frequency window around ``period`` at full resolution."""
    f0 = 1 / period
    half = 3 * coarse_df
    n = int(min(max(2 * half / fine_df, 50), 8000))
    freqs = np.linspace(max(f0 - half, 1 / pmax), min(f0 + half, 1 / pmin), n)
    res = model.power(np.sort(1 / freqs), durations, objective="likelihood")
    return res


def _stats(model: BoxLeastSquares, period: float, duration: float, t0: float) -> dict[str, Any]:
    s = model.compute_stats(period, duration, t0)
    out = {}
    for k in ("depth", "depth_odd", "depth_even", "depth_half", "depth_phased"):
        out[k] = [float(s[k][0]), float(s[k][1])]
    out["harmonic_amplitude"] = float(s["harmonic_amplitude"])
    out["harmonic_delta_log_likelihood"] = float(s["harmonic_delta_log_likelihood"])
    out["transit_times"] = np.asarray(s["transit_times"], float).tolist()
    out["per_transit_count"] = np.asarray(s["per_transit_count"], int).tolist()
    out["per_transit_log_likelihood"] = np.asarray(s["per_transit_log_likelihood"],
                                                    float).tolist()
    return out


def search(lc: lk.LightCurve, cfg: Config = CONFIG, period_min: float | None = None,
           period_max: float | None = None) -> SearchResult:
    """Run BLS on ``lc`` and return the periodogram and the best signal."""
    t, f, e = _arrays(lc)
    if t.size < 50:
        raise ValueError(f"only {t.size} points to search")
    baseline = float(t.max() - t.min())
    durations = cfg.search_durations()
    pmin = period_min or cfg.period_min_d
    pmax = min(period_max or cfg.period_max_d, baseline / 2)
    if pmax <= pmin:
        raise ValueError(f"baseline {baseline:.1f} d too short for periods >= {pmin} d")

    model = BoxLeastSquares(t, f, dy=e)
    periods, n_full = period_grid(model, durations, pmin, pmax, cfg)
    res = model.power(periods, durations, objective="likelihood")
    power = np.asarray(res.power, float)
    finite = np.isfinite(power)
    # SDE is measured on sqrt(power) (the "signal residue"). The likelihood power
    # itself has an exponential tail, so its largest value among tens of thousands
    # of trial periods is ~10 standard deviations high even for pure noise.
    sr = np.sqrt(np.clip(power[finite], 0, None))
    sde_mu, sde_sd = float(np.mean(sr)), float(np.std(sr))

    peak_idx = distinct_peaks(periods, np.where(finite, power, -np.inf),
                              max(cfg.n_top_peaks, 3))
    coarse = n_full > len(periods)
    fine_df = cfg.frequency_factor * durations.min() / baseline**2
    coarse_df = (1 / pmin - 1 / pmax) / max(len(periods) - 1, 1)

    peaks: list[Peak] = []
    for rank, i in enumerate(peak_idx):
        grid_pw = float(power[i])
        p, pw, t0, dur, dep = (periods[i], power[i], res.transit_time[i],
                               res.duration[i], res.depth[i])
        if coarse and rank < cfg.n_refine_peaks:
            r2 = _refine(model, durations, p, coarse_df, fine_df, pmin, pmax)
            j = int(np.nanargmax(r2.power))
            p, pw, t0, dur, dep = (r2.period[j], r2.power[j], r2.transit_time[j],
                                   r2.duration[j], r2.depth[j])
        peaks.append(Peak(float(p), float(pw), float(t0), float(dur), float(dep), grid_pw))
    peaks.sort(key=lambda pk: pk.power, reverse=True)

    best = peaks[0]
    stats = _stats(model, best.period, best.duration, best.t0)
    stats_2p = _stats(model, 2 * best.period, best.duration, best.t0) \
        if 2 * best.period <= baseline else {}
    depth, depth_err = stats["depth"]
    # SDE compares the peak with the rest of the *same* grid; refined powers are
    # systematically higher, so they are not used here.
    sig = Signal(period=best.period, t0=best.t0, duration=best.duration, depth=depth,
                 depth_err=depth_err, power=best.power,
                 sde=(np.sqrt(max(best.grid_power, 0)) - sde_mu) / sde_sd if sde_sd > 0 else 0.0,
                 stats=stats, stats_2p=stats_2p, top_peaks=peaks)
    log.info("BLS: P=%.5f d, depth=%.0f ppm, duration=%.2f h, SDE=%.1f (%d periods%s)",
             sig.period, depth * 1e6, sig.duration * 24, sig.sde, len(periods),
             ", refined" if coarse else "")
    return SearchResult(periods=periods, power=power, signal=sig,
                        grid_size_full=n_full, grid_size_used=len(periods))


def evaluate(lc: lk.LightCurve, signal: Signal, cfg: Config = CONFIG,
             window_frac: float = 0.002) -> Signal:
    """Re-fit a known signal on a (re-detrended) light curve with a narrow local search."""
    t, f, e = _arrays(lc)
    model = BoxLeastSquares(t, f, dy=e)
    durations = cfg.search_durations()
    p = signal.period
    periods = np.linspace(p * (1 - window_frac), p * (1 + window_frac), 400)
    res = model.power(periods, durations, objective="likelihood")
    j = int(np.nanargmax(res.power))
    period, t0, dur = float(res.period[j]), float(res.transit_time[j]), float(res.duration[j])
    stats = _stats(model, period, dur, t0)
    baseline = float(t.max() - t.min())
    stats_2p = _stats(model, 2 * period, dur, t0) if 2 * period <= baseline else {}
    return Signal(period=period, t0=t0, duration=dur, depth=stats["depth"][0],
                  depth_err=stats["depth"][1], power=float(res.power[j]), sde=signal.sde,
                  stats=stats, stats_2p=stats_2p, top_peaks=signal.top_peaks,
                  method=signal.method)


def mask_signal(lc: lk.LightCurve, signal: Signal, factor: float = 2.0) -> lk.LightCurve:
    """Remove the in-transit points of ``signal`` (for multi-signal search)."""
    t = lc.time.value
    dt = np.abs((t - signal.t0 + 0.5 * signal.period) % signal.period - 0.5 * signal.period)
    return lc[dt > factor * signal.duration / 2]


def multi_search(lc: lk.LightCurve, cfg: Config = CONFIG,
                 first: SearchResult | None = None) -> list[SearchResult]:
    """Find up to ``cfg.max_signals`` signals, masking each one before the next search."""
    results = [first or search(lc, cfg)]
    current = lc
    while len(results) < cfg.max_signals:
        current = mask_signal(current, results[-1].signal)
        try:
            nxt = search(current, cfg)
        except ValueError:
            break
        if not nxt.signal.detected(cfg):
            break
        results.append(nxt)
    return results
