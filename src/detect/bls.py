"""Periodic-dip search with Box Least Squares (BLS).

The search slides a box-shaped dip across the light curve for many trial
periods and durations, and scores how much better a dip fits than a flat line.

Grid: the period grid is uniform in frequency, with the same spacing as
``autoperiod``. With sectors years apart, a full-resolution grid has hundreds of
millions of periods, so only its size is computed. The search runs on a coarser
grid capped at ``cfg.max_grid_size`` and then refines the strongest peaks at full
resolution in narrow windows.
"""

# ── In plain words ───────────────────────────────────────────────────────────
# STEP 4 OF THE PIPELINE: find the repeating dip.
# Idea: guess a period (say 3.52 days), stack ("fold") every 3.52-day chunk of the
#   light curve on top of each other, and check whether a short box-shaped dip lines
#   up. Repeat for tens of thousands of guesses and many dip lengths; the guess where
#   the dip lines up best wins. That is Box Least Squares (BLS), the standard transit
#   search (Kovacs et al. 2002); astropy does the number crunching.
# Input:  the cleaned, flattened light curve (lc_flat), binned to 10 minutes.
# Output: a Signal - period, time of the first dip (t0), dip length, depth, how
#   clearly it stands out (SDE), and statistics used later by the vetting features.
# Key words:
#   period   time between dips, in days
#   t0       time of one dip's middle (BTJD days)
#   duration how long one dip lasts, in days
#   depth    how much light is lost (0.01 = 1%)
#   power    how much better a dip fits than a flat line, at one trial period
#   SDE      how far the best power stands out from all the others (>= 7 = detected)
# Speed: about (number of trial periods) x (number of points); the grid is capped at
#   40,000 periods, so one search takes a few seconds.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import lightkurve as lk
import numpy as np
from astropy.timeseries import BoxLeastSquares

from config import CONFIG, Config

log = logging.getLogger(__name__)

# "Aliases": the same signal also shows up at half, double, a third... of the
# true period. These ratios are used to skip such echoes of a stronger peak.
ALIAS_RATIOS = (1.0, 0.5, 2.0, 1 / 3, 3.0, 2 / 3, 1.5)


# One candidate period from the periodogram (the search keeps the best few).
@dataclass
class Peak:
    period: float
    power: float
    t0: float
    duration: float
    depth: float
    grid_power: float = float("nan")   # power on the search grid (used for SDE)


# The winning dip. Everything later in the pipeline reads this object.
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

    # Depth divided by its uncertainty: how clearly the dip stands out of the noise.
    @property
    def depth_snr(self) -> float:
        return self.depth / self.depth_err if self.depth_err > 0 else 0.0

    # "Ephemeris" = the timetable of the dips: period, first dip time, length.
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

    # Counts as a real detection when: SDE >= 7, the dip is a dip (depth > 0), and
    # the evidence does not come mostly from one single dip (that would be a glitch).
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


# The whole periodogram (for the BLS plot in the app) plus the winning Signal.
@dataclass
class SearchResult:
    periods: np.ndarray
    power: np.ndarray
    signal: Signal
    grid_size_full: int
    grid_size_used: int


# Plain NumPy arrays of time, flux and error, without missing or bad values.
def _arrays(lc: lk.LightCurve) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(lc.time.value, float)
    f = np.asarray(lc.flux.value, float)
    e = np.asarray(lc.flux_err.value, float) if lc.flux_err is not None else None
    ok = np.isfinite(t) & np.isfinite(f)
    if e is not None:
        ok &= np.isfinite(e) & (e > 0)
        return t[ok], f[ok], e[ok]
    return t[ok], f[ok], np.full(ok.sum(), np.std(f[ok]))


def full_grid_size(baseline: float, min_duration: float, pmin: float, pmax: float,
                   frequency_factor: float = 1.0) -> int:
    """Number of periods ``BoxLeastSquares.autoperiod`` would generate, without building it.

    The full-resolution frequency step is ``frequency_factor * min_duration / baseline**2``.
    With sectors years apart this count reaches hundreds of millions, so the
    array itself must never be allocated.
    """
    df = frequency_factor * min_duration / baseline**2
    return 1 + int(round((1 / pmin - 1 / pmax) / df))


def period_grid(baseline: float, durations: np.ndarray, pmin: float, pmax: float,
                cfg: Config = CONFIG) -> tuple[np.ndarray, int]:
    """Frequency-uniform grid of at most ``cfg.max_grid_size`` periods.

    Returns the grid and the size the full-resolution grid would have had.
    """
    n_full = full_grid_size(baseline, float(np.min(durations)), pmin, pmax,
                            cfg.frequency_factor)
    if n_full > cfg.max_grid_size:
        log.warning("Full period grid has %d periods; searching %d and refining the best peaks",
                    n_full, cfg.max_grid_size)
    freqs = np.linspace(1 / pmax, 1 / pmin, min(n_full, cfg.max_grid_size))
    return np.sort(1 / freqs), n_full


@dataclass
class PowerResult:
    """The parts of astropy's BLS result the pipeline uses, one entry per period."""

    period: np.ndarray
    power: np.ndarray
    transit_time: np.ndarray
    duration: np.ndarray
    depth: np.ndarray


def limited_power(model: BoxLeastSquares, periods: np.ndarray, durations: np.ndarray,
                  max_duty: float) -> PowerResult:
    """BLS power where each trial period only tries durations <= ``max_duty`` x period.

    A "dip" lasting most of the period is not a dip: on a pulsating star such a
    fit can cover the whole cycle. astropy applies one list of durations to all
    periods, so the period grid is split into bands that share the same allowed
    durations. Periods with no allowed duration get NaN power.
    """
    periods = np.asarray(periods, float)
    durations = np.sort(np.asarray(durations, float))
    # For each period, how many of the (sorted) durations are short enough.
    n_allowed = np.searchsorted(durations, max_duty * periods, side="right")
    out = PowerResult(periods, *(np.full(periods.size, np.nan) for _ in range(4)))
    for n in np.unique(n_allowed):
        if n == 0:
            continue
        m = n_allowed == n
        r = model.power(periods[m], durations[:n], objective="likelihood")
        out.power[m] = r.power
        out.transit_time[m] = r.transit_time
        out.duration[m] = r.duration
        out.depth[m] = r.depth
    return out


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
            coarse_df: float, fine_df: float, pmin: float, pmax: float,
            max_duty: float) -> PowerResult:
    """Re-search a narrow frequency window around ``period`` at full resolution."""
    f0 = 1 / period
    half = 3 * coarse_df
    n = int(min(max(2 * half / fine_df, 50), 8000))
    freqs = np.linspace(max(f0 - half, 1 / pmax), min(f0 + half, 1 / pmin), n)
    return limited_power(model, np.sort(1 / freqs), durations, max_duty)


# Detailed numbers for one period: overall depth, depth of odd and even dips
# separately (binary test), depth half an orbit later, and the evidence from
# each single dip (used to spot one-off glitches).
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


def _stats_2p(model: BoxLeastSquares, period: float, duration: float, t0: float,
              baseline: float) -> dict[str, Any]:
    """Stats at twice the period, or {} when unavailable: 2P longer than the data, or
    data gaps leaving one half of the 2P fold with no in-transit points."""
    if 2 * period > baseline:
        return {}
    try:
        return _stats(model, 2 * period, duration, t0)
    except ValueError:
        log.info("2P stats unavailable at P=%.4f d (no in-transit points)", period)
        return {}


def search(lc: lk.LightCurve, cfg: Config = CONFIG, period_min: float | None = None,
           period_max: float | None = None) -> SearchResult:
    """Run BLS on ``lc`` and return the periodogram and the best signal."""
    t, f, e = _arrays(lc)
    if t.size < 50:
        raise ValueError(f"only {t.size} points to search")
    baseline = float(t.max() - t.min())
    durations = cfg.search_durations()
    pmin = period_min or cfg.period_min_d
    # At least two dips must fit in the data, so the longest period searched is half
    # the time span.
    pmax = min(period_max or cfg.period_max_d, baseline / 2)
    if pmax <= pmin:
        raise ValueError(f"baseline {baseline:.1f} d too short for periods >= {pmin} d")

    model = BoxLeastSquares(t, f, dy=e)
    # Trial periods, uniform in frequency and capped at cfg.max_grid_size.
    # Cost of the search ~ O(number of periods x number of points).
    periods, n_full = period_grid(baseline, durations, pmin, pmax, cfg)
    res = limited_power(model, periods, durations, cfg.max_duty_cycle)
    power = np.asarray(res.power, float)
    finite = np.isfinite(power)
    # SDE is measured on sqrt(power) (the "signal residue"). The likelihood power
    # itself has an exponential tail, so its largest value among tens of thousands
    # of trial periods is ~10 standard deviations high even for pure noise.
    # SDE = (best value - average) / spread, over all trial periods.
    sr = np.sqrt(np.clip(power[finite], 0, None))
    sde_mu, sde_sd = float(np.mean(sr)), float(np.std(sr))

    # Strongest peaks that are not harmonics (P/2, 2P, ...) of a stronger one; if the
    # grid was coarsened, the best few are re-searched at full resolution.
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
            r2 = _refine(model, durations, p, coarse_df, fine_df, pmin, pmax,
                         cfg.max_duty_cycle)
            j = int(np.nanargmax(r2.power))
            p, pw, t0, dur, dep = (r2.period[j], r2.power[j], r2.transit_time[j],
                                   r2.duration[j], r2.depth[j])
        peaks.append(Peak(float(p), float(pw), float(t0), float(dur), float(dep), grid_pw))
    peaks.sort(key=lambda pk: pk.power, reverse=True)

    # Detailed statistics of the winner: depth and its error, odd vs even dips, evidence
    # per dip, and the same at twice the period (a test for binaries).
    best = peaks[0]
    stats = _stats(model, best.period, best.duration, best.t0)
    stats_2p = _stats_2p(model, best.period, best.duration, best.t0, baseline)
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
    # Narrow re-search (+-0.2% around the known period) on the re-detrended data.
    periods = np.linspace(p * (1 - window_frac), p * (1 + window_frac), 400)
    res = limited_power(model, periods, durations, cfg.max_duty_cycle)
    j = int(np.nanargmax(res.power))
    period, t0, dur = float(res.period[j]), float(res.transit_time[j]), float(res.duration[j])
    stats = _stats(model, period, dur, t0)
    baseline = float(t.max() - t.min())
    stats_2p = _stats_2p(model, period, dur, t0, baseline)
    return Signal(period=period, t0=t0, duration=dur, depth=stats["depth"][0],
                  depth_err=stats["depth"][1], power=float(res.power[j]), sde=signal.sde,
                  stats=stats, stats_2p=stats_2p, top_peaks=signal.top_peaks,
                  method=signal.method)


# Cut out the dips of a found signal so a second, weaker signal can be searched.
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
        # Hide the signal just found, then search what is left; stop when nothing more
        # is detected or the preset's maximum number of signals is reached.
        current = mask_signal(current, results[-1].signal)
        try:
            nxt = search(current, cfg)
        except ValueError:
            break
        if not nxt.signal.detected(cfg):
            break
        results.append(nxt)
    return results
