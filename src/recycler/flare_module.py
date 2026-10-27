"""Flare detection: sudden brightenings from magnetic activity on the star.

The cleaning step clips points far *above* the local trend (they are usually
cosmic rays or glitches), which also removes flares. This module therefore
looks at each sector before that clip: quality mask, divide by the median,
subtract a running median (the star's quiet level), then

1. candidate points: above the quiet level by ``flare_sigma`` x the scatter, and
   still by ``flare_sigma_err`` x the scatter after subtracting the point's error;
2. a flare: at least ``flare_min_points`` consecutive candidates with no data gap
   (one high point is a cosmic ray, not a flare), widened on both sides while
   the light stays above the quiet level;
3. shape: a flare is impulsive. It rises within ``flare_max_rise_min`` minutes
   and in at most ``flare_max_rise_frac`` of its decay time. A slower rise is the
   star's own variability that the running median could not follow (a Cepheid),
   and a rise as slow as the decay is a pulsation peak (a delta Scuti star).
   A flare also jumps up from the light just before it: the peak must stand
   above the median of the preceding 30 minutes by at least half its height
   (wiggles on top of a slow hump do not);
4. not at a dip: when light returns after an eclipse or transit it rises above
   the running median, which the dip still pulls down, so events within
   ``flare_eclipse_pad`` x the dip duration of the detected dip (phase 0), of
   phase 0.5 or of the strongest second dip are ignored.

Only a *detected* dip is masked; when nothing was detected the search period
is meaningless and masking it would hide much of the light curve.

5. not part of the star's own cycle: pulsating stars (RR Lyrae, delta Scuti)
   also brighten quickly, but a pulsation peak comes back every cycle and a flare
   does not. An event is rejected when a bump at least half as high is there one
   cycle earlier and one cycle later, at one of the star's three strongest
   periodogram periods or at the detected dip's period (a binary's bumps between
   eclipses come back every orbit). If such repeating events outnumber the
   flares, the star is reported as pulsating.

Flares *bunching* at a period is not used against them: on a spotted star,
flares often come from active regions that rotate into view, so they can bunch
at the rotation period. That bunching (Rayleigh test) is reported for
information only.

On a star whose own brightness swings by more than ``flare_max_var_amp``
(5th-95th percentile), events are reported as unreliable rather than as flares.

Per flare: peak amplitude, duration and equivalent duration (the flare's
energy divided by the star's quiet brightness, in seconds). The energy itself is
equivalent duration x the star's luminosity in the TESS band (600-1000 nm,
blackbody at the catalog temperature and radius), so it is approximate.

Flares are reported alongside the classifier's verdict; they are not one of its
four classes and are not given to the model.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from lightkurve.utils import TessQualityFlags
from scipy.ndimage import median_filter

from config import CONFIG, Config
from src.preprocess.preprocessor import robust_std
from src.recycler.common import KeyFact, ModuleResult, fmt

MAX_LISTED = 25                 # flares kept in the report (largest first)
TAIL_SIGMA = 0.5                # widen a flare while it stays this far above quiet
REPEAT_FRAC = 0.5               # a bump this high one cycle away: it repeats, not a flare
PRE_WINDOW_D = 30 / 1440        # light just before the event...
PRE_JUMP_FRAC = 0.5             # ...must be below the peak by this share of its height
_H, _C, _K = 6.62607e-27, 2.99792e10, 1.38065e-16     # cgs
_RSUN_CM = 6.957e10


def tess_band_luminosity(teff: float, rad_rsun: float) -> float:
    """Blackbody luminosity in the TESS band (600-1000 nm), erg/s; NaN if unknown."""
    if not (math.isfinite(teff) and math.isfinite(rad_rsun) and teff > 0 and rad_rsun > 0):
        return float("nan")
    lam = np.linspace(600e-7, 1000e-7, 400)                       # cm
    b = 2 * _H * _C**2 / lam**5 / np.expm1(_H * _C / (lam * _K * teff))
    flux = np.pi * np.trapezoid(b, lam)                           # erg/s/cm2 at the surface
    return float(4 * np.pi * (rad_rsun * _RSUN_CM) ** 2 * flux)


def _runs(idx: np.ndarray, t: np.ndarray, max_gap: float) -> list[np.ndarray]:
    """Split sorted indices into runs of consecutive cadences with no time gap."""
    if idx.size == 0:
        return []
    cut = np.flatnonzero((np.diff(idx) > 1) | (np.diff(t[idx]) > max_gap)) + 1
    return np.split(idx, cut)


def repeats_each_cycle(t: np.ndarray, resid: np.ndarray, t_start: float, t_end: float,
                       amplitude: float, periods: tuple[float, ...]) -> bool:
    """True if the same bump is there one cycle earlier and later at one of ``periods``.

    A pulsation peak comes back every cycle; a flare does not. A side without data
    (a gap, or outside the sector) is skipped, but at least one side must have data.
    """
    span = t[-1] - t[0]
    for p in periods:
        if not (math.isfinite(p) and 0 < p < span):
            continue
        sides = []
        for shift in (-p, p):
            lo, hi = np.searchsorted(t, [t_start + shift, t_end + shift])
            if hi - lo >= 3:
                sides.append(float(np.max(resid[lo:hi])) > REPEAT_FRAC * amplitude)
        if sides and all(sides):
            return True
    return False


def find_flares(t: np.ndarray, f: np.ndarray, e: np.ndarray, cfg: Config = CONFIG,
                periods: tuple[float, ...] = ()) -> tuple[list[dict[str, float]], float, int]:
    """Flares in one sector of normalised flux.

    Returns (flares, scatter, number of flare-shaped events rejected because they
    repeat every cycle at one of ``periods``).
    """
    cadence = float(np.median(np.diff(t)))
    n = max(5, int(round(cfg.flare_window_d / cadence)) | 1)
    quiet = median_filter(f, size=n, mode="nearest")
    resid = f / quiet - 1
    s = robust_std(resid)
    if not math.isfinite(s) or s <= 0:
        return [], s, 0
    err = np.where(np.isfinite(e), e / quiet, 0.0)
    cand = (resid > cfg.flare_sigma * s) & (resid - err > cfg.flare_sigma_err * s)
    max_gap = 1.5 * cadence
    flares: list[dict[str, float]] = []
    last_end, n_repeating = -1, 0
    for run in _runs(np.flatnonzero(cand), t, max_gap):
        if run.size < cfg.flare_min_points:
            continue
        i0, i1 = int(run[0]), int(run[-1])
        while (i0 > 0 and i0 - 1 > last_end and resid[i0 - 1] > TAIL_SIGMA * s
               and t[i0] - t[i0 - 1] <= max_gap):
            i0 -= 1
        while (i1 + 1 < t.size and resid[i1 + 1] > TAIL_SIGMA * s
               and t[i1 + 1] - t[i1] <= max_gap):
            i1 += 1
        if i0 <= last_end:
            continue
        seg = slice(i0, i1 + 1)
        peak = i0 + int(np.argmax(resid[seg]))
        rise, decay = t[peak] - t[i0], t[i1] - t[peak]
        if rise > cfg.flare_max_rise_frac * decay or rise * 1440 > cfg.flare_max_rise_min:
            continue
        pre = slice(*np.searchsorted(t, [t[i0] - PRE_WINDOW_D, t[i0]]))
        if pre.stop - pre.start >= 3:
            jump = resid[peak] - float(np.median(resid[pre]))
            if jump < PRE_JUMP_FRAC * resid[peak] or jump < cfg.flare_sigma * s:
                continue
        if repeats_each_cycle(t, resid, t[i0], t[i1], resid[peak], periods):
            n_repeating += 1
            last_end = i1
            continue
        # one cadence before and after, so a short flare's area is not cut off
        tt = t[i0:i1 + 1] * 86400.0
        ed = float(np.trapezoid(resid[seg], tt)) + float(resid[i0] + resid[i1]) * cadence * 43200
        flares.append({"t_start": float(t[i0]), "t_peak": float(t[peak]),
                       "t_end": float(t[i1]), "amplitude": float(resid[peak]),
                       "duration_min": float((t[i1] - t[i0]) * 1440 + cadence * 1440),
                       "equiv_duration_s": ed, "n_points": int(i1 - i0 + 1)})
        last_end = i1
    return flares, s, n_repeating


def near_dip(t_start: float, t_end: float, ephemeris: tuple[float, float, float] | None,
             phases: tuple[float, ...], pad: float) -> bool:
    """True if [t_start, t_end] overlaps a padded dip window at any of ``phases``."""
    if not ephemeris:
        return False
    period, t0, duration = ephemeris
    if not all(math.isfinite(x) and x > 0 for x in (period, duration)) or \
            not math.isfinite(t0):
        return False
    half = pad * duration
    for ph in phases:
        centre = t0 + ph * period
        k = round((0.5 * (t_start + t_end) - centre) / period)
        c = centre + k * period
        if t_start <= c + half and t_end >= c - half:
            return True
    return False


def rayleigh_p(r: float, n: int) -> float:
    """Chance that ``n`` random phases bunch at least this tightly (Zar's approximation)."""
    if n < 2:
        return 1.0
    nr = n * r
    return float(min(1.0, math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - nr * nr)) - (1 + 2 * n))))


def phase_concentration(times: np.ndarray, period: float) -> float:
    """Rayleigh R of event phases at ``period``: 0 = spread out, 1 = all at one phase."""
    if times.size == 0 or not (math.isfinite(period) and period > 0):
        return 0.0
    return float(abs(np.mean(np.exp(2j * np.pi * times / period))))


def _sector_arrays(lc, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Quality-masked, time-sorted, median-normalised arrays of one sector."""
    t = np.asarray(lc.time.value, float)
    f = np.asarray(lc.flux.value, float)
    e = (np.asarray(lc.flux_err.value, float) if lc.flux_err is not None
         else np.full_like(f, np.nan))
    good = np.isfinite(t) & np.isfinite(f)
    if "quality" in lc.colnames:
        q = np.asarray(lc["quality"].value, dtype=np.int64)
        good &= (q & TessQualityFlags.DEFAULT_BITMASK) == 0
    t, f, e = t[good], f[good], e[good]
    order = np.argsort(t)
    t, f, e = t[order], f[order], e[order]
    med = np.nanmedian(f) if f.size else np.nan
    if not math.isfinite(med) or med <= 0:
        return t[:0], f[:0], e[:0], float("nan")
    return t, f / med, e / med, float(np.median(np.diff(t))) if t.size > 1 else float("nan")


def characterize(collection, info=None, features: dict[str, Any] | None = None,
                 cfg: Config = CONFIG,
                 ephemeris: tuple[float, float, float] | None = None) -> ModuleResult:
    """Find and describe the flares in every sector of ``collection``.

    ``ephemeris`` is the detected dip's (period, t0, duration), all in days; events
    at its dips are ignored (see rule 4 above).
    """
    features = features or {}
    sec_phase = features.get("secondary_best_phase")
    phases = (0.0, 0.5) + ((float(sec_phase),) if sec_phase is not None
                           and math.isfinite(float(sec_phase)) else ())
    # the star's own cycles, plus the detected dip's (orbital) period: a binary's
    # out-of-eclipse bumps come back every orbit
    cand_p = [features.get(f"ls_period_{k}") for k in (1, 2, 3)]
    cand_p += [ephemeris[0]] if ephemeris else []
    periods = tuple(float(p) for p in cand_p if p is not None and math.isfinite(float(p)))
    flares: list[dict[str, float]] = []
    n_at_dips = n_repeating = 0
    observed_d, cadences, skipped = 0.0, [], []
    for lc in collection:
        sector = int(lc.meta.get("SECTOR", -1))
        t, f, e, cad = _sector_arrays(lc, cfg)
        if t.size < 50 or not math.isfinite(cad):
            skipped.append(sector)
            continue
        if cad * 1440 > cfg.flare_max_cadence_min:
            skipped.append(sector)
            continue
        found, _, n_rep = find_flares(t, f, e, cfg, periods)
        n_repeating += n_rep
        kept = [fl for fl in found if not near_dip(fl["t_start"], fl["t_end"], ephemeris,
                                                   phases, cfg.flare_eclipse_pad)]
        n_at_dips += len(found) - len(kept)
        flares += [dict(fl, sector=sector) for fl in kept]
        observed_d += t.size * cad
        cadences.append(cad * 1440)

    if not cadences:
        return ModuleResult(
            module="flare",
            plain_summary="Flares could not be checked: the data are too coarse "
                          "(flares last minutes, so they need fast-cadence data).",
            key_facts=[KeyFact("Flares", "not checked", "",
                               f"Needs data every {cfg.flare_max_cadence_min:.0f} min or "
                               f"faster")],
            details={"checked": False, "skipped_sectors": skipped})

    n = len(flares)
    peaks = np.array([fl["t_peak"] for fl in flares])
    conc = [(phase_concentration(peaks, p), p) for p in periods]
    r_best, p_best = max(conc) if conc else (0.0, float("nan"))
    p_chance = rayleigh_p(r_best, n)                 # information only (see docstring)
    periodic = n_repeating >= cfg.flare_min_count and n_repeating > n
    var_amp = float(features.get("amp_p2p_ppm") or float("nan")) / 1e6
    unreliable = math.isfinite(var_amp) and var_amp > cfg.flare_max_var_amp

    l_tess = tess_band_luminosity(getattr(info, "teff", float("nan")),
                                  getattr(info, "rad", float("nan")))
    for fl in flares:
        en = fl["equiv_duration_s"] * l_tess
        fl["log10_energy_erg"] = math.log10(en) if math.isfinite(en) and en > 0 else None

    rate = n / observed_d if observed_d > 0 else float("nan")
    status = ("pulsation" if periodic else "unreliable" if unreliable and n else
              "flare_star" if n >= cfg.flare_min_count else "few" if n else "none")
    is_flare_star = status == "flare_star"
    biggest = max(flares, key=lambda x: x["amplitude"]) if flares else None
    energies = [fl["log10_energy_erg"] for fl in flares if fl["log10_energy_erg"] is not None]

    if periodic:
        summary = (f"Its {n_repeating} short brightenings come back every cycle of the star's "
                   f"own variation, so they are pulsations, not flares.")
    elif status == "unreliable":
        summary = (f"{n} short brightening{'s' if n > 1 else ''} seen, but the star's own "
                   f"brightness changes by {var_amp * 100:.0f}%, too much to tell flares from "
                   f"its variability.")
    elif is_flare_star:
        summary = (f"It is also a flare star: {n} sudden brightenings in {observed_d:.0f} days "
                   f"of data (about {rate:.2f} per day), the largest "
                   f"{biggest['amplitude'] * 100:.2g}% brighter for "
                   f"{biggest['duration_min']:.0f} minutes, a sign of strong magnetic activity.")
    elif n:
        summary = (f"{n} possible flare{'s' if n > 1 else ''} in {observed_d:.0f} days of data; "
                   f"too few to call it a flare star.")
    else:
        summary = f"No flares in {observed_d:.0f} days of data."

    facts = [
        KeyFact("Flares found", str(n), "",
                "Sudden brightenings lasting minutes, from magnetic outbursts on the star"),
        KeyFact("Flare rate", fmt(rate, 3), "per day"),
        KeyFact("Largest flare", fmt(biggest["amplitude"] * 100, 2) if biggest else "none", "%",
                "How much brighter the star got at the flare's peak"),
        KeyFact("Most energetic", f"10^{max(energies):.1f}" if energies else "unknown", "erg",
                "Approximate: equivalent duration x the star's TESS-band brightness. The Sun's "
                "largest flares reach about 10^32 erg"),
        KeyFact("Flare star?", {"pulsation": "pulsation, not flares",
                                "unreliable": "can't tell (star too variable)",
                                "flare_star": "yes"}.get(status, "no"), "",
                f"Yes when at least {cfg.flare_min_count} flares are found that do not repeat "
                f"with the star's own cycle"),
    ]
    listed = sorted(flares, key=lambda x: -x["equiv_duration_s"])[:MAX_LISTED]
    return ModuleResult(module="flare", plain_summary=summary, key_facts=facts, details={
        "checked": True, "status": status, "n_flares": n, "rate_per_day": rate,
        "observed_days": observed_d, "cadence_min": float(np.median(cadences)),
        "is_flare_star": is_flare_star, "periodic": periodic, "var_amplitude": var_amp,
        "phase_concentration": r_best, "phase_period_d": p_best, "periodic_p": p_chance,
        "tess_luminosity_erg_s": l_tess, "skipped_sectors": skipped,
        "n_ignored_at_dips": n_at_dips, "n_repeating_each_cycle": n_repeating,
        "flares": sorted(listed, key=lambda x: x["t_peak"])})
