"""Eclipsing-binary characterisation: true period, eclipses, binary type.

True orbital period. The dip search often locks onto half the orbital period
of an eclipsing binary: its two eclipses per orbit look like one repeating dip.
Which period is right is decided from the light curve alone:

1. Alternating dips have clearly different depths -> the search folded the
   primary and secondary eclipses together: true period = 2x detected.
2. Otherwise, a clear second eclipse halfway between the dips -> the detected
   period already contains both eclipses: true period = detected.
3. Otherwise the two cases look identical in the data (two equal eclipses, or a
   secondary too faint to see). The detected period is kept and flagged as
   ambiguous: the true period may be twice as long.

When the period is ambiguous, the module looks again at alternate eclipses
(folding at twice the period) using every point and every sector of the run:
depth, width and shape of the odd and even eclipses, with per-eclipse errors.
A clear difference means the true period is twice the detected one.

Binary type (detached / semi-detached / contact) is rule-based from the
light-curve shape; no model is trained on the small set of binaries.
Eclipse-timing (O-C) analysis is future work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, Config
from src.recycler.common import KeyFact, ModuleResult, fmt, period_words

SIGMA_MIN = 3.0          # "clearly different" / "clearly present": 3 standard deviations

# Morphology rules, set from how each type looks (not fitted to any star list).
# out-of-eclipse variation: smooth brightness change between eclipses relative to
# the eclipse depth; eclipse width: full eclipse width as a share of the orbit.
CONTACT_OOE, CONTACT_WIDTH = 0.30, 0.15
SEMI_OOE, SEMI_WIDTH = 0.08, 0.08
MORPH_WORDS = {"detached": "well-separated", "semi-detached": "close", "contact": "touching"}


@dataclass
class TruePeriod:
    period_d: float
    factor: int           # true period / detected period
    rule: str             # "odd_even_differ" | "secondary_seen" | "ambiguous"
    ambiguous: bool


def true_period(detected_period: float, features: dict[str, float]) -> TruePeriod:
    """Orbital period of an eclipsing binary from the detected dip period and features."""
    oe = features.get("odd_even_depth_sigma", np.nan)
    sec = features.get("secondary_snr", np.nan)
    if oe is not None and np.isfinite(oe) and oe >= SIGMA_MIN:
        return TruePeriod(2 * detected_period, 2, "odd_even_differ", False)
    if sec is not None and np.isfinite(sec) and sec >= SIGMA_MIN:
        return TruePeriod(detected_period, 1, "secondary_seen", False)
    return TruePeriod(detected_period, 1, "ambiguous", True)


def _f(d: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        v = float(d.get(key))
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def morphology(features: dict[str, Any], period_true: float) -> tuple[str, dict[str, float]]:
    """Detached / semi-detached / contact from the light-curve shape (rule-based).

    - contact: the light changes all the time between eclipses (variation at least
      30% of the eclipse depth) or the eclipses fill at least 15% of the orbit;
    - semi-detached: variation at least 8% of the depth, or eclipses at least 8%
      of the orbit;
    - detached: flat between short eclipses.
    """
    depth = _f(features, "depth_ppm")
    ooe = abs(_f(features, "ellip_amp_ppm", 0.0)) + abs(_f(features, "refl_amp_ppm", 0.0))
    ooe_rel = ooe / depth if depth > 0 else math.nan
    t14 = _f(features, "t14_hr")
    if not math.isfinite(t14):
        t14 = _f(features, "duration_hr")
    width = t14 / 24 / period_true if period_true and period_true > 0 else math.nan
    if (math.isfinite(ooe_rel) and ooe_rel >= CONTACT_OOE) or \
            (math.isfinite(width) and width >= CONTACT_WIDTH):
        kind = "contact"
    elif (math.isfinite(ooe_rel) and ooe_rel >= SEMI_OOE) or \
            (math.isfinite(width) and width >= SEMI_WIDTH):
        kind = "semi-detached"
    else:
        kind = "detached"
    return kind, {"out_of_eclipse_variation": ooe_rel, "eclipse_width_fraction": width}


def catalog_morph_class(morph: float) -> str:
    """The TESS EB catalog's morphology number (0 = well detached ... 1 = contact)."""
    if not math.isfinite(morph):
        return "unknown"
    return "detached" if morph < 0.5 else "semi-detached" if morph < 0.7 else "contact"


def morphology_measured(ecl: dict[str, float], period_true: float
                        ) -> tuple[str, dict[str, float]] | None:
    """Same rules as ``morphology``, from the module's own measurements at the true
    period: the fitted eclipse width and the out-of-eclipse variation measured on
    the light curve with stellar variability kept. None if they are missing."""
    depth = ecl.get("primary_depth", math.nan)
    width_d = ecl.get("primary_width_d", math.nan)
    ooe = ecl.get("out_of_eclipse_amp", math.nan)
    if not (math.isfinite(depth) and depth > 0 and math.isfinite(width_d)):
        return None
    feats = {"depth_ppm": depth * 1e6, "ellip_amp_ppm": (ooe if math.isfinite(ooe) else 0) * 1e6,
             "refl_amp_ppm": 0.0, "t14_hr": width_d * 24}
    return morphology(feats, period_true)


def out_of_eclipse_amplitude(t: np.ndarray, f: np.ndarray, period: float, t0: float,
                             mask_halfwidth: float, sec_phase: float) -> float:
    """Smooth brightness change between eclipses (sum of the once- and twice-per-orbit
    semi-amplitudes), fitted with both eclipses left out."""
    from src.vetting.common import phase, phase_time

    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if t.size < 200:
        return math.nan
    ph = phase(t, period, t0)
    sec = np.abs(phase_time(t, period, t0 + sec_phase * period))
    keep = (np.abs(phase_time(t, period, t0)) > mask_halfwidth) & (sec > mask_halfwidth)
    if keep.sum() < 100:
        return math.nan
    x = 2 * np.pi * ph[keep]
    X = np.column_stack([np.ones(x.size), np.cos(x), np.sin(x), np.cos(2 * x), np.sin(2 * x)])
    c, *_ = np.linalg.lstsq(X, f[keep] / np.median(f[keep]), rcond=None)
    return float(np.hypot(c[1], c[2]) + np.hypot(c[3], c[4]))


def _eclipse_events(t: np.ndarray, period: float, t0: float, halfwidth: float
                    ) -> np.ndarray:
    """Index of the eclipse each point belongs to (-1 if outside every eclipse window)."""
    n = np.round((t - t0) / period)
    near = np.abs(t - (t0 + n * period)) < halfwidth
    return np.where(near, n, -1).astype(np.int64)


def odd_even_at_double(t: np.ndarray, f: np.ndarray, period: float, t0: float,
                       duration: float, n_boot: int = 40, seed: int = 0) -> dict[str, float]:
    """Compare alternate eclipses (odd vs even, i.e. folding at 2x ``period``).

    Depth: one depth per eclipse (local baseline), compared as the mean of odd vs
    even with errors from the eclipse-to-eclipse scatter. Width and shape
    (ingress share): a trapezoid fit to the stacked odd and stacked even eclipses,
    with errors from resampling whole eclipses. Returns the significance of each
    difference in standard deviations.
    """
    from src.vetting.common import phase_time
    from src.vetting.transit_features import fit_trapezoid

    out = {"sigma_depth": math.nan, "sigma_width": math.nan, "sigma_shape": math.nan,
           "n_odd": 0, "n_even": 0}
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = np.asarray(t, float)[ok], np.asarray(f, float)[ok]
    if t.size < 200 or not (period > 0 and duration > 0):
        return out
    cadence = float(np.median(np.diff(t)))
    win = 2.5 * duration
    ev = _eclipse_events(t, period, t0, win)
    depths = {0: [], 1: []}
    for n in np.unique(ev[ev >= 0]):
        m = ev == n
        dt = np.abs(t[m] - (t0 + n * period))
        inside, base = dt < duration / 2, dt > duration
        if inside.sum() >= 3 and base.sum() >= 5:
            depths[int(n % 2)].append(np.median(f[m][base]) - np.mean(f[m][inside]))
    d0, d1 = np.asarray(depths[0]), np.asarray(depths[1])
    out["n_even"], out["n_odd"] = int(d0.size), int(d1.size)
    if d0.size >= 2 and d1.size >= 2:
        e = np.hypot(d0.std(ddof=1) / np.sqrt(d0.size), d1.std(ddof=1) / np.sqrt(d1.size))
        out["sigma_depth"] = float(abs(d0.mean() - d1.mean()) / e) if e > 0 else math.nan

    rng = np.random.default_rng(seed)

    def fit(parity: int, events: np.ndarray | None = None):
        m = (ev >= 0) & (ev % 2 == parity)
        if events is not None:
            idx = np.concatenate([np.flatnonzero(ev == n) for n in events])
            tt, ff = t[idx], f[idx]
        else:
            tt, ff = t[m], f[m]
        depth0 = max(float(np.mean(depths[parity])) if depths[parity] else 0.0, 1e-6)
        pt = phase_time(tt, 2 * period, t0 + parity * period)
        r = fit_trapezoid(pt, ff, duration, depth0, cadence)
        return (r["t14"], r["frac"]) if r["ok"] else (math.nan, math.nan)

    stats = {}
    for parity in (0, 1):
        evs = np.unique(ev[(ev >= 0) & (ev % 2 == parity)])
        if evs.size < 2:
            return out
        best = fit(parity)
        boots = np.array([fit(parity, rng.choice(evs, evs.size)) for _ in range(n_boot)])
        stats[parity] = (best, np.nanstd(boots, axis=0))
    for k, name in ((0, "sigma_width"), (1, "sigma_shape")):
        a, b = stats[0][0][k], stats[1][0][k]
        e = math.hypot(stats[0][1][k], stats[1][1][k])
        if math.isfinite(a) and math.isfinite(b) and e > 0:
            out[name] = float(abs(a - b) / e)
    return out


def measure_eclipses(t: np.ndarray, f: np.ndarray, period: float, t0: float,
                     duration: float, t_var: np.ndarray | None = None,
                     f_var: np.ndarray | None = None) -> dict[str, float]:
    """Primary and secondary eclipse depth, phase and width at the true period.

    The deeper of the two eclipses is called primary. Approximations (labelled
    as such in the report):
    - e cos(w) ~ (pi / 2) (secondary phase - 0.5): how far the secondary is from
      halfway;
    - e sin(w) ~ (w_sec - w_pri) / (w_sec + w_pri): from the eclipse widths;
    - temperature ratio ~ (secondary depth / primary depth) ** 0.25, assuming
      similar eclipse geometry and that the depths compare surface brightness.

    ``t_var, f_var``: the light curve with stellar variability kept, used for the
    out-of-eclipse variation (the flattened curve has it removed).
    """
    from src.vetting.common import box_depth, phase_time
    from src.vetting.transit_features import fit_trapezoid, secondary_search

    nan = math.nan
    out = {k: nan for k in ("primary_depth", "secondary_depth", "secondary_phase",
                            "secondary_snr", "primary_width_d", "secondary_width_d",
                            "ecosw", "esinw", "temp_ratio", "out_of_eclipse_amp")}
    t, f = np.asarray(t, float), np.asarray(f, float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if t.size < 100 or not (period > 0 and duration > 0):
        return out
    cadence = float(np.median(np.diff(t)))

    def depth_at(epoch: float) -> float:
        pt = phase_time(t, period, epoch)
        d, _, _ = box_depth(f, np.abs(pt) < duration / 2,
                            (np.abs(pt) > duration) & (np.abs(pt) < 3 * duration))
        return d

    if depth_at(t0 + period / 2) > depth_at(t0):        # the deeper eclipse is primary
        t0 = t0 + period / 2
    d0 = depth_at(t0)
    prim = fit_trapezoid(phase_time(t, period, t0), f, duration, max(d0, 1e-6), cadence)
    w_p = prim["t14"] if prim["ok"] else duration
    sec = secondary_search(t, f, period, t0, duration, 1.0, max(duration, w_p))
    out.update(primary_depth=prim["depth"] if prim["ok"] else d0, primary_width_d=w_p,
               secondary_phase=sec["secondary_best_phase"],
               secondary_depth=sec["secondary_best_depth_ppm"] / 1e6,
               secondary_snr=sec["secondary_best_snr"])
    if math.isfinite(out["secondary_snr"]) and out["secondary_snr"] >= SIGMA_MIN:
        t0s = t0 + out["secondary_phase"] * period
        fs = fit_trapezoid(phase_time(t, period, t0s), f, duration,
                           max(out["secondary_depth"], 1e-6), cadence)
        w_s = fs["t14"] if fs["ok"] else nan
        out["secondary_width_d"] = w_s
        if fs["ok"] and prim["ok"]:
            # both depths from the same trapezoid fit, so their ratio is like for like
            out["secondary_depth"] = fs["depth"]
        out["ecosw"] = math.pi / 2 * (out["secondary_phase"] - 0.5)
        if math.isfinite(w_s) and w_s + w_p > 0:
            out["esinw"] = (w_s - w_p) / (w_s + w_p)
        if out["primary_depth"] > 0 and out["secondary_depth"] > 0:
            out["temp_ratio"] = (out["secondary_depth"] / out["primary_depth"]) ** 0.25
    if t_var is not None and f_var is not None:
        sp = out["secondary_phase"] if out["secondary_snr"] >= SIGMA_MIN else 0.5
        out["out_of_eclipse_amp"] = out_of_eclipse_amplitude(
            np.asarray(t_var, float), np.asarray(f_var, float), period, t0,
            0.75 * max(w_p, duration), sp)
    return out


@lru_cache(maxsize=1)
def _eb_catalog(path: str) -> pd.DataFrame:
    ebs = pd.read_parquet(path)
    return ebs.assign(tic_id=ebs["TIC"].astype(np.int64)).drop_duplicates("tic_id") \
        .set_index("tic_id")


def catalog_entry(tic_id: int | None, cfg: Config = CONFIG) -> dict[str, Any] | None:
    """The cached TESS EB catalog entry for this star, if any."""
    path = cfg.paths.labels / "raw" / "tess_ebs.parquet"
    if not tic_id or not path.exists():
        return None
    ebs = _eb_catalog(str(path))
    if int(tic_id) not in ebs.index:
        return None
    r = ebs.loc[int(tic_id)]
    morph = float(r["Morph"]) if pd.notna(r["Morph"]) else math.nan
    return {"period_d": float(r["Per"]), "morph": morph, "type": catalog_morph_class(morph)}


def true_period_with_data(signal: dict[str, Any], features: dict[str, Any],
                          t: np.ndarray | None, f: np.ndarray | None
                          ) -> tuple[TruePeriod, dict[str, float]]:
    """``true_period``; if that is ambiguous, look again at alternate eclipses with
    all the data (``odd_even_at_double``)."""
    tp = true_period(float(signal["period"]), features)
    check: dict[str, float] = {}
    if tp.ambiguous and t is not None and f is not None:
        check = odd_even_at_double(np.asarray(t, float), np.asarray(f, float),
                                   float(signal["period"]), float(signal["t0"]),
                                   float(signal["duration"]))
        sig = [check.get(k, math.nan) for k in ("sigma_depth", "sigma_width", "sigma_shape")]
        if any(math.isfinite(x) and x >= SIGMA_MIN for x in sig):
            tp = TruePeriod(2 * float(signal["period"]), 2, "alternate_eclipses_differ", False)
    return tp, check


def characterize(signal: dict[str, Any], features: dict[str, Any], t: np.ndarray | None = None,
                 f: np.ndarray | None = None, tic_id: int | None = None,
                 cfg: Config = CONFIG, t_var: np.ndarray | None = None,
                 f_var: np.ndarray | None = None) -> ModuleResult:
    """EB module: true period, eclipses and binary type, in plain words.

    ``t, f``: flattened light curve (eclipses); ``t_var, f_var``: light curve with
    stellar variability kept (out-of-eclipse variation).
    """
    tp, check = true_period_with_data(signal, features, t, f)
    ecl = (measure_eclipses(t, f, tp.period_d, float(signal["t0"]), float(signal["duration"]),
                            t_var, f_var) if t is not None and f is not None else {})
    measured = morphology_measured(ecl, tp.period_d) if ecl else None
    kind, morph_inputs = measured if measured else morphology(features, tp.period_d)
    cat = catalog_entry(tic_id, cfg)
    nan = math.nan
    sec_seen = ecl.get("secondary_snr", 0) >= SIGMA_MIN

    if tp.ambiguous:
        when = (f"every {period_words(tp.period_d)} or {period_words(2 * tp.period_d)}; this "
                f"data can't tell which")
    else:
        when = f"every {period_words(tp.period_d)}"
    summary = (f"Not a planet, but still interesting: this is a pair of {MORPH_WORDS[kind]} "
               f"stars eclipsing each other {when}.")

    def pct(x: float) -> str:
        return fmt(x * 100, 3) if math.isfinite(x) else "unknown"

    facts = [
        KeyFact("Orbital period",
                f"{fmt(tp.period_d, 4)} or {fmt(2 * tp.period_d, 4)}" if tp.ambiguous
                else fmt(tp.period_d, 4), "days",
                "Time for the two stars to go once around each other."
                + (" Two equal eclipses per orbit and one eclipse per orbit (the second one "
                   "too faint to see) look the same in this data, so both are possible."
                   if tp.ambiguous else "")),
        KeyFact("Binary type", kind, "",
                "Detached: well apart. Semi-detached: one star fills its space and may pass "
                "gas to the other. Contact: the stars touch. Judged from the light-curve shape."),
        KeyFact("Main eclipse depth", pct(ecl.get("primary_depth", nan)), "%",
                "Light lost when the brighter star is hidden."),
        KeyFact("Second eclipse depth",
                pct(ecl.get("secondary_depth", nan)) if sec_seen else "not detected", "%",
                "Light lost when the fainter star is hidden."),
        KeyFact("Second eclipse position",
                fmt(ecl.get("secondary_phase", nan), 3) if sec_seen else "not detected",
                "of the orbit",
                "0.5 means exactly halfway between main eclipses, as for a circular orbit."),
        KeyFact("Eclipse widths", f"{fmt(ecl.get('primary_width_d', nan) * 24, 1)} / "
                f"{fmt(ecl.get('secondary_width_d', nan) * 24, 1)}", "hours",
                "How long the main / second eclipse lasts."),
        KeyFact("Temperature ratio (approx.)", fmt(ecl.get("temp_ratio", nan), 2), "",
                "Fainter star's temperature divided by the brighter star's, from the eclipse "
                "depths. Rough: assumes similar eclipse shapes."),
        KeyFact("e cos w, e sin w (approx.)", f"{fmt(ecl.get('ecosw', nan), 3)}, "
                f"{fmt(ecl.get('esinw', nan), 3)}", "",
                "Rough hints of an oval (eccentric) orbit from the second eclipse's position "
                "and width. Both near 0 means a circular orbit."),
        KeyFact("Catalog check",
                f"in the TESS eclipsing binary catalog: period {cat['period_d']:.4f} days, "
                f"type {cat['type']}" if cat else "not in the TESS eclipsing binary catalog", "",
                "Cached copy of the TESS eclipsing binary catalog."),
    ]
    details = {"true_period": tp.__dict__, "alternate_eclipse_check": check,
               "morphology": kind, "morphology_source": "measured" if measured else "features",
               "morphology_inputs": morph_inputs, "eclipses": ecl, "catalog": cat,
               "future_work": ["eclipse-timing (O-C) analysis across sectors"]}
    return ModuleResult("eb", summary, facts, details)
