"""Display-only views of a report: folds at the final period, eclipse zooms, top peaks.

``compute_views`` works on the full cleaned light curve (time, flux) and the report
dict. The app calls it after a live run; scripts/build_examples.py calls it at build
time and stores the result in the precomputed report, so examples need no network.
Nothing here feeds back into the analysis.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

FOLD_POINTS = 4000      # points kept per scatter plot (display only)
ZOOM_POINTS = 2500


def _num(v: Any) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return math.nan
    return v


def _thin(n: int, *arrays: np.ndarray) -> list[list[float]]:
    step = max(1, int(math.ceil(len(arrays[0]) / n)))
    return [np.round(np.asarray(a)[::step], 6).tolist() for a in arrays]


def binned_median(x: np.ndarray, y: np.ndarray, edges: np.ndarray, min_count: int = 3
                  ) -> tuple[list[float], list[float]]:
    """Median of y in each x bin (bins with fewer than ``min_count`` points dropped)."""
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1) & np.isfinite(y)
    idx, yv = idx[ok], y[ok]
    order = np.argsort(idx, kind="stable")
    idx, yv = idx[order], yv[order]
    starts = np.flatnonzero(np.r_[True, np.diff(idx) > 0])
    ends = np.r_[starts[1:], len(idx)]
    xs, ys = [], []
    for s, e in zip(starts, ends):
        if e - s >= min_count:
            k = idx[s]
            xs.append(round(float(0.5 * (edges[k] + edges[k + 1])), 6))
            ys.append(round(float(np.median(yv[s:e])), 7))
    return xs, ys


def _signal(report: dict[str, Any]) -> dict[str, Any]:
    sigs = report.get("signals") or [{}]
    return sigs[0] or {}


def eb_period(report: dict[str, Any]) -> float:
    eb = ((report.get("modules") or {}).get("eb") or {}).get("details") or {}
    return _num((eb.get("true_period") or {}).get("period_d"))


def variable_period(report: dict[str, Any]) -> float:
    f = report.get("features") or {}
    p = _num(f.get("var_period_d"))
    return p if math.isfinite(p) else _num(f.get("ls_period_1"))


def final_period(report: dict[str, Any]) -> tuple[float, str]:
    """(period, kind): the EB module's true period for binaries, the variable
    module's period for variable stars, otherwise the detected transit period."""
    final = (report.get("routing") or {}).get("primary_class")
    if final == "EB" and math.isfinite(eb_period(report)):
        return eb_period(report), "eb"
    if final == "VARIABLE" and math.isfinite(variable_period(report)):
        return variable_period(report), "variable"
    return _num(_signal(report).get("period")), "transit"


def fold(t: np.ndarray, f: np.ndarray, period: float, t0: float, nbins: int = 200,
         start: float = -0.5) -> dict[str, Any]:
    """Phase ``start``..``start + 1`` with the dip at 0: thinned points plus the binned
    median. Binaries use start=-0.25 so both eclipses (0 and ~0.5) are in view."""
    ph = ((t - t0) / period - start) % 1.0 + start
    bx, by = binned_median(ph, f, np.linspace(start, start + 1, nbins + 1))
    order = np.argsort(ph)
    px, py = _thin(FOLD_POINTS, ph[order], f[order])
    return {"phase": px, "flux": py, "bin_phase": bx, "bin_flux": by}


def zoom(t: np.ndarray, f: np.ndarray, period: float, centre: float, half_d: float,
         nbins: int = 60) -> dict[str, Any]:
    """Points within ``half_d`` days of ``centre`` (a time of one eclipse), in hours."""
    dt = (t - centre + 0.5 * period) % period - 0.5 * period
    m = np.abs(dt) < half_d
    h = dt[m] * 24
    bx, by = binned_median(h, f[m], np.linspace(-half_d * 24, half_d * 24, nbins + 1))
    order = np.argsort(h)
    px, py = _thin(ZOOM_POINTS, h[order], f[m][order])
    return {"hours": px, "flux": py, "bin_hours": bx, "bin_flux": by}


def top_peaks(report: dict[str, Any], n: int = 3) -> list[dict[str, float]]:
    """The strongest distinct BLS peaks the search reported."""
    peaks = _signal(report).get("top_peaks") or []
    return [{"period": _num(p.get("period")), "power": _num(p.get("power"))} for p in peaks[:n]]


def compute_views(t: np.ndarray, f: np.ndarray, report: dict[str, Any]) -> dict[str, Any]:
    t, f = np.asarray(t, float), np.asarray(f, float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    sig = _signal(report)
    t0, dur = _num(sig.get("t0")), _num(sig.get("duration"))
    views: dict[str, Any] = {"top_peaks": top_peaks(report)}
    period, kind = final_period(report)
    if not (len(t) and math.isfinite(period) and period > 0 and math.isfinite(t0)):
        return views
    views["final"] = dict(fold(t, f, period, t0, start=-0.25 if kind == "eb" else -0.5),
                          period=period, kind=kind,
                          half_window=(0.5 * dur / period if math.isfinite(dur) else None))
    mods = report.get("modules") or {}
    if kind == "variable" or "variable" in mods:
        vp = variable_period(report)
        if math.isfinite(vp) and vp > 0:
            v = fold(t, f, vp, t[0])
            # put the faintest point of the cycle at phase 0
            if v["bin_flux"]:
                shift = v["bin_phase"][int(np.argmin(v["bin_flux"]))]
                v = fold(t, f, vp, t[0] + shift * vp)
            views["variable"] = dict(v, period=vp)
    if "eb" in mods and math.isfinite(dur):
        P = eb_period(report)
        if math.isfinite(P) and P > 0:
            ecl = (mods["eb"].get("details") or {}).get("eclipses") or {}
            sec_phase = _num(ecl.get("secondary_phase"))
            sec_phase = sec_phase if math.isfinite(sec_phase) else 0.5
            half = min(3 * dur, 0.25 * P)
            cyc = np.round((t - t0) / P).astype(int)
            odd = cyc % 2 == 1
            views["eb"] = {
                "period": P, "secondary_phase": sec_phase, "duration_hr": dur * 24,
                "primary": zoom(t, f, P, t0, half),
                "secondary": zoom(t, f, P, t0 + sec_phase * P, half),
                "odd": zoom(t[odd], f[odd], P, t0, half),
                "even": zoom(t[~odd], f[~odd], P, t0, half),
            }
    return views


def report_with_views(rep) -> dict[str, Any]:
    """A TargetReport as a dict, with the display views added to its plot data."""
    d = rep.to_dict()
    pre = (getattr(rep, "_objects", None) or {}).get("pre")
    if pre is not None and d.get("plot_data"):
        d["plot_data"]["views"] = compute_views(pre.lc_flat.time.value, pre.lc_flat.flux.value, d)
    return d
