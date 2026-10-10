"""End-to-end analysis of one target. Shared by the UI, the CLI and the batch scripts."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from config import CONFIG, Config
from src.report import TargetReport

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]


def _noop(stage: str, frac: float) -> None:
    pass


def physical_parameters(signal, star) -> dict[str, float]:
    """Planet-like interpretation of the dip using the star's catalog values."""
    from src import physics

    nan = float("nan")
    rad = getattr(star, "rad", nan)
    mass = getattr(star, "mass", nan)
    teff = getattr(star, "teff", nan)
    out = {"period_d": signal.period, "depth_ppm": signal.depth * 1e6,
           "rp_over_rs": physics.radius_ratio(signal.depth),
           "rp_rearth": nan, "rp_rjup": nan, "a_au": nan, "t_eq_k": nan}
    if np.isfinite(rad):
        out["rp_rearth"] = physics.planet_radius_rearth(signal.depth, rad)
        out["rp_rjup"] = out["rp_rearth"] * physics.R_EARTH_M / physics.R_JUP_M
    if np.isfinite(mass):
        out["a_au"] = physics.semi_major_axis_au(signal.period, mass)
        if np.isfinite(teff) and np.isfinite(rad):
            out["t_eq_k"] = physics.equilibrium_temp_k(teff, rad, out["a_au"])
    return out


def classify(features: dict[str, float], cfg: Config = CONFIG) -> dict[str, Any]:
    """Score the features with the trained triage model, if one exists."""
    from src.triage.model import load_model

    model = load_model(cfg)
    if model is None:
        return {"status": "no_model",
                "message": "No trained classifier found in models/. "
                           "Run scripts/build_labels.py, build_features.py and train_triage.py."}
    return model.predict(features, cfg)


def _downsample(x: np.ndarray, *ys: np.ndarray, n: int) -> list[list[float]]:
    """Evenly thin arrays to at most ``n`` points (plotting only)."""
    step = max(1, int(np.ceil(len(x) / n)))
    return [np.asarray(a)[::step].tolist() for a in (x, *ys)]


def _peak_preserving(x: np.ndarray, y: np.ndarray, n: int) -> list[list[float]]:
    """Thin a periodogram keeping the maximum in each chunk so peaks survive."""
    if len(x) <= n:
        return [x.tolist(), y.tolist()]
    chunks = np.array_split(np.arange(len(x)), n)
    idx = [c[np.nanargmax(y[c])] for c in chunks if len(c)]
    return [x[idx].tolist(), y[idx].tolist()]


def _plot_data(pre, result, signal, details, cfg: Config) -> dict[str, Any]:
    from src.vetting.common import bin_xy, phase_time

    n = cfg.plot_max_points
    t, f = pre.lc_norm.time.value, pre.lc_norm.flux.value
    tn, fn, sn = _downsample(t, f, pre.lc_norm["sector"].value, n=n)
    tf, ff = _downsample(pre.lc_flat.time.value, pre.lc_flat.flux.value, n=n)
    pt = phase_time(pre.lc_flat.time.value, signal.period, signal.t0)
    edges = np.linspace(-0.5, 0.5, 201) * signal.period
    xb, yb, _, _ = bin_xy(pt, pre.lc_flat.flux.value, edges)
    zoom = np.abs(pt) < 3 * signal.duration
    xz, yz = _downsample(pt[zoom] * 24, pre.lc_flat.flux.value[zoom], n=5000)
    xa, ya = _downsample(pt / signal.period, pre.lc_flat.flux.value, n=5000)
    return {
        "lc_norm": {"time": tn, "flux": fn, "sector": sn},
        "lc_flat": {"time": tf, "flux": ff},
        "periodogram": dict(zip(("period", "power"),
                                _peak_preserving(result.periods, result.power, 4000))),
        "fold": {"phase": xa, "flux": ya, "bin_phase": (xb / signal.period).tolist(),
                 "bin_flux": yb.tolist()},
        "fold_zoom": {"hours": xz, "flux": yz},
        "trapezoid": details.get("trapezoid", {}),
        "per_sector": details.get("per_sector", []),
        "ls": details.get("ls", {}),
    }


def analyze_target(target: str, preset: str = "standard",
                   progress_cb: ProgressCallback | None = None,
                   cfg: Config = CONFIG, collection=None, target_info=None) -> TargetReport:
    """Resolve, download, clean, search and describe one star.

    ``collection`` (a LightCurveCollection) and ``target_info`` let tests and
    offline callers skip the network.
    """
    from src.detect.bls import SearchResult, evaluate, multi_search, search
    from src.ingest.fetcher import fetch_lightcurves
    from src.ingest.resolver import TargetInfo, resolve_target
    from src.preprocess.preprocessor import (bin_lightcurve, prewhiten, preprocess,
                                             redetrend_masked)
    from src.vetting.features import extract_features

    cfg = cfg.with_preset(preset)
    cb = progress_cb or _noop
    report = TargetReport(target=target, preset=preset)
    timings: dict[str, float] = {}
    t_last = t_start = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t_last
        now = time.perf_counter()
        timings[name] = round(now - t_last, 2)
        t_last = now

    cb("Looking up the star", 0.0)
    info = target_info or (resolve_target(target, cfg) if collection is None
                           else TargetInfo(tic_id=0, common_name=target))
    report.tic_id = info.tic_id
    report.star = info.to_dict()
    lap("resolve")

    if collection is None:
        collection, sectors = fetch_lightcurves(
            info.tic_id, cfg, progress_cb=lambda s, f: cb(s, 0.05 + 0.35 * f))
        report.sectors = [s.to_dict() for s in sectors]
    else:
        report.sectors = [{"sector": int(lc.meta.get("SECTOR", -1)),
                           "author": lc.meta.get("AUTHOR", "unknown"),
                           "exptime_s": lc.meta.get("EXPTIME")} for lc in collection]
    lap("download")

    cb("Cleaning the data", 0.4)
    pre = preprocess(collection, cfg)
    lap("preprocess")

    cb("Searching for repeating dips", 0.5)
    lc_search = bin_lightcurve(pre.lc_flat, cfg.search_bin_minutes)
    # The search runs on a copy with strong, fast stellar variability removed;
    # depths and features below are measured on the untouched light curve.
    result = search(lc_search, cfg)
    lc_detect, prewhitened = prewhiten(lc_search, cfg)
    if prewhitened:
        alt = search(lc_detect, cfg)
        sde_without = result.signal.sde
        used = (alt.signal.detected(cfg)
                and alt.signal.sde - sde_without >= cfg.prewhiten_min_sde_gain)
        log.info("Prewhitened search: P=%.5f SDE %.1f vs %.1f without -> %s", alt.signal.period,
                 alt.signal.sde, sde_without, "used" if used else "not used")
        if used:
            result = alt
        else:
            lc_detect = lc_search
        report.extra["prewhitening"] = {"removed": prewhitened, "used": bool(used),
                                        "period_d": alt.signal.period, "sde": alt.signal.sde,
                                        "sde_without": sde_without}
    if cfg.use_tls:
        from src.detect.tls import search_tls, tls_available
        if tls_available():
            tls_sig = search_tls(lc_detect, cfg, info)
            log.info("TLS cross-check: P=%.5f (BLS %.5f)", tls_sig.period, result.signal.period)
            report.extra["tls"] = tls_sig.to_dict()
        else:
            log.warning("use_tls is on but transitleastsquares is not installed; BLS only")
    lap("search")

    cb("Re-cleaning around the dips", 0.7)
    pre = redetrend_masked(pre, [result.signal.ephemeris], cfg)
    lc_search = bin_lightcurve(pre.lc_flat, cfg.search_bin_minutes)
    signal = evaluate(lc_search, result.signal, cfg)
    result = SearchResult(result.periods, result.power, signal, result.grid_size_full,
                          result.grid_size_used)
    used_pw = report.extra.get("prewhitening", {}).get("used", False)
    results = (multi_search(prewhiten(lc_search, cfg)[0] if used_pw else lc_search, cfg,
                            first=result) if cfg.max_signals > 1 else [result])
    report.preprocessing = dict(pre.stats, prewhitened=prewhitened if used_pw else [],
                                prewhitening_tried=prewhitened)
    lap("redetrend")

    cb("Checking whether it's really a planet", 0.8)
    search_cadence = float(np.median(np.diff(lc_search.time.value)))
    features, details = extract_features(pre, signal, info, search_cadence)
    lap("features")

    cb("Classifying the signal", 0.9)
    report.triage = classify(features, cfg)
    lap("triage")

    report.signals = [r.signal.to_dict() for r in results]
    report.signals[0]["detected"] = bool(signal.detected(cfg))
    report.features = features
    report.physical = physical_parameters(signal, info)
    report.plot_data = _plot_data(pre, result, signal, details, cfg)
    report.timings = dict(timings, total_s=round(time.perf_counter() - t_start, 2))
    cb("Writing your report", 1.0)
    report._objects = {"pre": pre, "result": result}  # type: ignore[attr-defined]
    return report
