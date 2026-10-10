"""Features describing how the star's brightness varies (from ``lc_norm``).

These use the normalised but *not* flattened light curve, so slow and smooth
variations such as starspots, pulsations or the stretching of a star by a close
companion are still present.
"""

from __future__ import annotations

from typing import Any

import lightkurve as lk
import numpy as np
from astropy.timeseries import LombScargle
from scipy import stats as sstats

from src.detect.bls import Signal, distinct_peaks
from src.preprocess.preprocessor import bin_lightcurve
from src.vetting.common import NAN, phase, phase_time

LS_BIN_MIN = 10.0
LS_PMIN_D = 1 / 48          # 30 minutes
LS_PMAX_D = 30.0


def _arrays(lc: lk.LightCurve):
    t = np.asarray(lc.time.value, float)
    f = np.asarray(lc.flux.value, float)
    sec = np.asarray(lc["sector"].value) if "sector" in lc.colnames else np.zeros(t.size)
    ok = np.isfinite(t) & np.isfinite(f)
    return t[ok], f[ok] - np.median(f[ok]), sec[ok]


def lomb_scargle(t: np.ndarray, f: np.ndarray, pmin: float = LS_PMIN_D,
                 pmax: float = LS_PMAX_D, spp: int = 5):
    baseline = t.max() - t.min()
    pmax = min(pmax, baseline / 2)
    ls = LombScargle(t, f)
    freq = np.linspace(1 / pmax, 1 / pmin, int(spp * baseline / pmin) + 10)
    if freq.size > 400_000:
        freq = np.linspace(1 / pmax, 1 / pmin, 400_000)
    power = ls.power(freq, method="fast")
    return ls, freq, power


def fourier_params(t: np.ndarray, f: np.ndarray, period: float) -> dict[str, float]:
    """Amplitude ratios and phase differences of the first three harmonics."""
    w = 2 * np.pi / period
    X = np.column_stack([np.ones_like(t)] + [fn(k * w * t) for k in (1, 2, 3)
                                             for fn in (np.sin, np.cos)])
    c, *_ = np.linalg.lstsq(X, f, rcond=None)
    A = [np.hypot(c[1 + 2 * k], c[2 + 2 * k]) for k in range(3)]
    ph = [np.arctan2(c[2 + 2 * k], c[1 + 2 * k]) for k in range(3)]
    if A[0] <= 0:
        return {"fourier_R21": NAN, "fourier_R31": NAN, "fourier_phi21": NAN,
                "fourier_phi31": NAN}
    return {"fourier_R21": float(A[1] / A[0]), "fourier_R31": float(A[2] / A[0]),
            "fourier_phi21": float((ph[1] - 2 * ph[0]) % (2 * np.pi)),
            "fourier_phi31": float((ph[2] - 3 * ph[0]) % (2 * np.pi))}


def acf_period(t: np.ndarray, f: np.ndarray, sec: np.ndarray,
               step_d: float = 1 / 48) -> tuple[float, float]:
    """Period from the first strong peak of the autocorrelation, averaged over sectors."""
    acfs = []
    for s in np.unique(sec):
        m = sec == s
        if m.sum() < 50:
            continue
        ts, fs = t[m], f[m] - np.mean(f[m])
        n = int((ts.max() - ts.min()) / step_d) + 1
        grid = np.zeros(n)
        cnt = np.zeros(n)
        idx = np.clip(((ts - ts.min()) / step_d).astype(int), 0, n - 1)
        np.add.at(grid, idx, fs)
        np.add.at(cnt, idx, 1)
        grid = np.where(cnt > 0, grid / np.maximum(cnt, 1), 0.0)
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        spec = np.fft.rfft(grid, nfft)
        ac = np.fft.irfft(spec * np.conj(spec), nfft)[: n // 2]
        if ac[0] <= 0:
            continue
        acfs.append(ac / ac[0])
    if not acfs:
        return NAN, NAN
    L = min(len(a) for a in acfs)
    ac = np.mean([a[:L] for a in acfs], axis=0)
    neg = np.flatnonzero(ac < 0)
    if neg.size == 0:
        return NAN, NAN
    start = neg[0]
    if L - start < 3:
        return NAN, NAN
    j = start + int(np.argmax(ac[start:]))
    if ac[j] <= 0 or j in (start, L - 1):
        return NAN, NAN
    return float(j * step_d), float(ac[j])


def phase_curve(t: np.ndarray, f: np.ndarray, signal: Signal) -> dict[str, float]:
    """Brightness change around the orbit, outside the dips.

    ``cos 2phi`` term: a star stretched by a close companion looks dimmer twice
    per orbit (ellipsoidal). ``cos phi`` term: light reflected by the companion.
    """
    P, t0, dur = signal.period, signal.t0, signal.duration
    ph = phase(t, P, t0)
    pt = np.abs(phase_time(t, P, t0))
    sec_dist = np.abs((ph - 0.5 + 0.5) % 1 - 0.5) * P
    keep = (pt > dur) & (sec_dist > dur)
    out = {"ellip_amp_ppm": NAN, "ellip_snr": NAN, "refl_amp_ppm": NAN}
    if keep.sum() < 50:
        return out
    x = 2 * np.pi * ph[keep]
    X = np.column_stack([np.ones(x.size), np.cos(x), np.sin(x), np.cos(2 * x), np.sin(2 * x)])
    y = f[keep]
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ c
    cov = np.linalg.pinv(X.T @ X) * np.var(resid)
    err2 = np.sqrt(max(cov[3, 3], 0))
    out["refl_amp_ppm"] = float(-c[1] * 1e6)
    out["ellip_amp_ppm"] = float(-c[3] * 1e6)
    out["ellip_snr"] = float(-c[3] / err2) if err2 > 0 else NAN
    return out


def variability_features(lc_norm: lk.LightCurve, signal: Signal | None,
                         ) -> tuple[dict[str, float], dict[str, Any]]:
    lc = bin_lightcurve(lc_norm, LS_BIN_MIN)
    t, f, sec = _arrays(lc)
    feats: dict[str, float] = {}
    details: dict[str, Any] = {}

    feats["rms_ppm"] = float(np.std(f) * 1e6)
    lo, hi = np.percentile(f, [5, 95])
    feats["amp_p2p_ppm"] = float((hi - lo) * 1e6)
    feats["skew"] = float(sstats.skew(f))
    feats["kurtosis"] = float(sstats.kurtosis(f))

    ls, freq, power = lomb_scargle(t, f)
    periods = 1 / freq[::-1]
    pw = power[::-1]
    idx = distinct_peaks(periods, pw, 3)
    for k in range(3):
        if k < len(idx):
            feats[f"ls_period_{k + 1}"] = float(periods[idx[k]])
            feats[f"ls_power_{k + 1}"] = float(pw[idx[k]])
        else:
            feats[f"ls_period_{k + 1}"] = NAN
            feats[f"ls_power_{k + 1}"] = NAN
    fap = float(ls.false_alarm_probability(feats["ls_power_1"], method="baluev",
                                           minimum_frequency=freq[0],
                                           maximum_frequency=freq[-1]))
    feats["ls_log10_fap"] = float(np.log10(max(fap, 1e-300)))
    step = max(1, len(periods) // 4000)
    details["ls"] = {"period": periods[::step].tolist(), "power": pw[::step].tolist()}

    p1 = feats["ls_period_1"]
    if signal is not None and np.isfinite(p1):
        ratios = [p1 / (r * signal.period) for r in (0.5, 1.0, 2.0)]
        feats["ls_period_matches_bls"] = float(any(abs(x - 1) < 0.01 for x in ratios))
    else:
        feats["ls_period_matches_bls"] = NAN

    feats["acf_period"], feats["acf_peak_height"] = acf_period(t, f, sec)
    feats.update(fourier_params(t, f, p1) if np.isfinite(p1) else
                 {"fourier_R21": NAN, "fourier_R31": NAN, "fourier_phi21": NAN,
                  "fourier_phi31": NAN})

    # Stability of the main period from sector to sector.
    per_sector = []
    for s in np.unique(sec):
        m = sec == s
        span = t[m].max() - t[m].min() if m.sum() > 50 else 0
        if span < 2 * p1 or not np.isfinite(p1):
            continue
        ls_s = LombScargle(t[m], f[m])
        fr = np.linspace(1 / (p1 * 1.5), 1 / (p1 / 1.5), 2000)
        per_sector.append(1 / fr[np.argmax(ls_s.power(fr))])
    feats["ls_period_sector_std"] = (float(np.std(per_sector) / np.mean(per_sector))
                                     if len(per_sector) >= 2 else NAN)

    if signal is not None:
        feats.update(phase_curve(t, f, signal))
    else:
        feats.update({"ellip_amp_ppm": NAN, "ellip_snr": NAN, "refl_amp_ppm": NAN})
    return feats, details
