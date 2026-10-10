"""Features comparing the dip across TESS sectors.

A real planet's dip has the same depth in every sector. A dip caused by a
neighbouring star leaking into the aperture can change depth from sector to
sector, because the aperture and the amount of light from neighbours change.
"""

from __future__ import annotations

from typing import Any

import lightkurve as lk
import numpy as np

from src.detect.bls import Signal
from src.vetting.common import NAN, box_depth, phase_time


def per_sector_depths(lc_flat: lk.LightCurve, signal: Signal,
                      beta: float = 1.0) -> list[dict[str, Any]]:
    """Depth of the dip measured separately in each sector."""
    t = np.asarray(lc_flat.time.value, float)
    f = np.asarray(lc_flat.flux.value, float)
    sec = np.asarray(lc_flat["sector"].value)
    pt = np.abs(phase_time(t, signal.period, signal.t0))
    dur = signal.duration
    rows = []
    for s in np.unique(sec):
        m = sec == s
        inside = m & (pt < 0.4 * dur)                       # flat part of the dip
        local_out = m & (pt > 0.75 * dur) & (pt < 3 * dur)
        depth, err, n_in = box_depth(f, inside, local_out, beta)
        rows.append({"sector": int(s), "depth": depth, "depth_err": err, "n_in": n_in,
                     "has_transit": bool(n_in >= 3 and np.isfinite(depth))})
    return rows


def sector_features(lc_flat: lk.LightCurve, signal: Signal,
                    sector_stats: list[dict[str, Any]] | None = None, beta: float = 1.0,
                    ) -> tuple[dict[str, float], dict[str, Any]]:
    rows = per_sector_depths(lc_flat, signal, beta)
    with_t = [r for r in rows if r["has_transit"]]
    d = np.array([r["depth"] for r in with_t])
    e = np.array([r["depth_err"] for r in with_t])

    feats: dict[str, float] = {
        "n_sectors": float(len(rows)),
        "n_sectors_with_transit": float(len(with_t)),
        "frac_sectors_with_transit": len(with_t) / len(rows) if rows else NAN,
        "sector_depth_cv": NAN, "sector_depth_chi2": NAN, "max_sector_depth_ratio": NAN,
    }
    if len(with_t) >= 2 and np.all(e > 0):
        w = 1 / e**2
        mean = float(np.sum(w * d) / np.sum(w))
        feats["sector_depth_chi2"] = float(np.sum((d - mean) ** 2 * w) / (len(d) - 1))
        if mean > 0:
            feats["sector_depth_cv"] = float(np.std(d) / mean)
            pos = d[d > 0]
            if pos.size:
                feats["max_sector_depth_ratio"] = float(pos.max() / np.median(pos))

    crowd = [s.get("crowdsap") for s in (sector_stats or []) if s.get("crowdsap") is not None]
    crowd = np.array([c for c in crowd if np.isfinite(c)], float)
    feats["crowdsap_min"] = float(crowd.min()) if crowd.size else NAN
    feats["crowdsap_range"] = float(np.ptp(crowd)) if crowd.size else NAN
    return feats, {"per_sector": rows}
