"""Precomputed example reports for the app (instant demo loading).

``build_one`` runs the full pipeline on one star and writes a compact report to
``data/examples/demo/<tic>.json``: plot arrays are thinned and rounded so all
examples together stay small enough to commit. Used by scripts/build_examples.py,
one star per worker process.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PLOT_LIMITS = {"lc_norm": 4000, "lc_flat": 4000, "fold": 3000, "fold_zoom": 2000,
               "periodogram": 2000}


def _thin(arrays: dict[str, list], n: int) -> dict[str, list]:
    lens = [len(v) for v in arrays.values() if isinstance(v, list)]
    if not lens or max(lens) <= n:
        return arrays
    step = int(math.ceil(max(lens) / n))
    return {k: (v[::step] if isinstance(v, list) and len(v) == max(lens) else v)
            for k, v in arrays.items()}


def _round(obj: Any, sig: int = 7) -> Any:
    if isinstance(obj, float):
        return float(f"{obj:.{sig}g}") if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _round(v, sig) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, sig) for v in obj]
    return obj


def compact(report: dict[str, Any]) -> dict[str, Any]:
    """Thin the plot arrays (display only; the analysis used every point)."""
    pd_ = dict(report.get("plot_data") or {})
    for key, n in PLOT_LIMITS.items():
        if isinstance(pd_.get(key), dict):
            pd_[key] = _thin(pd_[key], n)
    report = dict(report, plot_data=pd_)
    return _round(report)


def build_one(row: dict[str, Any], preset: str, cfg, out_dir: Path) -> None:
    """Analyse one example star and write its report (and a status file for the runner)."""
    from src.log import setup_logging
    from src.pipeline import analyze_target

    setup_logging("WARNING")
    tic = int(row["tic_id"])
    status = {"tic_id": tic, "label": row["group"], "status": "ok", "reason": ""}
    start = time.perf_counter()
    try:
        rep = analyze_target(f"TIC {tic}", preset=preset, cfg=replace(cfg, recycler_log=False))
        d = compact(rep.to_dict())
        d["example"] = {k: row[k] for k in ("group", "display_name", "hint")}
        d["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        demo_dir = Path(row["demo_dir"])
        demo_dir.mkdir(parents=True, exist_ok=True)
        (demo_dir / f"{tic}.json").write_text(json.dumps(d, separators=(",", ":")),
                                               encoding="utf-8")
        status.update(n_sectors=len(rep.sectors), verdict=rep.routing.get("primary_class"),
                      summary=rep.plain_summary)
    except Exception as e:  # noqa: BLE001 - one star must not stop the batch
        status.update(status="error", reason=f"{type(e).__name__}: {e}"[:300])
    status["runtime_s"] = round(time.perf_counter() - start, 1)
    (out_dir / f"tic_{tic}.json").write_text(json.dumps(status, default=float),
                                             encoding="utf-8")


def size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.glob("*.json")) / 1e6

