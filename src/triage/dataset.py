"""Run the real pipeline on every labelled star and collect its features.

Features always come from the pipeline, never from catalog values. Each star's
result is written to its own JSON file, so an interrupted run resumes where it
stopped. A recovery check records whether the detected period matches the
catalog period (within 1%, also at 1/2x, 2x and 3x).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, Config
from src.vetting.features import FEATURE_NAMES

log = logging.getLogger(__name__)

RECOVERY_RATIOS = (1.0, 0.5, 2.0, 3.0)
RECOVERY_TOL = 0.01


def recovered(period: float, catalog_periods: list[float],
              ratios=RECOVERY_RATIOS, tol: float = RECOVERY_TOL) -> tuple[bool, float]:
    """True if ``period`` matches any catalog period (or a simple multiple). Returns the ratio."""
    if not np.isfinite(period):
        return False, float("nan")
    for cp in catalog_periods:
        if not np.isfinite(cp) or cp <= 0:
            continue
        for r in ratios:
            if abs(period / (cp * r) - 1) < tol:
                return True, r
    return False, float("nan")


def _parse_periods(s: Any) -> list[float]:
    if not isinstance(s, str) or not s:
        return []
    return [float(x) for x in s.split(";") if x]


def feature_dir(cfg: Config) -> Path:
    return cfg.paths.processed / "features"


def run_one(row: dict[str, Any], preset: str, cfg: Config, out_dir: Path) -> dict[str, Any]:
    """Pipeline + recovery check for one labelled star. Never raises."""
    from src.errors import PipelineError
    from src.log import setup_logging
    from src.pipeline import analyze_target

    setup_logging("WARNING")
    tic = int(row["tic_id"])
    out = {"tic_id": tic, "label": row["label"], "source": row["source"],
           "catalog_period": row.get("catalog_period"), "status": "ok", "reason": ""}
    t0 = time.perf_counter()
    try:
        rep = analyze_target(f"TIC {tic}", preset=preset, cfg=cfg)
        sig = rep.signals[0]
        out.update({f: rep.features.get(f) for f in FEATURE_NAMES})
        out.update({
            "det_period": sig["period"], "det_t0": sig["t0"], "det_duration": sig["duration"],
            "detected": sig.get("detected"), "n_sectors_used": len(rep.sectors),
            "sectors": ",".join(str(s["sector"]) for s in rep.sectors),
            "authors": ",".join(sorted({s["author"] for s in rep.sectors})),
        })
        cps = _parse_periods(row.get("catalog_periods"))
        ok, ratio = recovered(sig["period"], cps)
        out["recovered"], out["recovered_ratio"] = ok, ratio
        if row["label"] == "VARIABLE":
            ok_ls, r_ls = recovered(rep.features.get("ls_period_1", np.nan), cps,
                                    ratios=(1.0, 0.5, 2.0))
            out["recovered_ls"], out["recovered_ls_ratio"] = ok_ls, r_ls
    except PipelineError as e:
        out.update(status=type(e).__name__, reason=str(e)[:300])
    except Exception as e:  # noqa: BLE001 - one bad star must not stop the batch
        import traceback

        frames = traceback.extract_tb(e.__traceback__)
        where = next((f"{Path(f.filename).name}:{f.lineno}" for f in reversed(frames)
                      if "Exoplanet_Detection" in f.filename and "site-packages" not in f.filename),
                     "?")
        out.update(status="error", reason=f"{type(e).__name__} at {where}: {e}"[:300])
    out["runtime_s"] = round(time.perf_counter() - t0, 1)
    (out_dir / f"tic_{tic}.json").write_text(json.dumps(out, default=float), encoding="utf-8")
    return out


def build_features(labels: pd.DataFrame, cfg: Config = CONFIG, preset: str = "quick",
                   n_jobs: int | None = None, limit: int | None = None,
                   retry_failed: bool = False) -> pd.DataFrame:
    """Run the pipeline on every label (parallel, resumable) and return all rows."""
    from joblib import Parallel, delayed
    from tqdm import tqdm

    out_dir = feature_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = labels if limit is None else labels.groupby("label").head(limit)

    def done(tic: int) -> bool:
        p = out_dir / f"tic_{tic}.json"
        if not p.exists():
            return False
        if not retry_failed:
            return True
        return json.loads(p.read_text(encoding="utf-8")).get("status") == "ok"

    pending = [r for r in todo.to_dict("records") if not done(int(r["tic_id"]))]
    log.info("%d labelled stars, %d already done, %d to run with preset %r",
             len(todo), len(todo) - len(pending), len(pending), preset)
    if pending:
        jobs = Parallel(n_jobs=n_jobs or cfg.n_jobs, return_as="generator_unordered",
                        batch_size=1)(delayed(run_one)(r, preset, cfg, out_dir) for r in pending)
        n_fail = 0
        for res in tqdm(jobs, total=len(pending), desc="features", mininterval=10):
            n_fail += res["status"] != "ok"
        log.info("Finished: %d failures out of %d", n_fail, len(pending))
    return collect(todo, cfg)


def collect(labels: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    """Load all per-star results for the given labels into one table."""
    out_dir = feature_dir(cfg)
    rows = []
    for tic in labels["tic_id"]:
        p = out_dir / f"tic_{int(tic)}.json"
        if p.exists():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Counts, recovery rates and failure reasons per class."""
    out: dict[str, Any] = {}
    for cls, g in df.groupby("label"):
        ok = g[g["status"] == "ok"]
        with_cat = ok[ok["catalog_period"].notna()]
        entry = {
            "n": int(len(g)), "n_ok": int(len(ok)),
            "n_failed": int(len(g) - len(ok)),
            "detected_frac": round(float(ok["detected"].mean()), 3) if len(ok) else None,
            "n_with_catalog_period": int(len(with_cat)),
            "recovery_rate": (round(float(with_cat["recovered"].mean()), 3)
                              if len(with_cat) else None),
            "recovered_at_ratio": (with_cat.loc[with_cat["recovered"], "recovered_ratio"]
                                   .value_counts().to_dict() if len(with_cat) else {}),
            "median_runtime_s": float(ok["runtime_s"].median()) if len(ok) else None,
        }
        if cls == "VARIABLE" and "recovered_ls" in with_cat:
            entry["ls_recovery_rate"] = round(float(with_cat["recovered_ls"].mean()), 3) \
                if len(with_cat) else None
        fails = g[g["status"] != "ok"]
        entry["failure_reasons"] = (fails["status"] + ": " + fails["reason"].str.slice(0, 80)) \
            .value_counts().head(8).to_dict()
        out[cls] = entry
    return out
