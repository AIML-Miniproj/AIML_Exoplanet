"""Run the real pipeline on every labelled star and collect its features.

Features always come from the pipeline, never from catalog values. Each star's
result is written to its own JSON file, so an interrupted run resumes where it
stopped.

Recovery check:
- PLANET / EB: the detected period matches the catalog period within 1%
  (also at 1/2x, 2x and 3x).
- VARIABLE: the periodogram finds a significant period (false-alarm probability
  below 1%). Where a catalog period exists, whether it matches (1x, 1/2x, 2x) is
  reported separately.
- NOISE: no catalog period, so no recovery check.
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
VARIABLE_LOG10_FAP_MAX = -2.0      # variable "recovered": periodogram peak with FAP < 1%


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
    """Pipeline + recovery check for one labelled star. Never raises.

    Writes its result to ``out_dir/tic_<id>.json``; ``run_with_timeout`` reads it back.
    """
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
        if row["label"] == "VARIABLE":
            # A variable counts as recovered when the periodogram finds a significant
            # period at all; most have no catalog period to compare with.
            fap = rep.features.get("ls_log10_fap", np.nan)
            out["recovered"] = bool(np.isfinite(fap) and fap < VARIABLE_LOG10_FAP_MAX)
            out["recovered_ratio"] = float("nan")
            ok_ls, r_ls = recovered(rep.features.get("var_period_d", np.nan), cps,
                                    ratios=(1.0, 0.5, 2.0))
            out["period_match_catalog"] = ok_ls if cps else None
            out["period_match_ratio"] = r_ls
        else:
            ok, ratio = recovered(sig["period"], cps)
            out["recovered"], out["recovered_ratio"] = ok, ratio
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


def _failure(row: dict[str, Any], status: str, reason: str, runtime_s: float,
             out_dir: Path) -> dict[str, Any]:
    """Record a star whose worker process never wrote a result."""
    out = {"tic_id": int(row["tic_id"]), "label": row["label"], "source": row["source"],
           "catalog_period": row.get("catalog_period"), "status": status, "reason": reason,
           "runtime_s": round(runtime_s, 1)}
    (out_dir / f"tic_{out['tic_id']}.json").write_text(json.dumps(out, default=float),
                                                       encoding="utf-8")
    return out


def run_with_timeout(rows: list[dict[str, Any]], preset: str, cfg: Config, out_dir: Path,
                     n_jobs: int, timeout_s: float, target=None) -> list[dict[str, Any]]:
    """Run ``run_one`` for each row in its own process, at most ``n_jobs`` at a time.

    A star still running after ``timeout_s`` is killed and recorded as "timeout".
    A separate process per star is the only way to stop a run stuck inside
    compiled code (e.g. the BLS search) and to hand its memory back right away.
    """
    import multiprocessing as mp

    target = target or run_one
    ctx = mp.get_context("spawn")
    queue = list(rows)
    running: dict[Any, tuple[dict[str, Any], float]] = {}
    results: list[dict[str, Any]] = []
    while queue or running:
        while queue and len(running) < n_jobs:
            row = queue.pop(0)
            (out_dir / f"tic_{int(row['tic_id'])}.json").unlink(missing_ok=True)
            proc = ctx.Process(target=target, args=(row, preset, cfg, out_dir), daemon=True)
            proc.start()
            running[proc] = (row, time.monotonic())
        time.sleep(0.2)
        for proc, (row, start) in list(running.items()):
            elapsed = time.monotonic() - start
            if proc.is_alive():
                if elapsed <= timeout_s:
                    continue
                proc.kill()
                proc.join()
                log.warning("TIC %s: timeout after %.0f s, skipped", row["tic_id"], elapsed)
                results.append(_failure(row, "timeout",
                                        f"no result within {timeout_s:.0f} s", elapsed, out_dir))
            else:
                proc.join()
                path = out_dir / f"tic_{int(row['tic_id'])}.json"
                if path.exists():
                    results.append(json.loads(path.read_text(encoding="utf-8")))
                else:
                    results.append(_failure(row, "crashed",
                                            f"worker exited with code {proc.exitcode}",
                                            elapsed, out_dir))
            del running[proc]
    return results


def estimate_minutes(n_stars: int, n_jobs: int, cfg: Config = CONFIG,
                     default_s: float = 60.0) -> float:
    """Rough wall-clock estimate from earlier per-star runtimes (or ``default_s``)."""
    times = []
    for p in feature_dir(cfg).glob("tic_*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") == "ok" and r.get("runtime_s"):
            times.append(float(r["runtime_s"]))
    per_star = float(np.median(times)) if times else default_s
    return n_stars * per_star / max(n_jobs, 1) / 60


def build_features(labels: pd.DataFrame, cfg: Config = CONFIG, preset: str = "training",
                   n_jobs: int | None = None, limit: int | None = None,
                   retry_failed: bool = False) -> pd.DataFrame:
    """Run the pipeline on every label (parallel, resumable) and return all rows.

    Stars are processed in chunks of ``cfg.chunk_size`` with a progress line after
    each chunk. Each star is killed after ``cfg.star_timeout_s``.
    """
    out_dir = feature_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = labels if limit is None else labels.groupby("label").head(limit)
    workers = cfg.workers(n_jobs)
    if n_jobs is not None and workers != n_jobs:
        log.warning("Requested %d workers; using %d (limit %d)", n_jobs, workers,
                    cfg.max_n_jobs)

    def done(tic: int) -> bool:
        p = out_dir / f"tic_{tic}.json"
        if not p.exists():
            return False
        if not retry_failed:
            return True
        return json.loads(p.read_text(encoding="utf-8")).get("status") == "ok"

    pending = [r for r in todo.to_dict("records") if not done(int(r["tic_id"]))]
    log.info("%d labelled stars, %d already done, %d to run with preset %r, %d workers, "
             "%.0f s timeout per star", len(todo), len(todo) - len(pending), len(pending),
             preset, workers, cfg.star_timeout_s)
    n_done = n_fail = 0
    start = time.monotonic()
    for i in range(0, len(pending), cfg.chunk_size):
        chunk = pending[i:i + cfg.chunk_size]
        res = run_with_timeout(chunk, preset, cfg, out_dir, workers, cfg.star_timeout_s)
        n_done += len(res)
        n_fail += sum(r["status"] != "ok" for r in res)
        per_star = (time.monotonic() - start) / n_done
        eta_min = per_star * (len(pending) - n_done) / 60
        log.info("Progress: %d/%d done, %d failed, %.1f s per star (wall clock), "
                 "ETA %.1f min", n_done, len(pending), n_fail, per_star, eta_min)
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
        # variables: recovery = significant periodogram peak, checkable for every star
        checkable = ok if cls == "VARIABLE" else with_cat
        entry = {
            "n": int(len(g)), "n_ok": int(len(ok)),
            "n_failed": int(len(g) - len(ok)),
            "detected_frac": round(float(ok["detected"].mean()), 3) if len(ok) else None,
            "n_with_catalog_period": int(len(with_cat)),
            "n_recovery_checked": int(len(checkable)),
            "recovery_rate": (round(float(checkable["recovered"].astype(bool).mean()), 3)
                              if len(checkable) and cls != "NOISE" else None),
            "recovered_at_ratio": (with_cat.loc[with_cat["recovered"].astype(bool),
                                                "recovered_ratio"].value_counts().to_dict()
                                   if len(with_cat) and cls != "VARIABLE" else {}),
            "median_runtime_s": float(ok["runtime_s"].median()) if len(ok) else None,
        }
        if cls == "VARIABLE" and "period_match_catalog" in with_cat:
            entry["catalog_period_match"] = (
                f"{int(with_cat['period_match_catalog'].astype(bool).sum())}"
                f"/{len(with_cat)}")
        fails = g[g["status"] != "ok"]
        entry["failure_reasons"] = (fails["status"] + ": " + fails["reason"].str.slice(0, 80)) \
            .value_counts().head(8).to_dict()
        out[cls] = entry
    return out
