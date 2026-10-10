"""NOISE results and the log of rejected / unsure signals.

NOISE and NEEDS_REVIEW results are appended to ``data/logs/recycler_log.parquet``
so they can be reviewed and reused as training examples later. Each star keeps
one row per model version (the latest run).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from config import CONFIG, Config
from src.recycler.common import KeyFact, ModuleResult, fmt, period_words

log = logging.getLogger(__name__)
LOG_FILE = "recycler_log.parquet"


def characterize(signal: dict[str, Any], features: dict[str, Any], reasons: list[str],
                 cfg: Config = CONFIG) -> ModuleResult:
    """Noise module: why the signal was not kept, in plain words."""
    sde = float(signal.get("sde", math.nan))
    detected = bool(signal.get("detected", False))
    summary = ("We did not find a convincing repeating signal: the small dips are most likely "
               "random noise or effects of the telescope, not something orbiting the star.")
    facts = [
        KeyFact("Strongest repeat found", fmt(float(signal.get("period", math.nan)), 3), "days",
                "The best period the dip search found, even though it was not convincing."),
        KeyFact("Dip depth", fmt(float(signal.get("depth_ppm", math.nan)), 0), "ppm",
                "Parts per million of the star's light (10,000 ppm = 1%)."),
        KeyFact("Detection strength", f"{fmt(sde, 1)} (threshold {cfg.sde_min:g})", "",
                "How much the best period stands out from all other trial periods."),
        KeyFact("Passed the detection test", "yes" if detected else "no", "",
                "Whether the dip is strong and regular enough to count as a detection."),
    ]
    details = {"reasons": reasons,
               "strongest_period_words": period_words(float(signal.get("period", math.nan)))}
    return ModuleResult("noise", summary, facts, details)


def log_path(cfg: Config = CONFIG) -> Path:
    return cfg.paths.logs / LOG_FILE


def append(report: dict[str, Any], cfg: Config = CONFIG) -> Path | None:
    """Append one NOISE / NEEDS_REVIEW result to the recycler log."""
    routing = report.get("routing", {})
    if routing.get("primary_class") not in ("NOISE", "NEEDS_REVIEW"):
        return None
    sig = (report.get("signals") or [{}])[0]
    row = {
        "logged_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tic_id": report.get("tic_id"), "target": report.get("target"),
        "preset": report.get("preset"), "primary_class": routing.get("primary_class"),
        "secondary_class": routing.get("secondary_class"),
        "confidence": routing.get("confidence_label"),
        "top_probability": routing.get("top_probability"),
        "probabilities": json.dumps(routing.get("probabilities", {})),
        "reasons": json.dumps(routing.get("reasons", [])),
        "period_d": sig.get("period"), "depth_ppm": sig.get("depth_ppm"), "sde": sig.get("sde"),
        "detected": sig.get("detected"),
        "model_version": (report.get("triage") or {}).get("model_version"),
        "sectors": ",".join(str(s.get("sector")) for s in report.get("sectors", [])),
    }
    path = log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    df = pd.concat([pd.read_parquet(path), new], ignore_index=True) if path.exists() else new
    # one row per star and model version: re-running a star replaces its entry
    df = df.drop_duplicates(["tic_id", "model_version"], keep="last").reset_index(drop=True)
    df.to_parquet(path, index=False)
    log.info("Logged %s result for %s to %s", row["primary_class"], row["target"], path.name)
    return path
