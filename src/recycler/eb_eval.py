"""Evaluate the EB module on catalog binaries with the data the app would use.

For each binary: load up to 10 sectors (the app's standard preset), refine the
detected period on all of them, then compare the EB module's orbital period and
binary type with the TESS EB catalog. "Before" = the period rule on the stored
features alone; "after" = the module with the alternate-eclipse check.
Used by scripts/eval_eb_module.py; one star per worker process.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


def evaluate_one(row: dict[str, Any], preset: str, cfg, out_dir: Path) -> None:
    from src.detect.bls import Signal, evaluate
    from src.ingest.fetcher import fetch_lightcurves
    from src.log import setup_logging
    from src.preprocess.preprocessor import bin_lightcurve, preprocess
    from src.recycler import eb_module

    setup_logging("WARNING")
    tic = int(row["tic_id"])
    out = {"tic_id": tic, "set": row["set"], "label": "EB", "status": "ok", "reason": ""}
    start = time.perf_counter()
    try:
        feats = json.loads(Path(row["feature_file"]).read_text(encoding="utf-8"))
        cat = eb_module.catalog_entry(tic, cfg)
        run_cfg = cfg.with_preset(preset)
        coll, sectors = fetch_lightcurves(tic, run_cfg)
        pre = preprocess(coll, run_cfg)
        lc = bin_lightcurve(pre.lc_flat, run_cfg.search_bin_minutes)
        sig0 = Signal(period=feats["det_period"], t0=feats["det_t0"],
                      duration=feats["det_duration"], depth=0.0, depth_err=1.0, power=0.0,
                      sde=0.0)
        sig = evaluate(lc, sig0, run_cfg)                      # refine on all sectors
        signal = {"period": sig.period, "t0": sig.t0, "duration": sig.duration}
        before = eb_module.true_period(feats["det_period"], feats)
        res = eb_module.characterize(
            signal, feats, pre.lc_flat.time.value, pre.lc_flat.flux.value, tic, run_cfg,
            pre.lc_norm.time.value, pre.lc_norm.flux.value)
        after = res.details["true_period"]
        cp = cat["period_d"] if cat else math.nan
        out.update(
            n_sectors=len(sectors), catalog_period=cp,
            catalog_type=cat["type"] if cat else "unknown",
            catalog_morph=cat["morph"] if cat else math.nan,
            detected_period=feats["det_period"], refined_period=sig.period,
            before_period=before.period_d, before_rule=before.rule,
            before_ok=bool(abs(before.period_d / cp - 1) < 0.01),
            after_period=after["period_d"], after_rule=after["rule"],
            after_ok=bool(abs(after["period_d"] / cp - 1) < 0.01),
            check=res.details["alternate_eclipse_check"],
            morph_features=eb_module.morphology(feats, after["period_d"])[0],
            morph_measured=res.details["morphology"],
            morph_source=res.details["morphology_source"],
            morph_inputs=res.details["morphology_inputs"])
    except Exception as e:  # noqa: BLE001 - one star must not stop the batch
        out.update(status="error", reason=f"{type(e).__name__}: {e}"[:300])
    out["runtime_s"] = round(time.perf_counter() - start, 1)
    (out_dir / f"tic_{tic}.json").write_text(json.dumps(out, default=float), encoding="utf-8")


def _clean(v):
    return None if isinstance(v, float) and not np.isfinite(v) else v


def build_rows(cfg) -> list[dict[str, Any]]:
    import pandas as pd

    rows = []
    for name, fset in (("training", "features"), ("fresh", "fresh_eb")):
        df = pd.read_parquet(cfg.paths.processed / f"{fset}.parquet")
        for tic in df[(df["label"] == "EB") & (df["status"] == "ok")]["tic_id"]:
            rows.append({"tic_id": int(tic), "set": name, "label": "EB", "source": name,
                         "feature_file": str(cfg.paths.processed / fset / f"tic_{int(tic)}.json")})
    return rows


def with_subset(cfg, tics):
    from src.ingest.catalogs import write_spoc_index_subset

    return replace(cfg, sector_source="spoc_index",
                   spoc_index_file=write_spoc_index_subset(tics, cfg))
