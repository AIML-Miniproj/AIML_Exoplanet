"""Run the pipeline on every training star (parallel, resumable, time-limited).

    python scripts/build_features.py [--labels PATH] [--limit N] [--n-jobs 2]

Reads data/labels/training_set.parquet (from scripts/select_stars.py) if it
exists, else data/labels/labels.parquet. Writes one JSON per star under
data/processed/features/, then data/processed/features.parquet and
data/processed/features_summary.json.

A run estimated at more than 30 minutes refuses to start unless --confirm-long
is given.
"""

# ── In plain words ───────────────────────────────────────────────────────────
# DATASET STEP 3 runner: builds data/processed/features.parquet (the training table).
# Run:    python scripts/build_features.py --n-jobs 3 --confirm-long
# Reads:  data/labels/training_set.parquet (which stars, which sectors)
# Writes: one JSON per star in data/processed/features/ and the combined
#         data/processed/features.parquet (see src/triage/dataset.py).
# Next:   python scripts/train_triage.py trains the model on that table.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace  # noqa: E402

import pandas as pd  # noqa: E402

from config import CONFIG  # noqa: E402
from src.ingest.catalogs import write_spoc_index_subset  # noqa: E402
from src.ingest.resolver import prefetch_tic  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.triage.dataset import build_features, estimate_minutes, summarize  # noqa: E402

log = logging.getLogger("build_features")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=None,
                    help="label table (default: training_set.parquet, else labels.parquet)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N stars of the table (e.g. 5 for a pilot)")
    ap.add_argument("--preset", default="training")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help=f"parallel workers (default {CONFIG.n_jobs}, max {CONFIG.max_n_jobs})")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--feature-set", default="features",
                    help="output folder under data/processed (e.g. fresh_eb for a check set)")
    ap.add_argument("--confirm-long", action="store_true",
                    help=f"allow runs estimated above {CONFIG.max_job_minutes:.0f} minutes")
    ap.add_argument("--sector-source", default="spoc_index", choices=["spoc_index", "search"],
                    help="spoc_index: SPOC 2-min files from MAST's bulk lists (fast, v1 policy)")
    args = ap.parse_args()
    setup_logging()
    cfg = replace(CONFIG, sector_source=args.sector_source, feature_set=args.feature_set)

    path = args.labels
    if path is None:
        sel = CONFIG.paths.labels / "training_set.parquet"
        path = sel if sel.exists() else CONFIG.paths.labels / "labels.parquet"
    labels = pd.read_parquet(path)
    if args.limit:
        labels = labels.head(args.limit)
    log.info("Label table %s: %d stars", path.name, len(labels))

    workers = cfg.workers(args.n_jobs)
    minutes = estimate_minutes(len(labels), workers, cfg)
    log.info("Estimated time: about %.0f min with %d workers", minutes, workers)
    if minutes > CONFIG.max_job_minutes and not args.confirm_long:
        log.error("Estimated %.0f min is above the %.0f min limit. Re-run with --confirm-long "
                  "once this has been approved.", minutes, CONFIG.max_job_minutes)
        sys.exit(2)

    if args.sector_source == "spoc_index":
        # workers load a small index of just these stars, not the full ~600k-star one
        cfg = replace(cfg, spoc_index_file=write_spoc_index_subset(labels["tic_id"], cfg))
    prefetch_tic(labels["tic_id"], cfg)
    df = build_features(labels, cfg=cfg, preset=args.preset, n_jobs=args.n_jobs,
                        retry_failed=args.retry_failed)
    out = CONFIG.paths.processed
    df.to_parquet(out / f"{args.feature_set}.parquet", index=False)
    summary = summarize(df)
    (out / f"{args.feature_set}_summary.json").write_text(json.dumps(summary, indent=1,
                                                                     default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
