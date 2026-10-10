"""Run the pipeline on every labelled star (parallel, resumable).

    python scripts/build_features.py [--limit-per-class N] [--preset quick] [--n-jobs 6]

Writes one JSON per star under data/processed/features/, then
data/processed/features.parquet and data/processed/features_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace  # noqa: E402

import pandas as pd  # noqa: E402

from config import CONFIG  # noqa: E402
from src.ingest.catalogs import fetch_spoc_lc_index  # noqa: E402
from src.ingest.resolver import prefetch_tic  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.triage.dataset import build_features, summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--preset", default="quick")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--sector-source", default="spoc_index", choices=["spoc_index", "search"],
                    help="spoc_index: SPOC 2-min files from MAST's bulk lists (fast, v1 policy)")
    args = ap.parse_args()
    setup_logging()
    cfg = replace(CONFIG, sector_source=args.sector_source)

    labels = pd.read_parquet(CONFIG.paths.labels / "labels.parquet")
    if args.sector_source == "spoc_index":
        fetch_spoc_lc_index(cfg)
    prefetch_tic(labels["tic_id"], cfg)
    df = build_features(labels, cfg=cfg, preset=args.preset, n_jobs=args.n_jobs,
                        limit=args.limit_per_class, retry_failed=args.retry_failed)
    out = CONFIG.paths.processed
    df.to_parquet(out / "features.parquet", index=False)
    summary = summarize(df)
    (out / "features_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
