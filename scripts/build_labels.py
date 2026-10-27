"""Build data/labels/labels.parquet from public catalogs.

    python scripts/build_labels.py [--limit-per-class 400] [--refresh]
"""

# ── In plain words ───────────────────────────────────────────────────────────
# DATASET STEP 1 (run this first): catalogs -> labelled candidate list.
# Run:    python scripts/build_labels.py --limit-per-class 400
# Reads:  the public catalogs (downloaded by src/ingest/catalogs.py, then cached)
# Writes: data/labels/labels.parquet       up to 400 candidates per class
#         data/labels/labels_summary.json  how many stars passed each rule
#         data/labels/conflicts.csv        stars found in more than one catalog
#         data/labels/holdout_candidates.parquet  unconfirmed candidates (never trained on)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG  # noqa: E402
from src.ingest import catalogs  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.triage.labels import (add_variable_periods, build_labels,  # noqa: E402
                               crossmatch_gaia_to_tic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-per-class", type=int, default=CONFIG.limit_per_class)
    ap.add_argument("--tmag-max", type=float, default=12.5)
    ap.add_argument("--refresh", action="store_true", help="re-download all catalogs")
    args = ap.parse_args()
    setup_logging()
    CONFIG.paths.ensure()

    toi = catalogs.fetch_toi(refresh=args.refresh)
    ebs = catalogs.fetch_eb_catalog(refresh=args.refresh)
    targets = catalogs.fetch_2min_targets(refresh=args.refresh)
    gaia = catalogs.fetch_gaia_variables(refresh=args.refresh)
    gaia_x = crossmatch_gaia_to_tic(gaia, targets)
    print(f"Gaia variables matched to 2-min TICs: {len(gaia_x)} of {len(gaia)}")
    print(gaia_x["best_class_name"].value_counts().to_string())

    ls = build_labels(toi, ebs, gaia_x, targets, limit_per_class=args.limit_per_class,
                      tmag_max=args.tmag_max)
    var_ids = [int(x) for x in
               ls.labels.loc[ls.labels["label"] == "VARIABLE", "gaia_source_id"].dropna()]
    ls.labels = add_variable_periods(ls.labels, catalogs.fetch_gaia_periods(var_ids))
    n_per = ls.labels.loc[ls.labels["label"] == "VARIABLE", "catalog_period"].notna().sum()
    ls.summary["variables_with_gaia_period"] = int(n_per)
    out = CONFIG.paths.labels
    ls.labels.to_parquet(out / "labels.parquet", index=False)
    ls.holdout.to_parquet(out / "holdout_candidates.parquet", index=False)
    ls.conflicts.to_csv(out / "conflicts.csv", index=False)
    (out / "labels_summary.json").write_text(json.dumps(ls.summary, indent=1, default=str))
    print(json.dumps(ls.summary, indent=1, default=str))


if __name__ == "__main__":
    main()
