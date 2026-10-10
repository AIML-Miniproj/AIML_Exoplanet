"""EB module vs the TESS EB catalog on the training + fresh binaries.

    python scripts/eval_eb_module.py [--limit N] [--n-jobs 2]

Loads up to 10 sectors per star (the app's standard preset), so the first run
downloads light curves. Writes data/processed/eb_eval/tic_*.json and prints
orbital-period accuracy before/after the alternate-eclipse check, and binary-type
agreement from features vs from the module's own measurements.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import CONFIG  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.recycler.eb_eval import build_rows, evaluate_one, with_subset  # noqa: E402
from src.triage.dataset import run_with_timeout  # noqa: E402

log = logging.getLogger("eval_eb_module")
ORDER = ["detached", "semi-detached", "contact"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    setup_logging()
    out_dir = CONFIG.paths.processed / "eb_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(CONFIG)[: args.limit] if args.limit else build_rows(CONFIG)
    if not args.report_only:
        cfg = with_subset(CONFIG, [r["tic_id"] for r in rows])
        todo = [r for r in rows if not (out_dir / f"tic_{r['tic_id']}.json").exists()]
        workers = cfg.workers(args.n_jobs)
        start, done, failed = time.monotonic(), 0, 0
        for i in range(0, len(todo), cfg.chunk_size):
            res = run_with_timeout(todo[i:i + cfg.chunk_size], "standard", cfg, out_dir,
                                   workers, cfg.star_timeout_s, target=evaluate_one)
            done += len(res)
            failed += sum(r["status"] != "ok" for r in res)
            per = (time.monotonic() - start) / done
            log.info("Progress: %d/%d done, %d failed, %.1f s per star, ETA %.1f min", done,
                     len(todo), failed, per, per * (len(todo) - done) / 60)
    recs = [json.loads((out_dir / f"tic_{r['tic_id']}.json").read_text(encoding="utf-8"))
            for r in rows if (out_dir / f"tic_{r['tic_id']}.json").exists()]
    df = pd.DataFrame(recs)
    bad = df[df["status"] != "ok"]
    for _, r in bad.iterrows():
        print(f"FAILED TIC {r['tic_id']} ({r['set']}): {r['status']} {r['reason']}")
    df = df[df["status"] == "ok"]
    pd.set_option("display.width", 160)
    for name, g in df.groupby("set"):
        tag = "fresh, never seen" if name == "fresh" else "seen during design"
        print(f"\n== {name} ({tag}), {len(g)} EBs, median {g['n_sectors'].median():.0f} sectors")
        print(f"orbital period correct: before {int(g['before_ok'].sum())}/{len(g)} -> after "
              f"{int(g['after_ok'].sum())}/{len(g)}")
        print("after, by rule:", g.groupby("after_rule")["after_ok"].agg(["sum", "size"])
              .to_dict("index"))
        changed = g[g["before_ok"] != g["after_ok"]]
        if len(changed):
            print(changed[["tic_id", "catalog_period", "before_period", "after_period",
                           "after_rule"]].round(4).to_string(index=False))
    known = df[df["catalog_type"] != "unknown"]
    for col, label in (("morph_features", "from features (BLS duration)"),
                       ("morph_measured", "from the module's own measurements")):
        ok = known[col] == known["catalog_type"]
        print(f"\nBinary type {label}: {int(ok.sum())}/{len(known)} agree "
              f"(training {int(ok[known.set == 'training'].sum())}/"
              f"{int((known.set == 'training').sum())}, fresh "
              f"{int(ok[known.set == 'fresh'].sum())}/{int((known.set == 'fresh').sum())})")
        print(pd.crosstab(known["catalog_type"], known[col]).reindex(index=ORDER, columns=ORDER,
                                                                      fill_value=0))


if __name__ == "__main__":
    main()
