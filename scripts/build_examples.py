"""Build the app's example list and precomputed demo reports.

    python scripts/build_examples.py [--n-jobs 2] [--confirm-long]

Writes data/examples/examples.json (the grouped dropdown) and
data/examples/demo/<tic>.json (full reports with the standard preset). Extra
examples are never training stars: eclipsing binaries come from the separate
fresh EB set, planets and variables from outside the training list. They were
chosen for good data (bright, clear signal, several sectors), not for how the
model does on them; the "Tricky cases" group shows real failures honestly.
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
from src.demo_cache import build_one, size_mb  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.triage.dataset import run_with_timeout  # noqa: E402

log = logging.getLogger("build_examples")

GROUPS = ["Confirmed planets", "Eclipsing binaries", "Variable stars",
          "Unconfirmed candidates", "Tricky cases"]
EXAMPLES = [
    ("Confirmed planets", 420814525, "HD 209458 b", "hot Jupiter, 3.5-day orbit"),
    ("Confirmed planets", 248853232, "TOI 5823.01", "known giant planet, 3.9-day orbit"),
    ("Confirmed planets", 376637093, "TOI 1516.01", "confirmed giant planet, 2.1-day orbit"),
    ("Eclipsing binaries", 300871545, "TOI 184.01",
     "planet candidate that turned out to be two stars"),
    ("Eclipsing binaries", 14709809, "TIC 14709809", "two stars eclipsing every 9.4 days"),
    ("Eclipsing binaries", 236327696, "TIC 236327696", "close pair of stars, 14-hour orbit"),
    ("Variable stars", 441584047, "HIP 36213", "pulsating RR Lyrae star"),
    ("Variable stars", 76046904, "TIC 76046904", "gently pulsating delta Scuti / gamma Dor star"),
    ("Variable stars", 42362686, "TIC 42362686", "pulsating giant (Cepheid)"),
    ("Unconfirmed candidates", 156987351, "TOI 476.01", "planet candidate, 3.1-day orbit"),
    ("Unconfirmed candidates", 327579226, "TOI 1532.01", "planet candidate, 8.9-day orbit"),
    ("Tricky cases", 329864959, "TOI 1385.01",
     "two stars whose eclipses look alike: we find half their true orbit"),
    ("Tricky cases", 309254930, "TOI 862.01",
     "a possible planet hidden by the star's own flickering"),
]
# Rough per-star runtimes measured on this machine (standard preset, single worker).
EST_CACHED_S, EST_DOWNLOAD_S = 45.0, 90.0


def estimate_minutes(rows: list[dict], workers: int) -> float:
    total = 0.0
    for r in rows:
        cached = any((CONFIG.paths.processed / f"tic_{r['tic_id']}").glob("*.npz"))
        total += EST_CACHED_S if cached else EST_DOWNLOAD_S
    return total / workers / 60


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--confirm-long", action="store_true")
    ap.add_argument("--only", type=int, nargs="*", help="rebuild only these TIC ids")
    args = ap.parse_args()
    setup_logging()

    ex_dir = CONFIG.paths.examples
    demo_dir = ex_dir / "demo"
    status_dir = CONFIG.paths.processed / "examples_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    training = set(pd.read_parquet(CONFIG.paths.labels / "training_set.parquet")["tic_id"])
    clash = [name for _, tic, name, _ in EXAMPLES if tic in training]
    if clash:
        sys.exit(f"Examples must not be training stars: {clash}")

    rows = [{"group": g, "tic_id": tic, "display_name": name, "hint": hint,
             "demo_dir": str(demo_dir), "label": g, "source": "example"}
            for g, tic, name, hint in EXAMPLES if not args.only or tic in args.only]
    workers = CONFIG.workers(args.n_jobs)
    minutes = estimate_minutes(rows, workers)
    log.info("%d examples, estimated %.0f min with %d workers", len(rows), minutes, workers)
    if minutes > CONFIG.max_job_minutes / 3 and not args.confirm_long:
        sys.exit(f"Estimated {minutes:.0f} min is above 10 min: re-run with --confirm-long "
                 f"once approved.")

    from src.recycler.eb_eval import with_subset
    cfg = with_subset(CONFIG, [r["tic_id"] for r in rows])
    start = time.monotonic()
    res = []
    for i in range(0, len(rows), cfg.chunk_size):
        res += run_with_timeout(rows[i:i + cfg.chunk_size], "standard", cfg, status_dir,
                                workers, 300.0, target=build_one)
        log.info("Progress: %d/%d done, %d failed, %.1f min", len(res), len(rows),
                 sum(r["status"] != "ok" for r in res), (time.monotonic() - start) / 60)
    for r in res:
        if r["status"] != "ok":
            log.error("TIC %s: %s %s", r["tic_id"], r["status"], r.get("reason", ""))

    items = []
    for g, tic, name, hint in EXAMPLES:
        path = demo_dir / f"{tic}.json"
        if not path.exists():
            continue
        rep = json.loads(path.read_text(encoding="utf-8"))
        items.append({"group": g, "tic_id": tic, "display_name": name, "hint": hint,
                      "n_sectors": len(rep.get("sectors", [])),
                      "verdict": rep.get("routing", {}).get("primary_class"),
                      "demo_file": f"demo/{tic}.json"})
    (ex_dir / "examples.json").write_text(json.dumps({"groups": GROUPS, "examples": items},
                                                     indent=1), encoding="utf-8")
    log.info("examples.json: %d examples; demo reports %.1f MB", len(items), size_mb(demo_dir))
    for it in items:
        print(f"{it['group']:<24} {it['display_name']:<14} {it['n_sectors']:>2} sectors -> "
              f"{it['verdict']}")


if __name__ == "__main__":
    main()
