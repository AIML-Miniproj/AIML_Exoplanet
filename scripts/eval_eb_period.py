"""How often the EB module reports the true orbital period.

    python scripts/eval_eb_period.py [--feature-set fresh_eb] [--feature-set features]

Compares the EB module's period with the TESS EB catalog period (match within 1%)
for every eclipsing binary in a feature table built by scripts/build_features.py.
The training EBs were looked at while designing the rule; the "fresh_eb" set never was.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import CONFIG  # noqa: E402
from src.recycler.eb_module import true_period  # noqa: E402


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    eb = df[(df["label"] == "EB") & (df["status"] == "ok")].copy()
    tps = [true_period(r["det_period"], r.to_dict()) for _, r in eb.iterrows()]
    eb["detected_ratio"] = (eb["det_period"] / eb["catalog_period"]).round(3)
    eb["true_period"] = [t.period_d for t in tps]
    eb["rule"] = [t.rule for t in tps]
    eb["true_ok"] = (eb["true_period"] / eb["catalog_period"] - 1).abs() < 0.01
    return eb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-set", action="append", default=None)
    args = ap.parse_args()
    for name in args.feature_set or ["fresh_eb", "features"]:
        df = pd.read_parquet(CONFIG.paths.processed / f"{name}.parquet")
        eb = evaluate(df)
        tag = "fresh, never seen" if name == "fresh_eb" else "seen during design"
        at = eb["detected_ratio"].round(1).map({1.0: "true P", 0.5: "half P"}).fillna("other")
        print(f"\n== {name} ({tag}): {len(eb)} EBs, failed runs "
              f"{int(((df['label'] == 'EB') & (df['status'] != 'ok')).sum())}")
        print(f"detected at: {at.value_counts().to_dict()}")
        print(f"EB module true period correct: {int(eb['true_ok'].sum())} / {len(eb)}")
        print(eb.groupby("rule")["true_ok"].agg(correct="sum", n="size").to_string())
        print(pd.crosstab(at, eb["true_ok"].map({True: "correct", False: "wrong"})).to_string())


if __name__ == "__main__":
    main()
