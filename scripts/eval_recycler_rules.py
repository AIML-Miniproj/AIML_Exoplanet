"""Check the rule-based recycler modules against catalog labels.

    python scripts/eval_recycler_rules.py

- EB binary type (detached / semi-detached / contact) vs the TESS EB catalog's
  morphology value, on the training EBs and the fresh EB set.
- Variable type vs the Gaia DR3 class, on the 25 training variables.

Uses the feature tables only (no light curves are re-processed).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import CONFIG  # noqa: E402
from src.recycler import eb_module, variable_module  # noqa: E402

ORDER_EB = ["detached", "semi-detached", "contact"]


def eb_morphology() -> pd.DataFrame:
    rows = []
    for name in ("features", "fresh_eb"):
        df = pd.read_parquet(CONFIG.paths.processed / f"{name}.parquet")
        for _, r in df[(df["label"] == "EB") & (df["status"] == "ok")].iterrows():
            feats = r.to_dict()
            tp = eb_module.true_period(feats["det_period"], feats)
            kind, _ = eb_module.morphology(feats, tp.period_d)
            cat = eb_module.catalog_entry(int(r["tic_id"]))
            rows.append({"set": name, "tic_id": int(r["tic_id"]), "rule": kind,
                         "catalog": cat["type"] if cat else "unknown",
                         "morph": cat["morph"] if cat else float("nan")})
    return pd.DataFrame(rows)


def variable_types() -> pd.DataFrame:
    df = pd.read_parquet(CONFIG.paths.processed / "features.parquet")
    lab = pd.read_parquet(CONFIG.paths.labels / "training_set.parquet")
    var_class = lab.set_index("tic_id")["var_class"]
    rows = []
    for _, r in df[(df["label"] == "VARIABLE") & (df["status"] == "ok")].iterrows():
        kind, rule = variable_module.classify(r.to_dict())
        rows.append({"tic_id": int(r["tic_id"]), "rule_class": kind, "rule": rule,
                     "gaia_class": var_class.get(int(r["tic_id"])),
                     "period_d": r["var_period_d"], "amp_pct": r["amp_p2p_ppm"] / 1e4,
                     "ls_power": r["ls_power_1"], "teff": r["teff"]})
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 160)
    eb = eb_morphology()
    for name, g in [("all 50", eb), *eb.groupby("set")]:
        g = g[g["catalog"] != "unknown"]
        print(f"\nEB type, {name}: rule agrees with catalog for {int((g.rule == g.catalog).sum())}"
              f" / {len(g)}")
    print(pd.crosstab(eb["catalog"], eb["rule"]).reindex(index=ORDER_EB, columns=ORDER_EB,
                                                          fill_value=0))
    var = variable_types()
    ok = var["rule_class"] == var["gaia_class"]
    print(f"\nVariable type: rule agrees with Gaia for {int(ok.sum())} / {len(var)}")
    print(pd.crosstab(var["gaia_class"], var["rule_class"]))
    print(var.round(3).sort_values("gaia_class").to_string(index=False))


if __name__ == "__main__":
    main()
