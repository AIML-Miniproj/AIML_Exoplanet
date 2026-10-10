"""Post-training checks: verdict stability across data amounts, and demo-star verdicts.

    python scripts/check_verdicts.py

1. Stability: the same star analysed with few and with many sectors must get the
   same triage verdict (model inputs don't scale with the amount of data).
2. Demo verdicts: the 4 demo stars with the app's default preset.

Writes outputs/verdict_checks.json.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import CONFIG, Preset  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.pipeline import analyze_target  # noqa: E402

HD209458 = 420814525
TOI862 = 309254930          # 18 SPOC sectors: a real few-vs-many comparison
TOI476 = 156987351          # back-up unconfirmed-candidate demo star


def verdict(tic: int, preset, cfg) -> dict:
    rep = analyze_target(f"TIC {tic}", preset=preset, cfg=cfg)
    tri = rep.triage
    return {"tic_id": tic, "preset": getattr(preset, "name", preset),
            "sectors": [s["sector"] for s in rep.sectors],
            "period_d": rep.signals[0]["period"],
            "rp_rjup": rep.physical.get("rp_rjup"),
            "var_period_d": rep.features.get("var_period_d"),
            "label": tri.get("predicted_class"), "confidence": tri.get("confidence"),
            "probabilities": tri.get("probabilities"), "status": tri.get("status")}


def main() -> None:
    setup_logging("WARNING")
    cfg = CONFIG
    out: dict = {"stability": [], "demo": []}

    runs = [(HD209458, Preset("1-sector", 1, False, 1)),
            (HD209458, Preset("2-sector", 2, False, 1)),
            (HD209458, Preset("all-sectors", None, False, 1)),
            (TOI862, Preset("2-sector", 2, False, 1)),
            (TOI862, Preset("10-sector", 10, False, 1))]
    for tic, preset in runs:
        v = verdict(tic, preset, cfg)
        out["stability"].append(v)
        print(f"TIC {tic} {preset.name:<12} sectors {v['sectors']}: {v['label']} "
              f"({v['confidence']}) P={v['period_d']:.5f} probs {v['probabilities']}",
              flush=True)

    demo = pd.read_csv(CONFIG.paths.labels / "demo_set.csv")
    for _, row in demo.iterrows():
        v = verdict(int(row["tic_id"]), "standard", cfg)
        v.update(slot=row["slot"], name=row["name"], catalog_period=row["period_d"])
        out["demo"].append(v)
        print(f"{row['slot']:<10} {row['name']:<12} sectors {v['sectors']}: {v['label']} "
              f"({v['confidence']}) P={v['period_d']:.5f} var_P={v['var_period_d']:.5f} "
              f"probs {v['probabilities']}", flush=True)

    toi = pd.read_parquet(CONFIG.paths.labels / "raw" / "toi.parquet")
    disp = sorted(set(toi.loc[toi["tid"] == TOI476, "tfopwg_disp"].dropna()))
    v = verdict(TOI476, "standard", cfg)
    v.update(slot="toi_pc_backup", name="TOI 476.01", tfopwg_disp=disp)
    out["demo"].append(v)
    print(f"backup     TOI 476.01   (TOI table: {disp}) sectors {v['sectors']}: {v['label']} "
          f"({v['confidence']}) P={v['period_d']:.5f} probs {v['probabilities']}", flush=True)

    path = CONFIG.paths.outputs / "verdict_checks.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
