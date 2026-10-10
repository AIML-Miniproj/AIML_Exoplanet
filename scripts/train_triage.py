"""Train the triage classifier from data/processed/features.parquet.

    python scripts/train_triage.py

Writes models/triage_xgb.json, models/triage_meta.json and evaluation plots in outputs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import CONFIG  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.triage.train import train  # noqa: E402


def _table(models: dict) -> str:
    lines = ["model                 macroF1  acc   " + "  ".join(
        f"{c[:6]:>6}F1 {c[:6]:>6}AP" for c in ("PLANET", "EB", "VARIABLE", "NOISE"))]
    for name, m in models.items():
        t = m["test"]
        per = "  ".join(f"{t['per_class'][c]['f1']:8.3f} {t['per_class'][c]['pr_auc']:8.3f}"
                        for c in ("PLANET", "EB", "VARIABLE", "NOISE"))
        lines.append(f"{name:<21} {t['macro_f1']:.3f}   {t['accuracy']:.3f} {per}")
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    feats = pd.read_parquet(CONFIG.paths.processed / "features.parquet")
    labels = pd.read_parquet(CONFIG.paths.labels / "labels.parquet")
    res = train(feats, labels)
    print(_table(res["models"]))
    th = res["meta"]["thresholds"]
    print(json.dumps(th, indent=1))
    (CONFIG.paths.outputs / "triage_results.json").write_text(
        json.dumps(res["models"], indent=1, default=float))


if __name__ == "__main__":
    main()
