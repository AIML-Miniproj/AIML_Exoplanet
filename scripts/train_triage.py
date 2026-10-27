"""Train the triage classifier from data/processed/features.parquet.

    python scripts/train_triage.py

Evaluates with 5-fold StratifiedGroupKFold by TIC (no separate test set at this
dataset size). Writes models/triage_xgb.json, models/triage_meta.json,
outputs/triage_results.json and evaluation plots in outputs/.
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

CLASSES = ("PLANET", "EB", "VARIABLE", "NOISE")


def _cell(d: dict) -> str:
    return f"{d['mean']:.2f} +/- {d['std']:.2f}"


def _table(models: dict) -> str:
    """Mean +/- std across the grouped folds."""
    lines = [f"{'model':<30} {'macro-F1':>13} {'accuracy':>13} "
             + " ".join(f"{c + ' F1':>13}" for c in CLASSES)]
    for name, m in models.items():
        f = m["folds"]
        lines.append(f"{name:<30} {_cell(f['macro_f1']):>13} {_cell(f['accuracy']):>13} "
                     + " ".join(f"{_cell(f['per_class'][c]['f1']):>13}" for c in CLASSES))
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    feats = pd.read_parquet(CONFIG.paths.processed / "features.parquet")
    sel = CONFIG.paths.labels / "training_set.parquet"
    labels = pd.read_parquet(sel if sel.exists() else CONFIG.paths.labels / "labels.parquet")
    res = train(feats, labels)
    print(_table(res["models"]))
    print(json.dumps(res["meta"]["thresholds"], indent=1))
    (CONFIG.paths.outputs / "triage_results.json").write_text(
        json.dumps(res["models"], indent=1, default=float))


if __name__ == "__main__":
    main()
