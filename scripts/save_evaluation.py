"""Collect the saved training results into one file for the app's Results tab.

    python scripts/save_evaluation.py

Reads what training and the evaluation scripts already wrote (no retraining):
  outputs/triage_results.json          5-fold grouped CV metrics of all three models
  models/triage_meta.json + model      to recompute global mean |SHAP| on the training stars
  data/processed/features_summary.json recovery per class
  data/processed/eb_eval/*.json        EB true-period accuracy, fresh vs training binaries
and writes models/model_evaluation.json (committed, so the app works on a clean clone).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import CONFIG  # noqa: E402

CLASSES = ["PLANET", "EB", "VARIABLE", "NOISE"]
MODEL_NAMES = {"xgboost_calibrated": "XGBoost (calibrated)",
               "logistic_regression": "Logistic regression", "random_forest": "Random forest"}


def global_importance(top: int = 10) -> list[tuple[str, float]]:
    """Mean |SHAP| summed over the four classes, on the stars the model was trained on."""
    import shap

    from src.triage.model import TriageModel

    model = TriageModel.load(CONFIG.paths.models)
    if model is None:
        sys.exit("No trained model in models/: run scripts/train_triage.py first.")
    df = pd.read_parquet(CONFIG.paths.processed / "features.parquet")
    df = df[df["status"] == "ok"]
    X = df[model.features].astype(float).replace([np.inf, -np.inf], np.nan)
    sv = np.asarray(shap.TreeExplainer(model.model).shap_values(X))
    if sv.ndim == 3 and sv.shape[0] == len(CLASSES) and sv.shape[-1] != len(CLASSES):
        sv = np.moveaxis(sv, 0, -1)
    imp = np.abs(sv).mean(axis=0).sum(axis=1)
    order = np.argsort(-imp)[:top]
    return [(model.features[i], round(float(imp[i]), 4)) for i in order]


def eb_period_accuracy() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for p in sorted((CONFIG.paths.processed / "eb_eval").glob("tic_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") != "ok":
            continue
        s = out.setdefault(r["set"], {"correct": 0, "n": 0})
        s["n"] += 1
        s["correct"] += bool(r.get("after_ok"))
    return out


def main() -> None:
    res = json.loads((CONFIG.paths.outputs / "triage_results.json").read_text(encoding="utf-8"))
    xgb = res["xgboost_calibrated"]
    per_class = {c: {m: xgb["folds"]["per_class"][c][m] for m in ("precision", "recall", "f1",
                                                                   "pr_auc")}
                 for c in CLASSES}
    comparison = []
    for key, name in MODEL_NAMES.items():
        f = res[key]["folds"]
        comparison.append({"model": name, "macro_f1": f["macro_f1"], "accuracy": f["accuracy"],
                           "pooled_macro_f1": res[key]["cv"]["macro_f1"]})
    summary = json.loads((CONFIG.paths.processed / "features_summary.json")
                         .read_text(encoding="utf-8"))
    recovery = {c: {k: summary[c].get(k) for k in ("n", "n_ok", "n_recovery_checked",
                                                   "recovery_rate", "recovered_at_ratio")}
                for c in CLASSES if c in summary}
    out = {
        "classes": CLASSES,
        "evaluation": "5-fold StratifiedGroupKFold by TIC (every star scored once by a model "
                      "that never saw it); no separate held-out set",
        "n_stars": int(sum(summary[c]["n_ok"] for c in CLASSES if c in summary)),
        "confusion": xgb["cv"]["confusion"],
        "confusion_normalized": xgb["cv"]["confusion_normalized"],
        "per_class": per_class,
        "macro_f1": xgb["folds"]["macro_f1"],
        "comparison": comparison,
        "importance": global_importance(),
        "recovery": recovery,
        "eb_true_period": eb_period_accuracy(),
    }
    path = CONFIG.paths.models / "model_evaluation.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path}")
    print(json.dumps({k: out[k] for k in ("macro_f1", "eb_true_period")}, indent=1))
    print("importance:", out["importance"])


if __name__ == "__main__":
    main()
