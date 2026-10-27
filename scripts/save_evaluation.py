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


def _row(name: str, folds: dict, macro: dict, pooled_f1: float, used: bool) -> dict:
    """One model's line in the comparison table (each metric: mean and std over the folds)."""
    return {"model": name, "used_by_app": used,
            "accuracy": folds["accuracy"], "precision": macro["precision"],
            "recall": macro["recall"], "macro_f1": folds["macro_f1"],
            "pr_auc": macro["pr_auc"], "log_loss": folds["log_loss"],
            "per_class_f1": {c: folds["per_class"][c]["f1"] for c in CLASSES},
            "pooled_macro_f1": pooled_f1}


def model_comparison(res: dict) -> list[dict]:
    """All classifiers on the same 5 folds, best first.

    The calibrated XGBoost (the model the app uses) comes from scripts/train_triage.py.
    The others come from scripts/compare_classifiers.py when it has been run (random
    forest, logistic regression, SVM, KNN); otherwise only the training baselines.
    """
    xgb = res["xgboost_calibrated"]["folds"]
    # macro precision / recall / PR-AUC: mean over classes of the fold means (no
    # per-fold macro values are stored for this model, so no spread)
    macro = {m: {"mean": float(np.mean([xgb["per_class"][c][m]["mean"] for c in CLASSES])),
                 "std": None} for m in ("precision", "recall", "pr_auc")}
    rows = [_row("XGBoost (calibrated)", xgb, macro,
                 res["xgboost_calibrated"]["cv"]["macro_f1"], True)]
    extra = CONFIG.paths.outputs / "classifier_comparison.json"
    if extra.exists():
        for r in json.loads(extra.read_text(encoding="utf-8")):
            if r["model"].startswith("XGBoost"):
                continue                      # the calibrated version above is the one used
            rows.append(_row(r["model"], r["folds"], r["macro_folds"],
                             r["pooled"]["macro_f1"], False))
    else:
        for key, name in MODEL_NAMES.items():
            if key != "xgboost_calibrated":
                f = res[key]["folds"]
                m = {k: {"mean": float(np.mean([f["per_class"][c][k]["mean"] for c in CLASSES])),
                         "std": None} for k in ("precision", "recall", "pr_auc")}
                rows.append(_row(name, f, m, res[key]["cv"]["macro_f1"], False))
    return sorted(rows, key=lambda r: -r["macro_f1"]["mean"])


def feature_selection() -> dict | None:
    """Summary of scripts/select_features.py (score vs features kept), if it was run."""
    path = CONFIG.paths.outputs / "feature_selection.json"
    if not path.exists():
        return None
    fs = json.loads(path.read_text(encoding="utf-8"))
    return {k: fs[k] for k in ("n_features_all", "curve", "nested", "adopted", "decision")} | {
        "top_features": fs["ranking_all_stars"][:15]}


def main() -> None:
    res = json.loads((CONFIG.paths.outputs / "triage_results.json").read_text(encoding="utf-8"))
    xgb = res["xgboost_calibrated"]
    per_class = {c: {m: xgb["folds"]["per_class"][c][m] for m in ("precision", "recall", "f1",
                                                                   "pr_auc")}
                 for c in CLASSES}
    comparison = model_comparison(res)
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
        "feature_selection": feature_selection(),
    }
    path = CONFIG.paths.models / "model_evaluation.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path}")
    print(json.dumps({k: out[k] for k in ("macro_f1", "eb_true_period")}, indent=1))
    print("importance:", out["importance"])


if __name__ == "__main__":
    main()
