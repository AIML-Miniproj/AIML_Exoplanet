"""Ensemble experiments: hard voting, weighted soft voting and seed-bagged XGBoost.

    python scripts/compare_ensembles.py

All on the same 5 grouped folds as scripts/train_triage.py. The voting weights
are chosen by an inner 3-fold grouped CV inside each outer training part
(nested), so the outer test fold never influences them. Writes
outputs/ensemble_comparison.json and .md. Nothing in models/ or the app changes.
"""

# ── In plain words ───────────────────────────────────────────────────────────
# An ensemble combines several models so their mistakes cancel out.
#   Hard voting:          each model names one class; the majority wins
#                         (ties go to XGBoost, the strongest model).
#   Weighted soft voting: average the class probabilities, giving each model a
#                         weight (e.g. 70% XGBoost, 30% random forest).
#   Seed bagging:         10 XGBoost models that differ only in their random seed,
#                         averaged.
# It only helps when the members are right on DIFFERENT stars. This script also
# counts how many of XGBoost's mistakes the other models could fix.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from compare_classifiers import make_knn, make_svm  # noqa: E402
from config import CONFIG  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.triage.labels import CLASSES  # noqa: E402
from src.triage.train import _fit, cv_folds, load_dataset, make_baselines, make_xgb  # noqa: E402

SEEDS = range(10)
VOTERS = ("xgb", "rf", "svm")                         # members of the weighted soft vote
# all weight triples on a 0.1 grid that sum to 1 (66 combinations)
GRID = [w for w in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=3)
        if abs(sum(w) - 1) < 1e-9]


def factories(cfg):
    base = make_baselines(cfg)
    return {"xgb": lambda: make_xgb(cfg), "rf": lambda: clone(base["random_forest"]),
            "svm": make_svm, "lr": lambda: clone(base["logistic_regression"]), "knn": make_knn}


def hard_vote(labels: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    """Majority class per star; among tied classes, the first model in ``order`` decides."""
    stacked = np.vstack([labels[n] for n in order])
    out = []
    for j in range(stacked.shape[1]):
        votes = np.bincount(stacked[:, j], minlength=len(CLASSES))
        tied = np.flatnonzero(votes == votes.max())
        out.append(next(p for p in stacked[:, j] if p in tied))
    return np.array(out)


def blend(probs: dict[str, np.ndarray], w) -> np.ndarray:
    return sum(wi * probs[n] for wi, n in zip(w, VOTERS))


def macro_f1(y, labels) -> float:
    return float(f1_score(y, labels, average="macro"))


def main() -> None:
    setup_logging()
    cfg = CONFIG
    ds = load_dataset(pd.read_parquet(cfg.paths.processed / "features.parquet"))
    folds = cv_folds(ds, cfg)
    make = factories(cfg)
    pred = {n: np.zeros(len(ds.y), int) for n in make}
    scores = {k: [] for k in ("xgb", "hard3", "hard5", "weighted", "bag10")}
    weights = []
    for k, (tr, va) in enumerate(folds, 1):
        Xtr, ytr, gtr, Xte, yte = ds.X.iloc[tr], ds.y[tr], ds.groups[tr], ds.X.iloc[va], ds.y[va]
        models = {n: _fit(f(), Xtr, ytr) for n, f in make.items()}
        prob = {n: m.predict_proba(Xte) for n, m in models.items()}
        for n in make:
            pred[n][va] = prob[n].argmax(1)
        # weights for the soft vote, chosen on the training part only (inner 3-fold CV)
        inner = {n: np.zeros((len(tr), len(CLASSES))) for n in VOTERS}
        for itr, iva in StratifiedGroupKFold(3, shuffle=True, random_state=k).split(Xtr, ytr, gtr):
            for n in VOTERS:
                inner[n][iva] = _fit(make[n](), Xtr.iloc[itr], ytr[itr]).predict_proba(
                    Xtr.iloc[iva])
        w = max(GRID, key=lambda w: (macro_f1(ytr, blend(inner, w).argmax(1)), w[0]))
        weights.append([float(x) for x in w])
        bag = np.mean([_fit(make_xgb(cfg).set_params(random_state=s), Xtr, ytr)
                       .predict_proba(Xte) for s in SEEDS], axis=0)
        scores["xgb"].append(macro_f1(yte, pred["xgb"][va]))
        scores["hard3"].append(macro_f1(yte, hard_vote({n: pred[n][va] for n in make},
                                                       ["xgb", "rf", "svm"])))
        scores["hard5"].append(macro_f1(yte, hard_vote({n: pred[n][va] for n in make},
                                                       ["xgb", "rf", "svm", "lr", "knn"])))
        scores["weighted"].append(macro_f1(yte, blend(prob, w).argmax(1)))
        scores["bag10"].append(macro_f1(yte, bag.argmax(1)))
        print(f"fold {k}: weights xgb/rf/svm = {w}")

    names = {"xgb": "XGBoost alone", "hard3": "Hard voting: XGBoost + RF + SVM",
             "hard5": "Hard voting: all 5 models",
             "weighted": "Weighted soft voting (nested weights)",
             "bag10": "Seed-bagged XGBoost (10 seeds)"}
    base = np.array(scores["xgb"])
    rows = []
    for key, v in scores.items():
        v = np.array(v)
        rows.append({"method": names[key], "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                     "gain": float((v - base).mean()), "folds_better": int((v > base).sum())})
    err = {n: pred[n] != ds.y for n in make}
    fixable = int((err["xgb"] & ~err["rf"] & ~err["svm"]).sum())
    breakable = int((~err["xgb"] & err["rf"] & err["svm"]).sum())
    shared = {n: int((err["xgb"] & err[n]).sum()) for n in make if n != "xgb"}
    out = {"results": rows, "weights_per_fold": weights,
           "errors": {n: int(e.sum()) for n, e in err.items()},
           "xgb_errors_shared_with": shared,
           "hard_vote_can_fix": fixable, "hard_vote_can_break": breakable}
    lines = [f"# Ensembles ({len(ds.y)} stars, {len(folds)} grouped folds)", "",
             "| Method | Macro-F1 | vs XGBoost | Better in |", "|---|---|---|---|"]
    lines += [f"| {r['method']} | {r['mean']:.3f} ± {r['std']:.3f} | {r['gain']:+.3f} | "
              f"{r['folds_better']}/{len(folds)} |" for r in rows]
    lines += ["", f"Errors (of {len(ds.y)} stars): {out['errors']}",
              f"XGBoost errors also made by: {shared}",
              f"Hard vote XGB+RF+SVM can fix {fixable} XGBoost errors and break {breakable} "
              "correct answers.", f"Soft-vote weights per fold (xgb, rf, svm): {weights}"]
    path = cfg.paths.outputs
    path.mkdir(parents=True, exist_ok=True)
    (path / "ensemble_comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (path / "ensemble_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
