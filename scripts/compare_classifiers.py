"""Compare extra classifiers (SVM, k-nearest neighbours) with the existing models.

    python scripts/compare_classifiers.py

Every model is scored on the SAME 5 grouped folds as scripts/train_triage.py
(StratifiedGroupKFold by TIC), so every star is predicted once by a model that
never saw it. Nothing in models/ or the app changes: results go to
outputs/classifier_comparison.json, outputs/classifier_comparison.md and one
confusion-matrix plot per model in outputs/.
"""

# ── In plain words ───────────────────────────────────────────────────────────
# A side experiment: would a completely different kind of classifier do better?
#   SVM (support vector machine, RBF kernel): draws curved boundaries between
#     the classes with the widest possible gap ("margin") around them.
#   KNN (k-nearest neighbours): no training at all; a new star gets the class
#     most common among the k most similar training stars.
# Both need every feature on the same scale and no missing values, so they get
# the same preparation as logistic regression: fill gaps with the median (+ a
# 0/1 "was missing" column) and standardise. They are compared with XGBoost,
# random forest and logistic regression on identical folds and metrics.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from config import CONFIG  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.triage.labels import CLASSES  # noqa: E402
from src.triage.train import (_fit, cv_folds, fold_summary, load_dataset,  # noqa: E402
                              make_baselines, make_xgb, metrics, plot_confusion)


def make_svm():
    # C = how strongly misclassified training stars are penalised (higher = tighter fit);
    # gamma="scale" sets the kernel width from the data; probability=True adds Platt
    # scaling inside the SVM so it can output class probabilities like the others.
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced",
                             probability=True, random_state=CONFIG.random_state))


def make_knn():
    # k = 9 neighbours (about 3% of a 320-star training fold); closer neighbours count
    # more ("distance" weights), so a far-away neighbour cannot outvote a close one.
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         KNeighborsClassifier(n_neighbors=9, weights="distance"))


def evaluate(name, factory, ds, folds, cfg):
    """Out-of-fold probabilities + per-fold metrics + time to train and predict."""
    oof = np.zeros((len(ds.y), len(CLASSES)))
    per_fold, fit_s, pred_s = [], 0.0, 0.0
    for tr, va in folds:
        t0 = time.perf_counter()
        m = _fit(factory(), ds.X.iloc[tr], ds.y[tr])
        fit_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        oof[va] = m.predict_proba(ds.X.iloc[va])
        pred_s += time.perf_counter() - t0
        per_fold.append(metrics(ds.y[va], oof[va], cfg))
    pooled = metrics(ds.y, oof, cfg)
    folds_m = fold_summary(per_fold)
    # macro precision / recall / PR-AUC per fold (fold_summary keeps them per class)
    macro = {k: {"mean": float(np.mean([np.mean([f["per_class"][c][k] for c in CLASSES])
                                        for f in per_fold])),
                 "std": float(np.std([np.mean([f["per_class"][c][k] for c in CLASSES])
                                      for f in per_fold], ddof=1))}
             for k in ("precision", "recall", "pr_auc")}
    return {"model": name, "pooled": pooled, "folds": folds_m, "macro_folds": macro,
            "fit_seconds_total": fit_s, "predict_ms_per_star": 1000 * pred_s / len(ds.y)}


def _ms(d):
    return f"{d['mean']:.2f} ± {d['std']:.2f}"


def main() -> None:
    setup_logging()
    cfg = CONFIG
    ds = load_dataset(pd.read_parquet(cfg.paths.processed / "features.parquet"))
    folds = cv_folds(ds, cfg)
    base = make_baselines(cfg)
    models = {
        "XGBoost (raw, uncalibrated)": lambda: make_xgb(cfg),
        "Random forest": lambda: clone(base["random_forest"]),
        "Logistic regression": lambda: clone(base["logistic_regression"]),
        "SVM (RBF kernel)": make_svm,
        "KNN (k = 9)": make_knn,
    }
    results = []
    for name, factory in models.items():
        r = evaluate(name, factory, ds, folds, cfg)
        results.append(r)
        print(f"{name:<30} macro-F1 {_ms(r['folds']['macro_f1'])}")
        slug = name.split(" ")[0].lower()
        plot_confusion(r["pooled"]["confusion_normalized"],
                       cfg.paths.outputs / f"compare_confusion_{slug}.png",
                       f"{name}: out-of-fold predictions")

    lines = [f"# Classifier comparison ({len(ds.y)} stars, {len(folds)} grouped folds)", "",
             "Mean ± std over folds. Precision, recall and PR-AUC are macro averages "
             "(mean over the 4 classes).", "",
             "| Model | Accuracy | Precision | Recall | Macro-F1 | PR-AUC | Log loss | "
             "Planets kept at 5% EB | Train time (5 folds) | Predict per star |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        f, m = r["folds"], r["macro_folds"]
        lines.append(f"| {r['model']} | {_ms(f['accuracy'])} | {_ms(m['precision'])} | "
                     f"{_ms(m['recall'])} | {_ms(f['macro_f1'])} | {_ms(m['pr_auc'])} | "
                     f"{_ms(f['log_loss'])} | {_ms(f['planet_recall_at_eb_rate'])} | "
                     f"{r['fit_seconds_total']:.1f} s | {r['predict_ms_per_star']:.2f} ms |")
    lines += ["", "## F1 per class", "",
              "| Model | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for r in results:
        lines.append(f"| {r['model']} | " + " | ".join(
            _ms(r["folds"]["per_class"][c]["f1"]) for c in CLASSES) + " |")
    lines += ["", "## Confusion matrices (rows = true class, columns = predicted)", ""]
    for r in results:
        lines += [f"**{r['model']}**", "", "| true \\ predicted | " + " | ".join(CLASSES) + " |",
                  "|---|" + "---|" * len(CLASSES)]
        lines += [f"| {c} | " + " | ".join(str(v) for v in row) + " |"
                  for c, row in zip(CLASSES, r["pooled"]["confusion"])]
        lines.append("")
    out = cfg.paths.outputs
    out.mkdir(parents=True, exist_ok=True)
    (out / "classifier_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "classifier_comparison.json").write_text(json.dumps(results, indent=1, default=float),
                                                     encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
