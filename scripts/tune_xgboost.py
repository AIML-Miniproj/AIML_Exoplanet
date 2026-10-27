"""Hyperparameter tuning of the XGBoost classifier, with nested cross-validation.

    python scripts/tune_xgboost.py [--n-iter 40]

Outer loop: the same 5 grouped folds as scripts/train_triage.py. Inside each
outer training part, RandomizedSearchCV tries ``--n-iter`` random settings,
scoring each with an inner 3-fold grouped CV, and refits the best one; that
model is then scored once on the untouched outer test fold. The mean over the 5
outer folds is an honest estimate of "XGBoost + tuning". It is compared with the
current hand-chosen settings on the same outer folds.

Nothing in models/ or the app changes: results go to outputs/xgb_tuning.json and
outputs/xgb_tuning.md.
"""

# ── In plain words ───────────────────────────────────────────────────────────
# Hyperparameters are the knobs set BEFORE training (how deep each tree may grow,
# how many trees, how big each learning step is, ...). Tuning = trying many knob
# settings and keeping the best.
# The trap: if the same stars are used to pick the knobs AND to report the score,
# the score is too optimistic (the knobs were fitted to those stars). Nested CV
# avoids that: the knobs are picked using only the training part of each outer
# fold, and the outer test fold is touched once, at the end.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import loguniform, randint, uniform  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from config import CONFIG  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.triage.train import _fit, cv_folds, load_dataset, make_xgb  # noqa: E402

# The search space: each knob, and the range it is drawn from.
SPACE = {
    "max_depth": randint(2, 7),                  # tree depth 2-6 (current: 4)
    "n_estimators": randint(100, 801),           # number of trees (current: 400)
    "learning_rate": loguniform(0.01, 0.3),      # step size per tree (current: 0.05)
    "subsample": uniform(0.6, 0.4),              # share of stars per tree 0.6-1.0 (0.8)
    "colsample_bytree": uniform(0.5, 0.5),       # share of features per tree 0.5-1.0 (0.8)
    "min_child_weight": randint(1, 11),          # min evidence in a leaf (current: 2)
    "reg_lambda": loguniform(0.1, 10),           # L2 penalty on leaf values (current: 1.0)
    "gamma": uniform(0, 1),                      # min loss gain to split (current: 0)
}
KNOBS = list(SPACE)


def base_xgb() -> XGBClassifier:
    return XGBClassifier(objective="multi:softprob", tree_method="hist", eval_metric="mlogloss",
                         random_state=CONFIG.random_state, n_jobs=1)


def search(X, y, groups, n_iter: int, seed: int) -> RandomizedSearchCV:
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    s = RandomizedSearchCV(base_xgb(), SPACE, n_iter=n_iter, scoring="f1_macro",
                           cv=inner, random_state=seed, n_jobs=-1, refit=True)
    s.fit(X, y, groups=groups, sample_weight=compute_sample_weight("balanced", y))
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-iter", type=int, default=40, help="random settings tried per search")
    args = ap.parse_args()
    setup_logging()
    cfg = CONFIG
    ds = load_dataset(pd.read_parquet(cfg.paths.processed / "features.parquet"))
    folds = cv_folds(ds, cfg)
    t_start = time.perf_counter()

    rows = []
    for k, (tr, va) in enumerate(folds, 1):
        Xtr, ytr, gtr = ds.X.iloc[tr], ds.y[tr], ds.groups[tr]
        s = search(Xtr, ytr, gtr, args.n_iter, cfg.random_state + k)
        tuned_f1 = f1_score(ds.y[va], s.best_estimator_.predict(ds.X.iloc[va]), average="macro")
        default = _fit(make_xgb(cfg), Xtr, ytr)
        default_f1 = f1_score(ds.y[va], default.predict(ds.X.iloc[va]), average="macro")
        rows.append({"fold": k, "default_f1": float(default_f1), "tuned_f1": float(tuned_f1),
                     "inner_best_f1": float(s.best_score_),
                     "best_params": {p: (float(v) if isinstance(v, float) else int(v))
                                     for p, v in s.best_params_.items()}})
        print(f"outer fold {k}: default {default_f1:.3f}  tuned {tuned_f1:.3f}  "
              f"(inner CV best {s.best_score_:.3f})")

    d = np.array([r["default_f1"] for r in rows])
    t = np.array([r["tuned_f1"] for r in rows])
    inner = np.array([r["inner_best_f1"] for r in rows])
    gain = t - d
    # final search on all 400 stars: the settings to use IF tuning were adopted
    final = search(ds.X, ds.y, ds.groups, args.n_iter, cfg.random_state)
    final_params = {p: (round(float(v), 4) if isinstance(v, float) else int(v))
                    for p, v in final.best_params_.items()}
    minutes = (time.perf_counter() - t_start) / 60

    summary = {
        "n_iter": args.n_iter, "outer_folds": len(folds), "inner_folds": 3,
        "fits": len(folds) * args.n_iter * 3 + args.n_iter * 3,
        "default_macro_f1": {"mean": float(d.mean()), "std": float(d.std(ddof=1))},
        "tuned_macro_f1_nested": {"mean": float(t.mean()), "std": float(t.std(ddof=1))},
        "inner_cv_best_mean": float(inner.mean()),
        "gain": {"mean": float(gain.mean()), "std": float(gain.std(ddof=1)),
                 "folds_better": int((gain > 0).sum())},
        "folds": rows, "final_best_params": final_params,
        "final_inner_cv_score": float(final.best_score_), "minutes": minutes,
    }
    lines = [f"# XGBoost hyperparameter tuning (nested CV, {args.n_iter} random settings "
             f"per search, {summary['fits']} model fits, {minutes:.1f} min)", "",
             "| | Macro-F1 (5 outer folds) |", "|---|---|",
             f"| Current hand-chosen settings | {d.mean():.3f} ± {d.std(ddof=1):.3f} |",
             f"| Tuned (nested, honest) | {t.mean():.3f} ± {t.std(ddof=1):.3f} |",
             f"| Inner-CV best score (optimistic, for comparison) | {inner.mean():.3f} |",
             f"| Gain | {gain.mean():+.3f} ± {gain.std(ddof=1):.3f} "
             f"(tuned better in {(gain > 0).sum()} of {len(folds)} folds) |", "",
             "## Best settings per outer fold", "",
             "| Fold | Default F1 | Tuned F1 | " + " | ".join(KNOBS) + " |",
             "|---|---|---|" + "---|" * len(KNOBS)]
    for r in rows:
        bp = r["best_params"]
        lines.append(f"| {r['fold']} | {r['default_f1']:.3f} | {r['tuned_f1']:.3f} | " + " | ".join(
            f"{bp[p]:.3g}" if isinstance(bp[p], float) else str(bp[p]) for p in KNOBS) + " |")
    depth_votes = Counter(r["best_params"]["max_depth"] for r in rows)
    lines += ["", f"max_depth chosen per fold: {dict(depth_votes)}", "",
              "## Best settings on all 400 stars (if tuning were adopted)", "",
              "| " + " | ".join(KNOBS) + " |", "|" + "---|" * len(KNOBS),
              "| " + " | ".join(str(final_params[p]) for p in KNOBS) + " |", "",
              "Current settings: max_depth 4, n_estimators 400, learning_rate 0.05, "
              "subsample 0.8, colsample_bytree 0.8, min_child_weight 2, reg_lambda 1.0, "
              "gamma 0."]
    out = cfg.paths.outputs
    out.mkdir(parents=True, exist_ok=True)
    (out / "xgb_tuning.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "xgb_tuning.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
