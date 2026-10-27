"""Feature selection for the XGBoost classifier, evaluated with nested cross-validation.

    python scripts/select_features.py

Two selection methods, both run inside each outer training fold only (the outer
test fold never influences which features are kept):

1. SHAP ranking: fit XGBoost on the training part, rank features by mean |SHAP|,
   keep the top k. Scored for k = 5, 10, 15, 20, 25, 30, 40 and all.
2. Recursive feature elimination (RFE): repeatedly refit and drop the 4 least
   important features (XGBoost gain importance) down to k.

Then the honest estimate of "XGBoost + feature selection": inside each outer
training part, an inner 3-fold grouped CV picks k (SHAP ranking), and the model
with those k features is scored once on the outer test fold.

Writes outputs/feature_selection.json (read by scripts/save_evaluation.py for the
app) and outputs/feature_selection.md. The deployed model is not changed here.
"""

# ── In plain words ───────────────────────────────────────────────────────────
# The model gets 56 numbers per star. Some matter a lot, some barely, and with
# only 400 stars the useless ones can let the model learn chance patterns.
# Feature selection asks: how few features do we need, and does dropping the
# weak ones help? The answer is a curve (score vs number of features kept).
# Ranking and choosing are done on training stars only, so the test stars stay
# untouched and the scores are honest.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from config import CONFIG  # noqa: E402
from src.log import setup_logging  # noqa: E402
from src.triage.train import _fit, cv_folds, load_dataset, make_xgb  # noqa: E402

KS = (5, 10, 15, 20, 25, 30, 40)
RFE_STEP = 4


def shap_ranking(X: pd.DataFrame, y: np.ndarray) -> list[str]:
    """Features sorted by mean |SHAP| (summed over the 4 classes), most important first."""
    m = _fit(make_xgb(CONFIG), X, y)
    dm = xgb.DMatrix(X, feature_names=list(X.columns), missing=np.nan)
    sv = m.get_booster().predict(dm, pred_contribs=True)      # (n, classes, features + bias)
    imp = np.abs(np.asarray(sv)[:, :, :-1]).mean(axis=0).sum(axis=0)
    return [X.columns[i] for i in np.argsort(-imp)]


def rfe_ranking(X: pd.DataFrame, y: np.ndarray) -> list[str]:
    """Order in which RFE would keep features: last survivors first."""
    remaining, dropped = list(X.columns), []
    while len(remaining) > RFE_STEP:
        m = _fit(make_xgb(CONFIG), X[remaining], y)
        gain = m.get_booster().get_score(importance_type="gain")
        order = sorted(remaining, key=lambda f: gain.get(f, 0.0))   # weakest first
        drop = order[:RFE_STEP]
        dropped = drop + dropped
        remaining = [f for f in remaining if f not in drop]
    return remaining + dropped


def score(Xtr, ytr, Xte, yte, cols) -> float:
    m = _fit(make_xgb(CONFIG), Xtr[cols], ytr)
    return float(f1_score(yte, m.predict(Xte[cols]), average="macro"))


def main() -> None:
    setup_logging()
    cfg = CONFIG
    ds = load_dataset(pd.read_parquet(cfg.paths.processed / "features.parquet"))
    n_all = len(ds.features)
    ks = [k for k in KS if k < n_all] + [n_all]
    folds = cv_folds(ds, cfg)

    curves = {"shap": {k: [] for k in ks}, "rfe": {k: [] for k in ks}}
    nested = []
    for fi, (tr, va) in enumerate(folds, 1):
        Xtr, ytr, gtr = ds.X.iloc[tr], ds.y[tr], ds.groups[tr]
        Xte, yte = ds.X.iloc[va], ds.y[va]
        rank_shap, rank_rfe = shap_ranking(Xtr, ytr), rfe_ranking(Xtr, ytr)
        for k in ks:
            curves["shap"][k].append(score(Xtr, ytr, Xte, yte, rank_shap[:k]))
            curves["rfe"][k].append(score(Xtr, ytr, Xte, yte, rank_rfe[:k]))
        # nested choice of k: inner 3-fold CV on the training part only
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cfg.random_state + fi)
        inner_scores = {k: [] for k in ks}
        for itr, iva in inner.split(Xtr, ytr, gtr):
            r = shap_ranking(Xtr.iloc[itr], ytr[itr])
            for k in ks:
                inner_scores[k].append(score(Xtr.iloc[itr], ytr[itr], Xtr.iloc[iva], ytr[iva],
                                             r[:k]))
        best_k = max(ks, key=lambda k: (np.mean(inner_scores[k]), -k))
        nested.append({"fold": fi, "chosen_k": best_k,
                       "selected_f1": score(Xtr, ytr, Xte, yte, rank_shap[:best_k]),
                       "all_features_f1": curves["shap"][n_all][-1]})
        print(f"outer fold {fi}: chosen k = {best_k}, selected {nested[-1]['selected_f1']:.3f}, "
              f"all {nested[-1]['all_features_f1']:.3f}")

    def ms(v):
        return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}

    sel = np.array([r["selected_f1"] for r in nested])
    full = np.array([r["all_features_f1"] for r in nested])
    gain = sel - full
    final_rank = shap_ranking(ds.X, ds.y)
    best_curve_k = max(ks, key=lambda k: (np.mean(curves["shap"][k]), -k))
    adopt = bool(gain.mean() > gain.std(ddof=1)) and gain.mean() > 0
    out = {
        "n_features_all": n_all,
        "curve": [{"k": k, "shap": ms(curves["shap"][k]), "rfe": ms(curves["rfe"][k])}
                  for k in ks],
        "nested": {"selected": ms(sel), "all_features": ms(full),
                   "gain": {**ms(gain), "folds_better": int((gain > 0).sum())},
                   "chosen_k_per_fold": [r["chosen_k"] for r in nested]},
        "best_k_on_curve": best_curve_k,
        "ranking_all_stars": final_rank,
        "adopted": adopt,
        "decision": ("Selected features adopted: the gain is larger than the fold-to-fold spread."
                     if adopt else
                     "All features kept: selection did not beat the full set by more than the "
                     "fold-to-fold spread."),
    }
    lines = [f"# Feature selection ({len(ds.y)} stars, {len(folds)} outer folds)", "",
             "| Features kept | SHAP ranking macro-F1 | RFE macro-F1 |", "|---|---|---|"]
    for row in out["curve"]:
        lines.append(f"| {row['k']} | {row['shap']['mean']:.3f} ± {row['shap']['std']:.3f} | "
                     f"{row['rfe']['mean']:.3f} ± {row['rfe']['std']:.3f} |")
    lines += ["", "## Nested estimate (k chosen inside each training fold)", "",
              f"- With selection: {sel.mean():.3f} ± {sel.std(ddof=1):.3f}",
              f"- All {n_all} features: {full.mean():.3f} ± {full.std(ddof=1):.3f}",
              f"- Gain: {gain.mean():+.3f} ± {gain.std(ddof=1):.3f} "
              f"(better in {(gain > 0).sum()} of {len(folds)} folds)",
              f"- k chosen per fold: {Counter(r['chosen_k'] for r in nested)}", "",
              f"Decision: {out['decision']}", "",
              "## Top 15 features (SHAP, all 400 stars)", "",
              ", ".join(final_rank[:15])]
    path = cfg.paths.outputs
    path.mkdir(parents=True, exist_ok=True)
    (path / "feature_selection.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (path / "feature_selection.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
