"""Train and evaluate the 4-class triage classifier.

- Splits are grouped by TIC (``StratifiedGroupKFold``), so one star never appears
  on both sides of a split.
- The dataset is small (~25 stars per class), so there is no separate test set:
  5-fold grouped cross-validation scores every star once, and metrics are given
  as mean +/- standard deviation across folds.
- XGBoost is compared with logistic regression and a random forest on the same folds.
- Probabilities are calibrated with per-class sigmoid curves. Calibrated scores are
  themselves cross-validated (each fold's curve is fitted on the other folds).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,  # noqa: E402
                             log_loss, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from config import CONFIG, Config  # noqa: E402
from src.triage.labels import CLASSES, label_hash  # noqa: E402
from src.triage.model import META_FILE, MODEL_FILE, apply_calibration  # noqa: E402
from src.vetting.features import MODEL_FEATURES  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    rows: pd.DataFrame
    features: list[str]


def load_dataset(features: pd.DataFrame) -> Dataset:
    """Successful pipeline runs only; drop features that are missing for every star."""
    ok = features[features["status"] == "ok"].reset_index(drop=True)
    feats = [f for f in MODEL_FEATURES if f in ok and ok[f].notna().any()]
    dropped = sorted(set(MODEL_FEATURES) - set(feats))
    if dropped:
        log.info("Dropping features with no values: %s", dropped)
    X = ok[feats].astype(float).replace([np.inf, -np.inf], np.nan)
    y = ok["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    return Dataset(X, y, ok["tic_id"].to_numpy(), ok, feats)


def make_xgb(cfg: Config) -> XGBClassifier:
    return XGBClassifier(objective="multi:softprob", n_estimators=400, max_depth=4,
                         learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=2, reg_lambda=1.0, tree_method="hist",
                         eval_metric="mlogloss", random_state=cfg.random_state,
                         n_jobs=cfg.workers())


def make_baselines(cfg: Config) -> dict[str, Any]:
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                   class_weight="balanced_subsample",
                                   random_state=cfg.random_state, n_jobs=cfg.workers())),
    }


def _fit(model, X, y):
    w = compute_sample_weight("balanced", y)
    if isinstance(model, XGBClassifier):
        model.fit(X, y, sample_weight=w, verbose=False)
    else:
        model.fit(X, y)
    return model


def cv_folds(ds: Dataset, cfg: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped, stratified folds: a star (TIC) is never in both halves of a split.

    The same folds are used for every model, so their scores are comparable.
    """
    sgkf = StratifiedGroupKFold(n_splits=cfg.cv_folds, shuffle=True,
                                random_state=cfg.random_state)
    folds = list(sgkf.split(ds.X, ds.y, ds.groups))
    for tr, va in folds:
        assert not set(ds.groups[tr]) & set(ds.groups[va]), "a TIC is on both sides of a fold"
    return folds


def cv_predict(factory, ds: Dataset, folds, cfg: Config
               ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Out-of-fold probabilities for every star, plus the metrics of each fold."""
    oof = np.zeros((len(ds.y), len(CLASSES)))
    per_fold = []
    for tr, va in folds:
        m = _fit(factory(), ds.X.iloc[tr], ds.y[tr])
        oof[va] = m.predict_proba(ds.X.iloc[va])
        per_fold.append(metrics(ds.y[va], oof[va], cfg))
    return oof, per_fold


def fold_summary(per_fold: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation across folds of the headline metrics."""
    def ms(vals):
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0} \
            if v.size else {"mean": float("nan"), "std": float("nan")}

    out = {k: ms([m[k] for m in per_fold])
           for k in ("macro_f1", "accuracy", "log_loss", "planet_recall_at_eb_rate")}
    out["per_class"] = {c: {k: ms([m["per_class"][c][k] for m in per_fold])
                            for k in ("precision", "recall", "f1", "pr_auc")} for c in CLASSES}
    return out


def fit_calibration(oof: np.ndarray, y: np.ndarray,
                    n_points: int = 101) -> dict[str, dict[str, list[float]]]:
    """Per-class sigmoid (Platt) curves, one class against the rest.

    A sigmoid has two parameters, so it stays stable on ~100 stars where an
    isotonic curve would overfit. Stored as sampled (x, y) points so inference
    only needs interpolation.
    """
    x = np.linspace(0, 1, n_points)
    curves = {}
    for k, cls in enumerate(CLASSES):
        z = np.log(np.clip(oof[:, [k]], 1e-6, 1 - 1e-6) / (1 - np.clip(oof[:, [k]], 1e-6,
                                                                         1 - 1e-6)))
        lr = LogisticRegression(C=1e6).fit(z, (y == k).astype(int))
        zx = np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
        curves[cls] = {"x": x.tolist(), "y": lr.predict_proba(zx[:, None])[:, 1].tolist()}
    return curves


def cross_calibrate(oof: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """Calibrated out-of-fold probabilities: each fold's curves are fitted on the
    other folds only, so no star is calibrated with its own label."""
    out = np.zeros_like(oof)
    for tr, va in folds:
        out[va] = apply_calibration(oof[va], fit_calibration(oof[tr], y[tr]), CLASSES)
    return out


def planet_recall_at_eb_rate(p_planet: np.ndarray, y: np.ndarray,
                             max_eb_rate: float) -> tuple[float, float]:
    """Lowest P(planet) threshold letting through at most ``max_eb_rate`` of EBs,
    and the share of real planets kept at that threshold."""
    pl, eb = CLASSES.index("PLANET"), CLASSES.index("EB")
    p_eb = np.sort(p_planet[y == eb])
    if p_eb.size == 0 or (y == pl).sum() == 0:
        return float("nan"), float("nan")
    k = int(np.floor(max_eb_rate * p_eb.size))           # EBs allowed above threshold
    thr = float(np.nextafter(p_eb[-k - 1], 1)) if k < p_eb.size else 0.0
    return thr, float(np.mean(p_planet[y == pl] >= thr))


def _normalise(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1)
    return p / p.sum(axis=1, keepdims=True)


def metrics(y: np.ndarray, proba: np.ndarray, cfg: Config) -> dict[str, Any]:
    pred = proba.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)),
                                                 zero_division=0)
    per = {}
    for k, cls in enumerate(CLASSES):
        yk = (y == k).astype(int)
        per[cls] = {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]),
                    "pr_auc": float(average_precision_score(yk, proba[:, k])) if yk.any()
                    else float("nan"), "support": int(s[k])}
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    thr, rec = planet_recall_at_eb_rate(proba[:, CLASSES.index("PLANET")], y,
                                        cfg.eb_contamination_target)
    return {"per_class": per, "macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float(np.mean(pred == y)),
            "log_loss": float(log_loss(y, _normalise(proba), labels=range(len(CLASSES)))),
            "confusion": cm.tolist(), "confusion_normalized": np.round(cmn, 3).tolist(),
            "planet_threshold_at_eb_rate": thr, "planet_recall_at_eb_rate": rec,
            "eb_rate": cfg.eb_contamination_target}


# ── Plots ────────────────────────────────────────────────────

def plot_confusion(cmn: list[list[float]], path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cmn[i][j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i][j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability(y: np.ndarray, raw: np.ndarray, cal: np.ndarray, path) -> None:
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(14, 3.6), sharey=True)
    bins = np.linspace(0, 1, 11)
    for k, (ax, cls) in enumerate(zip(axes, CLASSES)):
        yk = (y == k).astype(float)
        for probs, label, style in ((raw, "raw", "o--"), (cal, "calibrated", "s-")):
            idx = np.clip(np.digitize(probs[:, k], bins) - 1, 0, 9)
            xs = [probs[idx == b, k].mean() for b in range(10) if (idx == b).sum() >= 3]
            ys = [yk[idx == b].mean() for b in range(10) if (idx == b).sum() >= 3]
            ax.plot(xs, ys, style, label=label, ms=4)
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8)
        ax.set_title(cls)
        ax.set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed fraction")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_shap_summary(model: XGBClassifier, X: pd.DataFrame, path, top: int = 15
                      ) -> dict[str, list[tuple[str, float]]]:
    """Mean |SHAP| per feature and class (TreeExplainer); returns the top features."""
    import shap

    ex = shap.TreeExplainer(model)
    sv = np.asarray(ex.shap_values(X))           # (n, features, classes)
    if sv.ndim == 3 and sv.shape[0] == len(CLASSES) and sv.shape[-1] != len(CLASSES):
        sv = np.moveaxis(sv, 0, -1)
    imp = np.abs(sv).mean(axis=0)                # (features, classes)
    order = np.argsort(-imp.sum(1))[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    left = np.zeros(len(order))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#999999"]
    for k, cls in enumerate(CLASSES):
        ax.barh([X.columns[i] for i in order], imp[order, k], left=left, color=colors[k],
                label=cls)
        left += imp[order, k]
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("Which features drive the classifier")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {cls: [(X.columns[i], float(imp[i, k])) for i in np.argsort(-imp[:, k])[:8]]
            for k, cls in enumerate(CLASSES)}


# ── Main entry ───────────────────────────────────────────────

def train(features: pd.DataFrame, labels: pd.DataFrame, cfg: Config = CONFIG) -> dict[str, Any]:
    """Grouped k-fold evaluation of XGBoost and two baselines, then a final fit on all stars.

    With ~25 stars per class there is no separate held-out test set: every star is
    scored exactly once by a model that never saw it (out-of-fold), and metrics are
    reported as mean +/- standard deviation across the folds.
    """
    ds = load_dataset(features)
    folds = cv_folds(ds, cfg)
    counts = np.bincount(ds.y, minlength=len(CLASSES)).tolist()
    log.info("%d stars, class counts %s, %d grouped folds", len(ds.y), counts, len(folds))

    results: dict[str, Any] = {"models": {}}
    factories = {"xgboost": lambda: make_xgb(cfg),
                 **{k: (lambda v=v: __import__("sklearn").base.clone(v))
                    for k, v in make_baselines(cfg).items()}}
    for name, factory in factories.items():
        oof, per_fold = cv_predict(factory, ds, folds, cfg)
        results["models"][name] = {"cv": metrics(ds.y, oof, cfg),
                                   "folds": fold_summary(per_fold)}
        fs = results["models"][name]["folds"]["macro_f1"]
        log.info("%-20s macro-F1 %.3f +/- %.3f (5-fold, grouped by TIC)", name, fs["mean"],
                 fs["std"])
        if name == "xgboost":
            xgb_oof = oof

    cal_oof = cross_calibrate(xgb_oof, ds.y, folds)
    results["models"]["xgboost_calibrated"] = {
        "cv": metrics(ds.y, cal_oof, cfg),
        "folds": fold_summary([metrics(ds.y[va], cal_oof[va], cfg) for _, va in folds])}
    curves = fit_calibration(xgb_oof, ds.y)                      # deployed calibration
    thr, rec = planet_recall_at_eb_rate(cal_oof[:, CLASSES.index("PLANET")], ds.y,
                                        cfg.eb_contamination_target)

    xgb_model = _fit(make_xgb(cfg), ds.X, ds.y)                  # final model: all stars
    out = cfg.paths.outputs
    out.mkdir(parents=True, exist_ok=True)
    plot_confusion(results["models"]["xgboost_calibrated"]["cv"]["confusion_normalized"],
                   out / "triage_confusion.png",
                   "Out-of-fold predictions (normalised by true class)")
    plot_reliability(ds.y, xgb_oof, cal_oof, out / "triage_reliability.png")
    shap_top = plot_shap_summary(xgb_model, ds.X, out / "triage_shap_summary.png")

    cfg.paths.models.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(str(cfg.paths.models / MODEL_FILE))
    meta = {
        "classes": CLASSES,
        "features": ds.features,
        "calibration": curves,
        "calibration_method": "sigmoid (Platt), one class against the rest",
        "thresholds": {"planet_at_eb_contamination": thr,
                       "eb_contamination_target": cfg.eb_contamination_target,
                       "cv_planet_recall_at_threshold": rec,
                       "uncertain": cfg.uncertain_threshold, "high": cfg.conf_high},
        "trained_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "label_hash": label_hash(labels),
        "evaluation": f"{len(folds)}-fold StratifiedGroupKFold by TIC, no held-out test set",
        "n_train": int(len(ds.y)), "n_test": 0,
        "class_counts_train": dict(zip(CLASSES, counts)),
        "metrics": {k: v["cv"] for k, v in results["models"].items()},
        "fold_metrics": {k: v["folds"] for k, v in results["models"].items()},
        "shap_top_features": shap_top,
        "xgb_params": {k: v for k, v in xgb_model.get_params().items()
                       if isinstance(v, (int, float, str, bool)) or v is None},
    }
    (cfg.paths.models / META_FILE).write_text(json.dumps(meta, indent=1, default=float),
                                              encoding="utf-8")
    results["meta"] = meta
    return results
