"""Train and evaluate the 4-class triage classifier.

- Splits are grouped by TIC (``StratifiedGroupKFold``), so one star never appears
  in both training and test data.
- One grouped fold is held out as the test set; 5-fold grouped cross-validation on
  the rest gives out-of-fold predictions for calibration and model comparison.
- XGBoost is compared with logistic regression and a random forest on the same splits.
- Probabilities are calibrated with per-class isotonic curves fitted on the
  out-of-fold predictions, never on the test set.
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
from sklearn.isotonic import IsotonicRegression  # noqa: E402
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


def split_test(ds: Dataset, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    n = max(2, round(1 / cfg.test_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=cfg.random_state)
    train_idx, test_idx = next(sgkf.split(ds.X, ds.y, ds.groups))
    return train_idx, test_idx


def oof_predict(factory, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                cfg: Config) -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=cfg.cv_folds, shuffle=True,
                                random_state=cfg.random_state + 1)
    oof = np.zeros((len(y), len(CLASSES)))
    for tr, va in sgkf.split(X, y, groups):
        m = _fit(factory(), X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])
    return oof


def fit_calibration(oof: np.ndarray, y: np.ndarray) -> dict[str, dict[str, list[float]]]:
    """Per-class isotonic curves (one-vs-rest), stored as plain lists."""
    curves = {}
    for k, cls in enumerate(CLASSES):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(oof[:, k], (y == k).astype(float))
        curves[cls] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    return curves


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
    ds = load_dataset(features)
    tr, te = split_test(ds, cfg)
    assert not set(ds.groups[tr]) & set(ds.groups[te]), "a TIC is in both train and test"
    Xtr, ytr, gtr = ds.X.iloc[tr], ds.y[tr], ds.groups[tr]
    Xte, yte = ds.X.iloc[te], ds.y[te]
    log.info("Train %d / test %d stars; class counts train %s test %s", len(tr), len(te),
             np.bincount(ytr, minlength=4).tolist(), np.bincount(yte, minlength=4).tolist())

    results: dict[str, Any] = {"models": {}}
    factories = {"xgboost": lambda: make_xgb(cfg),
                 **{k: (lambda v=v: __import__("sklearn").base.clone(v))
                    for k, v in make_baselines(cfg).items()}}
    xgb_oof = None
    for name, factory in factories.items():
        oof = oof_predict(factory, Xtr, ytr, gtr, cfg)
        model = _fit(factory(), Xtr, ytr)
        test_p = model.predict_proba(Xte)
        results["models"][name] = {"cv": metrics(ytr, oof, cfg), "test": metrics(yte, test_p, cfg)}
        log.info("%-20s CV macro-F1 %.3f | test macro-F1 %.3f", name,
                 results["models"][name]["cv"]["macro_f1"],
                 results["models"][name]["test"]["macro_f1"])
        if name == "xgboost":
            xgb_model, xgb_oof, xgb_test_raw = model, oof, test_p

    curves = fit_calibration(xgb_oof, ytr)
    cal_oof = apply_calibration(xgb_oof, curves, CLASSES)
    cal_test = apply_calibration(xgb_test_raw, curves, CLASSES)
    results["models"]["xgboost_calibrated"] = {"cv": metrics(ytr, cal_oof, cfg),
                                               "test": metrics(yte, cal_test, cfg)}
    # Planet threshold chosen on out-of-fold data, then checked on the test set.
    thr, _ = planet_recall_at_eb_rate(cal_oof[:, 0], ytr, cfg.eb_contamination_target)
    pl, eb = CLASSES.index("PLANET"), CLASSES.index("EB")
    test_eb_rate = float(np.mean(cal_test[yte == eb, pl] >= thr)) if (yte == eb).any() else None
    test_pl_rec = float(np.mean(cal_test[yte == pl, pl] >= thr)) if (yte == pl).any() else None

    out = cfg.paths.outputs
    out.mkdir(parents=True, exist_ok=True)
    plot_confusion(results["models"]["xgboost_calibrated"]["test"]["confusion_normalized"],
                   out / "triage_confusion.png", "Test set (normalised by true class)")
    plot_reliability(yte, xgb_test_raw, cal_test, out / "triage_reliability.png")
    shap_top = plot_shap_summary(xgb_model, Xtr, out / "triage_shap_summary.png")

    cfg.paths.models.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(str(cfg.paths.models / MODEL_FILE))
    meta = {
        "classes": CLASSES,
        "features": ds.features,
        "calibration": curves,
        "thresholds": {"planet_at_eb_contamination": thr,
                       "eb_contamination_target": cfg.eb_contamination_target,
                       "test_eb_rate_at_threshold": test_eb_rate,
                       "test_planet_recall_at_threshold": test_pl_rec,
                       "uncertain": cfg.uncertain_threshold, "high": cfg.conf_high},
        "trained_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "label_hash": label_hash(labels),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "class_counts_train": dict(zip(CLASSES, np.bincount(ytr, minlength=4).tolist())),
        "class_counts_test": dict(zip(CLASSES, np.bincount(yte, minlength=4).tolist())),
        "test_tic_ids": sorted(int(t) for t in ds.groups[te]),
        "metrics": {k: v["test"] for k, v in results["models"].items()},
        "cv_metrics": {k: v["cv"] for k, v in results["models"].items()},
        "shap_top_features": shap_top,
        "xgb_params": {k: v for k, v in xgb_model.get_params().items()
                       if isinstance(v, (int, float, str, bool)) or v is None},
    }
    (cfg.paths.models / META_FILE).write_text(json.dumps(meta, indent=1, default=float),
                                              encoding="utf-8")
    results["meta"] = meta
    return results
