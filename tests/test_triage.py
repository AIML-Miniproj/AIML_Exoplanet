"""Triage training and inference on a small synthetic feature table."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from config import CONFIG, Paths
from src.triage.labels import CLASSES
from src.triage.model import TriageModel, apply_calibration, confidence_label, load_model
from src.triage.train import planet_recall_at_eb_rate, train
from src.vetting.features import FEATURE_NAMES


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("triage")
    paths = Paths(root=root, data=root / "data", raw=root / "raw", processed=root / "processed",
                  labels=root / "labels", logs=root / "logs", cache=root / "cache",
                  examples=root / "examples", models=root / "models", outputs=root / "outputs")
    return replace(CONFIG, paths=paths)


def synthetic_table(n_per_class=70, seed=0):
    """Features with class structure similar to the real ones, plus noise columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, cls in enumerate(CLASSES):
        for i in range(n_per_class):
            r = {f: rng.normal() for f in FEATURE_NAMES}
            # class signal goes into model inputs (per-transit features, not raw SNRs)
            r["odd_even_sigma_per_transit"] = rng.gamma(2, 1) + (6 if cls == "EB" else 0)
            r["secondary_snr_per_transit"] = rng.normal(0, 1) + (10 if cls == "EB" else 0)
            r["ingress_to_total_ratio"] = rng.uniform(0.05, 0.2) + (0.25 if cls == "EB" else 0)
            r["depth_snr_per_transit"] = rng.normal(1, 1) + (8 if cls in ("PLANET", "EB") else 0)
            r["ls_power_1"] = rng.uniform(0, 0.1) + (0.6 if cls == "VARIABLE" else 0)
            r["ruwe"] = np.nan                                  # all-missing: dropped
            if rng.random() < 0.1:
                r["teff"] = np.nan                              # some missing values
            r.update(tic_id=1000 * k + i, label=cls, status="ok", source="test")
            rows.append(r)
    rows.append({"tic_id": 99999, "label": "NOISE", "status": "DataDownloadError"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained(cfg):
    feats = synthetic_table()
    labels = feats[["tic_id", "label"]]
    return train(feats, labels, cfg), cfg


def test_training_writes_native_model_and_meta(trained):
    res, cfg = trained
    assert (cfg.paths.models / "triage_xgb.json").exists()
    meta = json.loads((cfg.paths.models / "triage_meta.json").read_text())
    assert meta["classes"] == CLASSES
    assert "ruwe" not in meta["features"] and "n_sectors" not in meta["features"]
    assert set(meta["calibration"]) == set(CLASSES)
    assert meta["n_train"] == 4 * 70 and meta["n_test"] == 0  # failed row excluded
    assert "StratifiedGroupKFold" in meta["evaluation"]
    for name in ("xgboost", "xgboost_calibrated", "logistic_regression", "random_forest"):
        assert name in meta["metrics"]
    for f in ("triage_confusion.png", "triage_reliability.png", "triage_shap_summary.png"):
        assert (cfg.paths.outputs / f).exists()


def test_metrics_reasonable_on_separable_data(trained):
    res, _ = trained
    m = res["models"]["xgboost_calibrated"]
    assert m["cv"]["macro_f1"] > 0.85
    assert set(m["cv"]["per_class"]) == set(CLASSES)
    assert np.allclose(np.sum(m["cv"]["confusion_normalized"], axis=1), 1, atol=0.01)
    f = m["folds"]["macro_f1"]
    assert f["mean"] > 0.85 and 0 <= f["std"] < 0.15
    for name in ("xgboost", "logistic_regression", "random_forest"):
        assert set(res["models"][name]["folds"]["per_class"]) == set(CLASSES)


def test_folds_are_grouped_by_tic():
    from src.triage.train import cv_folds, load_dataset

    feats = synthetic_table(n_per_class=20)
    feats = pd.concat([feats, feats[feats["status"] == "ok"].head(10)])  # repeat 10 TICs
    ds = load_dataset(feats)
    folds = cv_folds(ds, CONFIG)
    assert len(folds) == CONFIG.cv_folds
    for tr, va in folds:
        assert not set(ds.groups[tr]) & set(ds.groups[va])
    assert sorted(np.concatenate([va for _, va in folds])) == list(range(len(ds.y)))


def test_model_predict_and_explain(trained):
    _, cfg = trained
    model = TriageModel.load(cfg.paths.models)
    eb_row = synthetic_table(5, seed=9).query("label == 'EB'").iloc[0].to_dict()
    out = model.predict(eb_row, cfg)
    assert out["status"] == "ok" and out["predicted_class"] == "EB"
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-6
    assert out["confidence"] in ("High", "Medium", "Low")
    assert len(out["top_contributions"]) == 5
    top = {c["feature"] for c in out["top_contributions"]}
    assert top & {"odd_even_sigma_per_transit", "secondary_snr_per_transit",
                  "ingress_to_total_ratio"}
    # missing features are tolerated
    assert model.predict({"sde": 12.0}, cfg)["status"] == "ok"
    assert load_model(cfg) is not None


def test_load_model_missing(tmp_path):
    paths = replace(CONFIG.paths, models=tmp_path)
    assert load_model(replace(CONFIG, paths=paths)) is None


def test_calibration_and_confidence_helpers():
    curves = {c: {"x": [0, 1], "y": [0, 1]} for c in CLASSES}
    p = np.array([[0.7, 0.1, 0.1, 0.1]])
    np.testing.assert_allclose(apply_calibration(p, curves, CLASSES), p)
    assert confidence_label(0.9) == "High" and confidence_label(0.6) == "Medium"
    assert confidence_label(0.3) == "Low"


def test_planet_recall_at_eb_rate():
    y = np.array([0] * 10 + [1] * 20)
    p = np.r_[np.linspace(0.5, 1.0, 10), np.linspace(0.0, 0.6, 20)]
    thr, rec = planet_recall_at_eb_rate(p, y, 0.05)
    assert np.mean(p[y == 1] >= thr) <= 0.05
    assert 0 < rec <= 1
