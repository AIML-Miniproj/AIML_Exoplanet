"""Load the trained triage classifier and score one set of features.

Files (both plain JSON, no pickles):
- ``models/triage_xgb.json``: the XGBoost model in its native format
- ``models/triage_meta.json``: feature order, class order, calibration curves,
  thresholds, training date, label hash and test metrics
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from config import CONFIG, Config

log = logging.getLogger(__name__)

MODEL_FILE = "triage_xgb.json"
META_FILE = "triage_meta.json"


def confidence_label(p: float, cfg: Config = CONFIG) -> str:
    # High >= 80%, Medium >= 50%, Low below (the router turns Low into Needs review).
    if p >= cfg.conf_high:
        return "High"
    if p >= cfg.uncertain_threshold:
        return "Medium"
    return "Low"


def apply_calibration(probs: np.ndarray, curves: dict[str, dict[str, list[float]]] | None,
                      classes: list[str]) -> np.ndarray:
    """Map raw class probabilities through per-class sigmoid (Platt) curves, stored as
    sampled points, then renormalise."""
    if not curves:
        return probs
    out = np.empty_like(probs)
    for k, cls in enumerate(classes):
        c = curves.get(cls)
        out[:, k] = np.interp(probs[:, k], c["x"], c["y"]) if c else probs[:, k]
    out = np.clip(out, 1e-6, None)
    return out / out.sum(axis=1, keepdims=True)


@dataclass
class TriageModel:
    model: Any
    meta: dict[str, Any]

    @property
    def classes(self) -> list[str]:
        return self.meta["classes"]

    @property
    def features(self) -> list[str]:
        return self.meta["features"]

    @property
    def version(self) -> str:
        return f"{self.meta.get('trained_at', '?')}-{self.meta.get('label_hash', '?')}"

    @classmethod
    def load(cls, models_dir: Path | None = None) -> TriageModel | None:
        from xgboost import XGBClassifier

        d = models_dir or CONFIG.paths.models
        mp, jp = d / MODEL_FILE, d / META_FILE
        if not (mp.exists() and jp.exists()):
            return None
        m = XGBClassifier()
        m.load_model(str(mp))
        return cls(m, json.loads(jp.read_text(encoding="utf-8")))

    def _matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        return np.array([[np.nan if r.get(f) is None else float(r[f]) for f in self.features]
                         for r in rows], dtype=float)

    def predict_proba(self, rows: list[dict[str, float]], calibrated: bool = True) -> np.ndarray:
        p = self.model.predict_proba(self._matrix(rows))
        return apply_calibration(p, self.meta.get("calibration"), self.classes) if calibrated else p

    def contributions(self, row: dict[str, float]) -> np.ndarray:
        """Exact tree SHAP values, shape (n_classes, n_features), in log-odds units."""
        import xgboost as xgb

        dm = xgb.DMatrix(self._matrix([row]), feature_names=self.features, missing=np.nan)
        c = self.model.get_booster().predict(dm, pred_contribs=True)
        return np.asarray(c)[0][:, :-1]          # drop the bias column

    def explain(self, row: dict[str, float], cls: str, top: int = 5) -> list[dict[str, Any]]:
        """The ``top`` features pushing hardest towards (or away from) ``cls``."""
        contrib = self.contributions(row)[self.classes.index(cls)]
        order = np.argsort(-np.abs(contrib))[:top]
        return [{"feature": self.features[i], "value": row.get(self.features[i]),
                 "shap": float(contrib[i]), "towards": cls if contrib[i] > 0 else f"not {cls}"}
                for i in order]

    def predict(self, row: dict[str, float], cfg: Config = CONFIG) -> dict[str, Any]:
        # 1) raw XGBoost probabilities, 2) calibrated with the stored Platt curves,
        # 3) top class, confidence label and the SHAP reasons behind it.
        raw = self.model.predict_proba(self._matrix([row]))
        cal = apply_calibration(raw, self.meta.get("calibration"), self.classes)[0]
        k = int(np.argmax(cal))
        cls = self.classes[k]
        second = int(np.argsort(cal)[-2])
        return {
            "status": "ok",
            "predicted_class": cls,
            "probability": float(cal[k]),
            "confidence": confidence_label(float(cal[k]), cfg),
            "second_class": self.classes[second],
            "probabilities": {c: float(p) for c, p in zip(self.classes, cal)},
            "raw_probabilities": {c: float(p) for c, p in zip(self.classes, raw[0])},
            "top_contributions": self.explain(row, cls),
            "planet_threshold": self.meta.get("thresholds", {}).get("planet_at_eb_contamination"),
            "model_version": self.version,
        }


@lru_cache(maxsize=4)
def _cached_model(models_dir: str, mtime: float) -> TriageModel | None:
    return TriageModel.load(Path(models_dir))


def load_model(cfg: Config = CONFIG) -> TriageModel | None:
    """Load (and cache) the model; reloads automatically if the file changes."""
    mp = cfg.paths.models / MODEL_FILE
    if not mp.exists():
        return None
    return _cached_model(str(cfg.paths.models), mp.stat().st_mtime)
