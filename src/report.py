"""TargetReport: the single result object the pipeline hands to the UI, CLI and batch tools."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _clean(obj: Any) -> Any:
    """Make numpy / NaN values JSON-safe."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@dataclass
class TargetReport:
    """Everything known about one analysed target."""

    target: str
    tic_id: int | None = None
    preset: str = "standard"
    status: str = "ok"                       # ok | no_data | error
    message: str = ""
    star: dict[str, Any] = field(default_factory=dict)
    sectors: list[dict[str, Any]] = field(default_factory=list)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    signals: list[dict[str, Any]] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    physical: dict[str, Any] = field(default_factory=dict)
    triage: dict[str, Any] = field(default_factory=dict)
    plot_data: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def to_row(self) -> dict[str, Any]:
        """Flat one-line summary for batch CSVs."""
        sig = self.signals[0] if self.signals else {}
        row = {
            "target": self.target, "tic_id": self.tic_id, "status": self.status,
            "n_sectors": len(self.sectors), "period_d": sig.get("period"),
            "depth_ppm": sig.get("depth_ppm"), "sde": sig.get("sde"),
            "predicted_class": self.triage.get("predicted_class"),
            "confidence": self.triage.get("confidence"),
        }
        return _clean(row)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TargetReport:
        return cls(**d)
