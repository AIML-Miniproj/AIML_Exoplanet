"""Features about the star itself, taken from the TESS Input Catalog."""

from __future__ import annotations

import math


def stellar_features(star) -> dict[str, float]:
    def get(name: str) -> float:
        v = getattr(star, name, math.nan) if star is not None else math.nan
        return float(v) if v is not None else math.nan

    return {
        "teff": get("teff"),
        "rad": get("rad"),
        "logg": get("logg"),
        "tmag": get("tmag"),
        "contratio": get("contratio"),
        "ruwe": get("ruwe"),
    }
