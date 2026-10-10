"""Variable-star characterisation (uses the light curve with variability kept).

Coarse type from simple rules on the period, the size of the brightness change
and how regular the pattern is. No model is trained: about 25 example stars are
too few to learn five or more types.

Rules, in order (amplitude = spread between the 5th and 95th percentile of the
brightness; "regular" = the strongest periodogram peak explains at least
REGULAR_POWER of the variance):

1. No significant repeating pattern (false-alarm chance >= 1%)    -> OTHER
2. Amplitude >= 10% and regular:
   period 0.2-1 day -> RR_LYRAE; period 1-50 days -> CEPHEID;
   period < 0.2 day -> DELTA_SCUTI/GAMMA_DOR (high-amplitude delta Scuti)
3. Amplitude < 10%:
   period < 0.3 day                                              -> DELTA_SCUTI/GAMMA_DOR
   period 0.3-3 days, star hotter than 6300 K and regular        -> DELTA_SCUTI/GAMMA_DOR
   period >= 0.3 day otherwise                                   -> ROTATION/SPOTTED
4. Anything else                                                 -> OTHER
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, Config
from src.recycler.common import KeyFact, ModuleResult, fmt, period_words

BIG_AMPLITUDE = 0.10          # 10% peak-to-peak (5th-95th percentile)
REGULAR_POWER = 0.3
HOT_STAR_K = 6300.0
CLASSES = ("ROTATION/SPOTTED", "DELTA_SCUTI/GAMMA_DOR", "RR_LYRAE", "CEPHEID", "OTHER")

PLAIN = {
    "ROTATION/SPOTTED": "its brightness changes every {p} as dark spots on its surface turn "
                        "in and out of view",
    "DELTA_SCUTI/GAMMA_DOR": "it gently pulsates, brightening and fading every {p}",
    "RR_LYRAE": "it swells and shrinks, brightening and fading strongly every {p}, like an "
                "RR Lyrae star",
    "CEPHEID": "it is a pulsating giant star (a Cepheid) that brightens and fades every {p}",
    "OTHER": "it changes brightness on its own, but not in a pattern we can name",
}
DISPLAY = {"ROTATION/SPOTTED": "Rotation / starspots",
           "DELTA_SCUTI/GAMMA_DOR": "Delta Scuti / Gamma Doradus pulsator",
           "RR_LYRAE": "RR Lyrae", "CEPHEID": "Cepheid", "OTHER": "Other"}
TOOLTIP = {
    "ROTATION/SPOTTED": "Starspots, like sunspots, make one side darker; as the star spins its "
                        "brightness rises and falls.",
    "DELTA_SCUTI/GAMMA_DOR": "Hot stars a bit heavier than the Sun that pulsate in small, "
                             "regular ways.",
    "RR_LYRAE": "Old pulsating stars that swell and shrink about once or twice a day with a big "
                "change in brightness.",
    "CEPHEID": "Large pulsating stars with a period of days to weeks.",
    "OTHER": "A real brightness change that does not fit the simple types used here.",
}


def _f(d: dict[str, Any], key: str) -> float:
    try:
        v = float(d.get(key))
    except (TypeError, ValueError):
        return math.nan
    return v


def classify(features: dict[str, Any]) -> tuple[str, str]:
    """Coarse variable type and the rule that decided it."""
    p = _f(features, "var_period_d")
    if not math.isfinite(p):
        p = _f(features, "ls_period_1")
    fap = _f(features, "ls_log10_fap")
    amp = _f(features, "amp_p2p_ppm") / 1e6
    power = _f(features, "ls_power_1")
    teff = _f(features, "teff")
    regular = math.isfinite(power) and power >= REGULAR_POWER
    if not math.isfinite(p) or not (math.isfinite(fap) and fap < -2):
        return "OTHER", "no significant repeating pattern"
    if math.isfinite(amp) and amp >= BIG_AMPLITUDE and regular:
        if 0.2 <= p <= 1.0:
            return "RR_LYRAE", "large, regular change with a period of 0.2-1 day"
        if 1.0 < p <= 50:
            return "CEPHEID", "large, regular change with a period of 1-50 days"
        if p < 0.2:
            return "DELTA_SCUTI/GAMMA_DOR", "large, regular change faster than 0.2 day"
    if math.isfinite(amp) and amp < BIG_AMPLITUDE:
        if p < 0.3:
            return "DELTA_SCUTI/GAMMA_DOR", "small change faster than 0.3 day"
        if p <= 3 and regular and math.isfinite(teff) and teff > HOT_STAR_K:
            return "DELTA_SCUTI/GAMMA_DOR", "small, regular change on a hot star, 0.3-3 days"
        return "ROTATION/SPOTTED", "small change with a period of 0.3 day or more"
    return "OTHER", "does not match the simple rules"


@lru_cache(maxsize=1)
def _gaia_classes(path: str) -> dict[str, str]:
    from src.triage.labels import GAIA_COARSE

    g = pd.read_parquet(path)
    return dict(zip(g["source_id"].astype(np.int64).astype(str),
                    g["best_class_name"].map(GAIA_COARSE).fillna("OTHER")))


def catalog_class(gaia_id: str | None, cfg: Config = CONFIG) -> str | None:
    """Gaia's coarse class for this star from the cached Gaia variable list (G < 11)."""
    path = cfg.paths.labels / "raw" / "gaia_vari_simple_g11.parquet"
    if not gaia_id or not path.exists():
        return None
    return _gaia_classes(str(path)).get(str(gaia_id))


def characterize(features: dict[str, Any], star: Any = None, cfg: Config = CONFIG
                 ) -> ModuleResult:
    """Variable module: type, period, amplitude and stability, in plain words."""
    kind, rule = classify(features)
    p = _f(features, "var_period_d")
    if not math.isfinite(p):
        p = _f(features, "ls_period_1")
    amp = _f(features, "amp_p2p_ppm") / 1e4                        # percent
    stab = _f(features, "ls_period_sector_std")
    r21 = _f(features, "fourier_R21")
    gaia_id = star.get("gaia_id") if isinstance(star, dict) else getattr(star, "gaia_id", None)
    gaia = catalog_class(gaia_id, cfg)
    summary = ("Not a planet, but still interesting: this star changes brightness on its own; "
               + PLAIN[kind].format(p=period_words(p)) + ".")
    facts = [
        KeyFact("Type", DISPLAY[kind], "", TOOLTIP[kind]),
        KeyFact("Period", fmt(p, 4), "days", "Time for one full cycle of brightening and fading."),
        KeyFact("Brightness change", fmt(amp, 2), "%",
                "How much the brightness varies (5th to 95th percentile)."),
        KeyFact("Pattern regularity", fmt(_f(features, "ls_power_1"), 2), "(0-1)",
                "How much of the brightness change one regular cycle explains. Near 1: very "
                "regular (pulsation); low: irregular (spots that come and go)."),
        KeyFact("Period stability", f"{fmt(stab * 100, 2)}" if math.isfinite(stab) else
                "unknown", "% change between sectors",
                "How much the period differs from one observing sector to the next."),
        KeyFact("Light-curve shape (R21)", fmt(r21, 2) if math.isfinite(r21) and r21 <= 1
                else "unclear", "",
                "Strength of the second harmonic relative to the first: 0 for a smooth wave, "
                "larger for sharp, lopsided cycles such as pulsating stars. Values above 1 "
                "usually mean the period found is a multiple of the true one, so they are not "
                "shown."),
        KeyFact("Catalog check", f"Gaia classifies it as: {DISPLAY.get(gaia, gaia)}"
                if gaia else "not in the cached Gaia variable list", "",
                "Cached Gaia variability classification (stars brighter than Gaia G = 11)."),
    ]
    details = {"class": kind, "rule": rule, "gaia_class": gaia,
               "agrees_with_gaia": None if gaia is None else gaia == kind}
    return ModuleResult("variable", summary, facts, details)
