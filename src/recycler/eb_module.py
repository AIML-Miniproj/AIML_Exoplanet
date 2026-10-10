"""Eclipsing-binary characterisation (Phase 5). For now: the true orbital period.

The dip search often locks onto half the orbital period of an eclipsing binary:
its two eclipses per orbit look like one repeating dip. Which period is right is
decided from the light curve alone:

1. Alternating dips have clearly different depths -> the search folded the
   primary and secondary eclipses together: true period = 2x detected.
2. Otherwise, a clear second eclipse halfway between the dips -> the detected
   period already contains both eclipses: true period = detected.
3. Otherwise the two cases look identical in the data (two equal eclipses, or a
   secondary too faint to see). Two eclipses per orbit is the normal case for a
   binary, so 2x detected is reported, flagged as ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIGMA_MIN = 3.0          # "clearly different" / "clearly present": 3 standard deviations


@dataclass
class TruePeriod:
    period_d: float
    factor: int           # true period / detected period
    rule: str             # "odd_even_differ" | "secondary_seen" | "ambiguous_twin"
    ambiguous: bool


def true_period(detected_period: float, features: dict[str, float]) -> TruePeriod:
    """Orbital period of an eclipsing binary from the detected dip period and features."""
    oe = features.get("odd_even_depth_sigma", np.nan)
    sec = features.get("secondary_snr", np.nan)
    if np.isfinite(oe) and oe >= SIGMA_MIN:
        return TruePeriod(2 * detected_period, 2, "odd_even_differ", False)
    if np.isfinite(sec) and sec >= SIGMA_MIN:
        return TruePeriod(detected_period, 1, "secondary_seen", False)
    return TruePeriod(2 * detected_period, 2, "ambiguous_twin", True)
