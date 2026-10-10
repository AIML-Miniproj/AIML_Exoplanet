"""Route a classified signal to the module that characterises it.

Order of decisions:

1. Start from the classifier's probabilities (most likely class first).
2. Physics overrides, each logged as a reason:
   - the dip implies an object larger than ``cfg.max_planet_rjup`` Jupiter radii
     -> it cannot be a planet;
   - a clear second eclipse (SNR >= ``cfg.secondary_snr_eb``) -> an eclipsing
     binary, unless the brightness change over the orbit explains it (a planet's
     own reflected light disappearing behind the star gives a second dip about
     twice the reflection amplitude).
3. If the top probability is below ``cfg.uncertain_threshold`` the result is
   NEEDS_REVIEW and the modules of the two most likely classes both run. This
   also guarantees that a weak planet signal is never sent straight to NOISE.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from config import CONFIG, Config
from src import physics
from src.recycler.common import period_words

CLASSES = ("PLANET", "EB", "VARIABLE", "NOISE")
MODULE_FOR = {"PLANET": "planet", "EB": "eb", "VARIABLE": "variable", "NOISE": "noise"}
NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass
class RoutingDecision:
    primary_class: str                     # a class name or NEEDS_REVIEW
    confidence_label: str                  # High | Medium | Low
    secondary_class: str | None
    reasons: list[str] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)
    modules: list[str] = field(default_factory=list)
    top_probability: float = math.nan

    @property
    def needs_review(self) -> bool:
        return self.primary_class == NEEDS_REVIEW

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self), needs_review=self.needs_review)


def confidence_label(p: float, cfg: Config = CONFIG) -> str:
    if not math.isfinite(p):
        return "Low"
    if p >= cfg.conf_high:
        return "High"
    return "Medium" if p >= cfg.uncertain_threshold else "Low"


def _num(features: dict[str, Any], key: str) -> float:
    v = features.get(key)
    try:
        return float(v) if v is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


def route(probabilities: dict[str, float] | None, features: dict[str, Any],
          cfg: Config = CONFIG, signal: dict[str, Any] | None = None) -> RoutingDecision:
    """Decide which class (and which modules) a signal goes to.

    ``signal``: the main detected signal; when the result needs review and the dip
    search found a significant repeating dip, that is added as a hint.
    """
    if not probabilities:
        return RoutingDecision(NEEDS_REVIEW, "Low", None,
                               ["No trained classifier is available, so the signal "
                                "could not be classified."], {}, [])
    probs = {c: float(probabilities.get(c, 0.0)) for c in CLASSES}
    ranked = sorted(CLASSES, key=lambda c: probs[c], reverse=True)
    reasons: list[str] = []
    excluded: set[str] = set()
    forced_eb = False

    rp = _num(features, "implied_rp_rearth")
    rp_rjup = rp * physics.R_EARTH_M / physics.R_JUP_M if math.isfinite(rp) else math.nan
    if math.isfinite(rp_rjup) and rp_rjup > cfg.max_planet_rjup:
        excluded.add("PLANET")
        reasons.append(f"The dip would mean an object {rp_rjup:.1f} times the size of Jupiter, "
                       f"too big to be a planet (limit {cfg.max_planet_rjup:g}).")

    sec_snr = _num(features, "secondary_snr")
    if math.isfinite(sec_snr) and sec_snr >= cfg.secondary_snr_eb and ranked[0] != "EB":
        sec_ppm = _num(features, "secondary_depth_ppm")
        refl = abs(_num(features, "refl_amp_ppm"))
        explained = math.isfinite(refl) and math.isfinite(sec_ppm) and sec_ppm <= 2 * refl
        if explained:
            reasons.append("There is a faint second dip halfway between the main dips, but the "
                           "light reflected by the companion explains it.")
        else:
            reasons.insert(0, "A clear second eclipse was found halfway through the orbit, "
                              "which planets don't produce.")
            excluded.update({"PLANET", "NOISE"})
            forced_eb = True
            ranked = ["EB"] + [c for c in ranked if c != "EB"]

    ranked = [c for c in ranked if c not in excluded] + [c for c in ranked if c in excluded]
    top, second = ranked[0], ranked[1]
    p_top = probs[top]
    if excluded and top != max(probs, key=probs.get):
        reasons.append(f"Physics checks moved the verdict from "
                       f"{max(probs, key=probs.get).title()} to {top.title()}.")

    # A clear second eclipse is physical evidence on its own: no review needed.
    if p_top < cfg.uncertain_threshold and not forced_eb:
        why = (f"The classifier is unsure: {top.title()} {p_top:.0%}, "
               f"{second.title()} {probs[second]:.0%}.")
        if "PLANET" in (top, second) and "PLANET" not in excluded:
            why += " A possible planet is kept for review rather than discarded."
        reasons.append(why)
        if signal and signal.get("detected"):
            reasons.append(f"Worth a look: a repeating dip was found every "
                           f"{period_words(float(signal['period']))}.")
        modules = [MODULE_FOR[c] for c in (top, second)]
        return RoutingDecision(NEEDS_REVIEW, confidence_label(p_top, cfg), top, reasons, probs,
                               modules, p_top)
    return RoutingDecision(top, confidence_label(p_top, cfg), second, reasons, probs,
                           [MODULE_FOR[top]], p_top)
