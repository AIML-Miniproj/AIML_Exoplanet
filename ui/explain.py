"""Turn a report's routing reasons and SHAP contributions into plain sentences."""

from __future__ import annotations

import math
import re
from typing import Any

from ui.theme import label

UNSURE_PREFIX = "The classifier is unsure"
REVIEW_PREFIX = "Worth a look"

MODEL_LEANED = ("On the light-curve measurements alone the classifier leaned towards {model}; "
                "the physics check above overrides that.")
FEATURE_SENTENCES = {
    "odd_even_depth_ratio": ("Alternating dips have different depths.",
                             "Alternating dips have the same depth.", 0.9),
    "secondary_to_primary_ratio": ("There is no second, smaller dip halfway between, which "
                                   "eclipsing stars usually show.",
                                   "A second, smaller dip shows up halfway between the main "
                                   "dips.", 0.05),
    "secondary_best_to_primary_ratio": ("No second dip appears anywhere between the main dips.",
                                        "A second dip appears between the main dips.", 0.05),
    "ingress_to_total_ratio": ("The dips are U-shaped like a planet's, not V-shaped like a "
                               "star's.",
                               "The dips are V-shaped, more like two stars grazing each other.",
                               0.25),
    "sector_depth_cv": ("The dip depth stays the same across all observing sectors.",
                        "The dip depth changes from sector to sector.", 0.2),
    "red_noise_beta": ("The light curve is clean on the timescale of the dips.",
                       "The light curve has a lot of slow, irregular wobble.", 1.5),
    "ls_power_1": ("There is no strong, smooth brightness cycle.",
                   "The star's brightness rises and falls in a smooth, regular cycle.", 0.3),
    "acf_peak_height": ("The brightness pattern does not repeat cleanly.",
                        "The brightness pattern repeats very regularly.", 0.3),
    "ls_period_matches_bls": ("The dips are unrelated to any smooth brightness cycle.",
                              "The smooth brightness cycle has the same rhythm as the dips, "
                              "typical of close pairs of stars.", 0.5),
    "depth_snr_per_transit": ("Each dip is faint, close to the noise.",
                              "Each dip is clearly visible on its own.", 5.0),
    "single_event_excess": ("All the dips look alike.",
                            "One dip is much stronger than the others.", 2.0),
    "amp_p2p_ppm": ("The star's own brightness changes are small.",
                    "The star's brightness changes a lot on its own.", 10_000),
    "skew": ("The star spends most of its time at normal brightness, with short drops.",
             "The star spends most of its time faint, with short brightenings.", 0.0),
    "trapezoid_chi2_vs_box_chi2": ("A dip with sloped edges fits much better than a sharp box.",
                                   "A sharp box fits about as well as a sloped dip.", 0.9),
}
DURATION_SENTENCE_OK = ("The dips last about as long as expected for something orbiting this "
                        "star.")
DURATION_SENTENCE_OFF = ("The dips last much {word} than expected for something orbiting "
                         "this star.")
SIZE_SENTENCE = "The dip implies an object about {size}."
GENERIC_SENTENCE = "The measurement \"{name}\" points towards {cls}."
FEATURE_NAMES = {
    "teff": "star temperature", "rad": "star size", "logg": "star surface gravity",
    "tmag": "star brightness", "contratio": "light from nearby stars",
    "period_d": "time between dips", "duration_hr": "dip duration", "depth_ppm": "dip depth",
    "kurtosis": "how peaked the brightness distribution is",
    "fourier_R21": "shape of the brightness cycle", "fourier_R31": "shape of the brightness cycle",
    "refl_amp_ppm": "brightening once per orbit", "ellip_amp_ppm": "brightening twice per orbit",
    "crowdsap_min": "share of light from the target star",
    "var_period_d": "period of the star's own variation",
    "ls_period_1": "main rhythm in the brightness",
}

# Readable names for the model-evaluation charts (fallback: the raw name).
READABLE = {
    "acf_peak_height": "How regularly the light repeats",
    "skew": "Lopsidedness of the brightness",
    "ingress_to_total_ratio": "Dip shape (V vs U)",
    "duration_ratio_expected": "Dip length vs expected",
    "red_noise_beta": "Slow irregular wobble",
    "sector_depth_cv": "Depth change between sectors",
    "odd_even_depth_ratio": "Odd vs even dip depth",
    "teff": "Star temperature",
    "ls_period_matches_bls": "Brightness cycle matches dips",
    "refl_amp_ppm": "Brightening once per orbit",
    "ellip_amp_ppm": "Brightening twice per orbit",
    "kurtosis": "Peakedness of the brightness",
    "fourier_R21": "Cycle shape (R21)", "fourier_R31": "Cycle shape (R31)",
    "fourier_phi21": "Cycle shape (phi21)", "fourier_phi31": "Cycle shape (phi31)",
    "secondary_best_depth_ppm": "Depth of a second dip",
    "secondary_to_primary_ratio": "Second dip vs main dip",
    "secondary_best_to_primary_ratio": "Strongest second dip vs main dip",
    "secondary_snr_per_transit": "Second dip strength",
    "depth_snr_per_transit": "Dip strength per transit",
    "ls_power_1": "Strength of the main brightness cycle",
    "ls_period_1": "Main brightness cycle period",
    "amp_p2p_ppm": "Size of the star's own changes",
    "rms_ppm": "Scatter of the brightness",
    "implied_rp_rearth": "Implied object size",
    "trapezoid_chi2_vs_box_chi2": "Sloped vs box-shaped dip",
    "single_event_excess": "One dip much stronger",
    "period_d": "Time between dips", "depth_ppm": "Dip depth", "duration_hr": "Dip duration",
    "rad": "Star size", "logg": "Star surface gravity", "tmag": "Star brightness",
    "contratio": "Light from nearby stars", "crowdsap_min": "Share of light from the target",
    "var_period_d": "Period of the star's own changes",
    "var_period_halves_diff": "Odd vs even cycle difference",
    "acf_period": "Repeat time of the light",
    "ls_period_sector_std": "Period change between sectors",
    "harmonic_dll_per_transit": "Dip vs smooth wave fit",
    "transit_asymmetry": "Dip asymmetry",
}


def readable(feature: str) -> str:
    return READABLE.get(feature) or feature.replace("_", " ")


def _num(v: Any) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return math.nan
    return v


def feature_sentence(feature: str, value: Any, cls: str) -> str | None:
    """One plain-English sentence for a feature value, or None if there is nothing to say."""
    v = _num(value)
    if not math.isfinite(v):
        return None
    if feature in FEATURE_SENTENCES:
        low, high, thr = FEATURE_SENTENCES[feature]
        return high if v >= thr else low
    if feature == "duration_ratio_expected":
        if 0.5 <= v <= 2.0:
            return DURATION_SENTENCE_OK
        return DURATION_SENTENCE_OFF.format(word="longer" if v > 2 else "shorter")
    if feature == "implied_rp_rearth":
        size = (f"{v * 0.0892:.1f} times the size of Jupiter" if v >= 6
                else f"{v:.1f} times the size of Earth")
        return SIZE_SENTENCE.format(size=size)
    name = FEATURE_NAMES.get(feature)
    if name:
        return GENERIC_SENTENCE.format(name=name, cls=label(cls).lower())
    return None


def _class_words(sentence: str) -> str:
    """'... from Planet to Eb.' -> '... from planet to eclipsing binary.'"""
    m = re.search(r"from (\w+) to (\w+)\.$", sentence)
    if not m:
        return sentence
    a, b = (label(w.upper()).lower() for w in m.groups())
    return sentence[:m.start()] + f"from {a} to {b}."


def review_hint(report: dict[str, Any]) -> str | None:
    for r in (report.get("routing") or {}).get("reasons") or []:
        if r.startswith(REVIEW_PREFIX):
            return r
    return None


def reasons(report: dict[str, Any], n: int = 5) -> list[str]:
    """Router reasons first (physics overrides lead), then the classifier's top
    supporting features in plain words. Only reasons that support the final verdict
    are given; if physics overrode the classifier, that is said explicitly."""
    routing = report.get("routing") or {}
    triage = report.get("triage") or {}
    final = routing.get("primary_class")
    out: list[str] = []
    for r in routing.get("reasons") or []:
        if r.startswith((UNSURE_PREFIX, REVIEW_PREFIX)):   # the hint is shown on its own
            continue
        out.append(_class_words(r))
    predicted = triage.get("predicted_class")
    target = routing.get("secondary_class") if final == "NEEDS_REVIEW" else final
    if predicted and target and predicted != target and final != "NEEDS_REVIEW":
        out.append(MODEL_LEANED.format(model=label(predicted).lower()))
        return out[:n]
    seen = set(out)
    for c in triage.get("top_contributions") or []:
        if c.get("shap", 0) <= 0:          # only features pushing towards the verdict
            continue
        s = feature_sentence(c.get("feature", ""), c.get("value"), predicted or "")
        if s and s not in seen:
            out.append(s)
            seen.add(s)
        if len(out) >= n:
            break
    return out[:n]


def catalog_statement(report: dict[str, Any]) -> tuple[str, str]:
    """(kind, what the catalog says). kind is "agree", "disagree", "unsure" or "none"."""
    routing = report.get("routing") or {}
    final = routing.get("primary_class")
    mods = report.get("modules") or {}
    pm = ((mods.get("planet") or {}).get("details") or {}).get("catalog_match")
    em = ((mods.get("eb") or {}).get("details") or {}).get("catalog")
    vm = ((mods.get("variable") or {}).get("details") or {}).get("gaia_class")
    claims = []
    if pm and pm.get("toi"):
        disp = pm.get("disposition")
        what = {"CP": "confirmed planet", "KP": "known planet", "PC": "planet candidate",
                "APC": "planet candidate", "FP": "false positive (not a planet)",
                "FA": "false alarm"}.get(disp, disp)
        claim = {"CP": "PLANET", "KP": "PLANET", "FP": "NOT_PLANET"}.get(disp)
        claims.append((f"{pm['toi']} is a {what} (TESS candidate list)", claim))
    if em:
        claims.append((f"eclipsing binary, {em['period_d']:.4f}-day orbit (TESS EB catalog)",
                       "EB"))
    if vm:
        claims.append((f"variable star, {vm.replace('_', ' ').lower()} (Gaia)", "VARIABLE"))
    if not claims:
        return "none", "not in the catalogs we checked"
    what, claim = claims[0]
    if final == "NEEDS_REVIEW" or claim is None:
        return "unsure", what
    agree = (claim == final) or (claim == "NOT_PLANET" and final != "PLANET")
    return ("agree" if agree else "disagree"), what
