"""Turn a report's routing reasons and SHAP contributions into plain sentences."""

from __future__ import annotations

import math
from typing import Any

from ui.text import (CLASS_INFO, DURATION_SENTENCE_OFF, DURATION_SENTENCE_OK, FEATURE_NAMES,
                     FEATURE_SENTENCES, GENERIC_SENTENCE, MODEL_LEANED, SIZE_SENTENCE)

UNSURE_PREFIX = "The classifier is unsure"


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
        return GENERIC_SENTENCE.format(name=name, cls=CLASS_INFO.get(cls, (cls,))[0].lower())
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
        if r.startswith(UNSURE_PREFIX):
            continue
        out.append(r)
    predicted = triage.get("predicted_class")
    target = routing.get("secondary_class") if final == "NEEDS_REVIEW" else final
    if predicted and target and predicted != target and final != "NEEDS_REVIEW":
        out.append(MODEL_LEANED.format(model=CLASS_INFO.get(predicted, (predicted,))[0].lower()))
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


def catalog_statement(report: dict[str, Any]) -> tuple[str, str] | None:
    """(kind, sentence) comparing our verdict with what the catalogs say.

    kind is "agree", "disagree", "unsure" or "none".
    """
    from ui.text import CATALOG_AGREE, CATALOG_DISAGREE, CATALOG_NONE, CATALOG_UNSURE

    routing = report.get("routing") or {}
    final = routing.get("primary_class")
    mods = report.get("modules") or {}
    pm = ((mods.get("planet") or {}).get("details") or {}).get("catalog_match")
    em = ((mods.get("eb") or {}).get("details") or {}).get("catalog")
    vm = ((mods.get("variable") or {}).get("details") or {}).get("gaia_class")
    claims = []
    if pm and pm.get("toi"):
        disp = pm.get("disposition")
        what = {"CP": "a confirmed planet", "KP": "a known planet",
                "PC": "a planet candidate", "APC": "a planet candidate",
                "FP": "a false positive (not a planet)", "FA": "a false alarm"}.get(disp, disp)
        claim = {"CP": "PLANET", "KP": "PLANET", "FP": "NOT_PLANET"}.get(disp)
        claims.append((f"NASA's TESS candidate list ({pm['toi']})", what, claim))
    if em:
        claims.append(("The TESS eclipsing binary catalog",
                       f"an eclipsing binary with a {em['period_d']:.4f}-day orbit", "EB"))
    if vm:
        claims.append(("Gaia", f"a variable star ({vm.replace('_', ' ').lower()})", "VARIABLE"))
    if not claims:
        return ("none", CATALOG_NONE)
    source, what, claim = claims[0]
    if final == "NEEDS_REVIEW" or claim is None:
        return ("unsure", CATALOG_UNSURE.format(source=source, what=what))
    agree = (claim == final) or (claim == "NOT_PLANET" and final != "PLANET")
    return ("agree" if agree else "disagree",
            (CATALOG_AGREE if agree else CATALOG_DISAGREE).format(source=source, what=what))
