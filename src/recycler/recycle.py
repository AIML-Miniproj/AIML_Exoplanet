"""Run the router and the recycler modules on a finished analysis.

Fills ``report.routing``, ``report.modules``, ``report.plain_summary`` and
``report.key_facts``, and appends NOISE / NEEDS_REVIEW results to the log.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config import CONFIG, Config
from src.recycler import eb_module, planet_module, systematics_log, variable_module
from src.recycler.router import route

log = logging.getLogger(__name__)

CLASS_WORDS = {"PLANET": "a planet", "EB": "a pair of stars eclipsing each other",
               "VARIABLE": "a star that changes brightness on its own",
               "NOISE": "just noise"}


def run_module(name: str, report, pre, info, cfg: Config):
    sig = report.signals[0]
    if name == "planet":
        return planet_module.characterize(sig, info, report.tic_id, cfg)
    if name == "eb":
        t = f = None
        if pre is not None:
            t = np.asarray(pre.lc_flat.time.value, float)
            f = np.asarray(pre.lc_flat.flux.value, float)
        return eb_module.characterize(sig, report.features, t, f, report.tic_id, cfg)
    if name == "variable":
        return variable_module.characterize(report.features, info, cfg)
    return systematics_log.characterize(sig, report.features,
                                        report.routing.get("reasons", []), cfg)


def recycle(report, pre=None, info=None, cfg: Config = CONFIG) -> None:
    """Route the signal, run the module(s) and write the plain-language results."""
    probs = report.triage.get("probabilities") if report.triage.get("status") == "ok" else None
    decision = route(probs, report.features, cfg)
    report.routing = decision.to_dict()
    results: dict[str, Any] = {}
    for name in decision.modules:
        try:
            results[name] = run_module(name, report, pre, info, cfg)
        except Exception as e:  # noqa: BLE001 - one module failing must not lose the report
            log.warning("%s module failed: %s", name, e)
    report.modules = {k: v.to_dict() for k, v in results.items()}

    first = results.get(decision.modules[0]) if decision.modules else None
    if decision.needs_review and decision.modules:
        a = CLASS_WORDS.get(decision.secondary_class or "", "something")
        others = [c for c, m in (("PLANET", "planet"), ("EB", "eb"), ("VARIABLE", "variable"),
                                 ("NOISE", "noise")) if m in decision.modules[1:]]
        b = CLASS_WORDS.get(others[0], "something else") if others else "something else"
        report.plain_summary = (f"We're not sure: this could be {a} or {b}, so it is kept for "
                                f"a closer look rather than thrown away.")
    elif decision.needs_review:
        report.plain_summary = ("We could not classify this signal because no trained model is "
                                "available.")
    else:
        report.plain_summary = first.plain_summary if first else ""
    report.key_facts = [k.as_tuple() for k in first.key_facts] if first else []

    if cfg.recycler_log and report.tic_id:
        systematics_log.append(report.to_dict(), cfg)
