"""Planet characterisation: size, orbit, temperature, with uncertainties.

Uncertainties come from a Monte Carlo over the measured dip depth and the
star's catalog radius, mass and temperature errors (``cfg.mc_draws`` draws,
vectorised). The known-planet check uses the cached TOI table only.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, Config
from src import physics
from src.recycler.common import KeyFact, ModuleResult, fmt, fmt_pm, period_words

log = logging.getLogger(__name__)

ALBEDO = 0.3
# Upper radius edge of each size category, in Earth radii.
SIZE_CATEGORIES = ((1.25, "Earth-size"), (2.0, "super-Earth"), (6.0, "Neptune-size"),
                   (25.0, "Jupiter-size"), (math.inf, "larger than any known planet"))
TOI_STATUS = {"CP": "a confirmed planet", "KP": "a known planet", "PC": "a planet candidate",
              "APC": "an ambiguous planet candidate", "FP": "a false positive",
              "FA": "a false alarm"}


def _get(obj: Any, key: str) -> float:
    v = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    try:
        return float(v) if v is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


def physical_parameters(signal: Any, star: Any) -> dict[str, float]:
    """Planet-like interpretation of the dip using the star's catalog values."""
    nan = math.nan
    period, depth = _get(signal, "period"), _get(signal, "depth")
    rad, mass, teff = _get(star, "rad"), _get(star, "mass"), _get(star, "teff")
    out = {"period_d": period, "depth_ppm": depth * 1e6,
           "rp_over_rs": physics.radius_ratio(depth) if math.isfinite(depth) else nan,
           "rp_rearth": nan, "rp_rjup": nan, "a_au": nan, "t_eq_k": nan}
    if math.isfinite(rad) and math.isfinite(depth):
        out["rp_rearth"] = physics.planet_radius_rearth(depth, rad)
        out["rp_rjup"] = out["rp_rearth"] * physics.R_EARTH_M / physics.R_JUP_M
    if math.isfinite(mass) and math.isfinite(period):
        out["a_au"] = physics.semi_major_axis_au(period, mass)
        if math.isfinite(teff) and math.isfinite(rad):
            out["t_eq_k"] = physics.equilibrium_temp_k(teff, rad, out["a_au"], ALBEDO)
    return out


def monte_carlo(signal: Any, star: Any, n: int = CONFIG.mc_draws,
                seed: int = CONFIG.random_state) -> tuple[dict[str, tuple[float, float, float]],
                                                          list[str]]:
    """16th / 50th / 84th percentiles of radius, distance and temperature.

    Missing catalog errors are treated as zero (listed in the second return value),
    so the spread then reflects only the quantities that do have errors.
    """
    rng = np.random.default_rng(seed)
    missing = []

    def draw(value: float, err: float, name: str, floor: float) -> np.ndarray:
        if not math.isfinite(value):
            return np.full(n, np.nan)
        if not math.isfinite(err) or err <= 0:
            missing.append(name)
            return np.full(n, value)
        return np.maximum(rng.normal(value, err, n), floor)

    depth = draw(_get(signal, "depth"), _get(signal, "depth_err"), "dip depth", 0.0)
    rad = draw(_get(star, "rad"), _get(star, "rad_err"), "star radius", 0.05)
    mass = draw(_get(star, "mass"), _get(star, "mass_err"), "star mass", 0.05)
    teff = draw(_get(star, "teff"), _get(star, "teff_err"), "star temperature", 500.0)
    period = _get(signal, "period")

    # The physics.py formulas applied to all draws at once (vectorised): radius from
    # the depth, orbit from Kepler's third law, temperature from the star. The 16th /
    # 50th / 84th percentiles give the value and its error bars.
    rp_re = np.sqrt(depth) * rad * physics.R_SUN_M / physics.R_EARTH_M
    a_au = (physics.G_SI * mass * physics.M_SUN_KG * (period * physics.DAY_S) ** 2
            / (4 * np.pi ** 2)) ** (1 / 3) / physics.AU_M
    t_eq = teff * (1 - ALBEDO) ** 0.25 * np.sqrt(rad * physics.R_SUN_M / (2 * a_au * physics.AU_M))

    def pct(x: np.ndarray) -> tuple[float, float, float]:
        x = x[np.isfinite(x)]
        return tuple(float(v) for v in np.percentile(x, [16, 50, 84])) if x.size else \
            (math.nan,) * 3

    return {"rp_rearth": pct(rp_re), "rp_rjup": pct(rp_re * physics.R_EARTH_M / physics.R_JUP_M),
            "a_au": pct(a_au), "t_eq_k": pct(t_eq)}, missing


def size_category(rp_rearth: float) -> str:
    if not math.isfinite(rp_rearth):
        return "unknown"
    return next(name for edge, name in SIZE_CATEGORIES if rp_rearth < edge)


@lru_cache(maxsize=1)
def _toi_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def known_planet(tic_id: int | None, period: float, cfg: Config = CONFIG,
                 tol: float = 0.01) -> dict[str, Any] | None:
    """The cached TOI entry for this star whose period matches (1x, 1/2x or 2x)."""
    path = cfg.paths.labels / "raw" / "toi.parquet"
    if not tic_id or not path.exists() or not math.isfinite(period):
        return None
    toi = _toi_table(str(path))
    rows = toi[toi["tid"] == int(tic_id)]
    for _, r in rows.iterrows():
        for ratio in (1.0, 0.5, 2.0):
            if r["pl_orbper"] and abs(period / (r["pl_orbper"] * ratio) - 1) < tol:
                return {"toi": f"TOI {r['toi']}", "disposition": r["tfopwg_disp"],
                        "status": TOI_STATUS.get(r["tfopwg_disp"], r["tfopwg_disp"]),
                        "catalog_period_d": float(r["pl_orbper"]), "ratio": ratio}
    if len(rows):
        return {"toi": None, "disposition": None, "status": "listed, with a different period",
                "catalog_period_d": float(rows["pl_orbper"].iloc[0]), "ratio": math.nan}
    return None


def characterize(signal: dict[str, Any], star: Any, tic_id: int | None = None,
                 cfg: Config = CONFIG) -> ModuleResult:
    """Planet module: physical size, orbit and temperature, in plain words."""
    phys = physical_parameters(signal, star)
    mc, missing = monte_carlo(signal, star, cfg.mc_draws, cfg.random_state)
    rp_re = mc["rp_rearth"][1] if math.isfinite(mc["rp_rearth"][1]) else phys["rp_rearth"]
    category = size_category(rp_re)
    period = phys["period_d"]
    match = known_planet(tic_id, period, cfg)

    if math.isfinite(rp_re):
        big = rp_re >= 6
        size = mc["rp_rjup"] if big else mc["rp_rearth"]
        unit_words = "Jupiter" if big else "Earth"
        summary = (f"This looks like a planet about {size[1]:.1f} times the size of "
                   f"{unit_words} that circles its star every {period_words(period)}.")
        if category == "larger than any known planet":
            summary = (f"The dip would mean an object {size[1]:.1f} times the size of Jupiter, "
                       f"bigger than any known planet, so it is probably not a planet.")
    else:
        summary = (f"This looks like a planet that circles its star every "
                   f"{period_words(period)}; its size is unknown because the star's size is "
                   f"not in the catalog.")

    big = math.isfinite(rp_re) and rp_re >= 6
    rkey, runit = ("rp_rjup", "Jupiter radii") if big else ("rp_rearth", "Earth radii")
    facts = [
        KeyFact("Size", fmt_pm(*_mid(mc[rkey])), runit,
                "Estimated from how much starlight is blocked and the star's size."),
        KeyFact("Size category", category, "",
                "Earth-size < 1.25, super-Earth < 2, Neptune-size < 6, Jupiter-size < 25 "
                "Earth radii."),
        KeyFact("Orbit period", fmt(period, 3), "days", "Time for one trip around the star."),
        KeyFact("Distance from star", fmt_pm(*_mid(mc["a_au"]), digits=3), "AU",
                "1 AU is the distance from the Earth to the Sun."),
        KeyFact("Temperature", fmt_pm(*_mid(mc["t_eq_k"]), digits=0), "K",
                "Rough temperature if the planet reflects 30% of the starlight and spreads "
                "the heat evenly. Degrees Celsius = kelvin minus 273."),
        KeyFact("Dip depth", fmt(phys["depth_ppm"] / 1e4, 3), "%",
                "How much of the star's light is blocked during each dip."),
    ]
    if match and match["toi"]:
        facts.append(KeyFact("Catalog check", f"{match['toi']}: {match['status']}", "",
                             "NASA's list of TESS candidates (cached copy)."))
    else:
        facts.append(KeyFact("Catalog check", "not in the TESS candidate list" if not match else
                             "listed, but with a different period", "",
                             "NASA's list of TESS candidates (cached copy)."))
    details = {"physical": phys, "monte_carlo": {k: list(v) for k, v in mc.items()},
               "mc_draws": cfg.mc_draws, "errors_missing": missing, "size_category": category,
               "catalog_match": match}
    return ModuleResult("planet", summary, facts, details)


def _mid(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """(16, 50, 84) percentiles -> (median, low, high) for ``fmt_pm``."""
    return p[1], p[0], p[2]
