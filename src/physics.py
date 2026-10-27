"""Small physics helpers. Inputs are plain floats in the units named in each argument."""

from __future__ import annotations

import numpy as np
from astropy import constants as C

R_SUN_M = C.R_sun.value
R_JUP_M = C.R_jup.value
R_EARTH_M = C.R_earth.value
M_SUN_KG = C.M_sun.value
AU_M = C.au.value
G_SI = C.G.value
RHO_SUN = M_SUN_KG / (4 / 3 * np.pi * R_SUN_M**3)   # kg / m^3
DAY_S = 86400.0


def radius_ratio(depth: float) -> float:
    """Planet-to-star radius ratio from a fractional transit depth."""
    # depth = (Rp / R*)^2 for a dark disc crossing a uniformly bright star
    return float(np.sqrt(max(depth, 0.0)))


def planet_radius_rearth(depth: float, star_radius_rsun: float) -> float:
    """Planet radius in Earth radii, assuming the dip is caused by a dark disc."""
    return radius_ratio(depth) * star_radius_rsun * R_SUN_M / R_EARTH_M


def semi_major_axis_au(period_d: float, star_mass_msun: float) -> float:
    """Orbital distance from Kepler's third law (planet mass neglected)."""
    p_s = period_d * DAY_S
    # Kepler's third law: a^3 = G M P^2 / (4 pi^2)
    a_m = (G_SI * star_mass_msun * M_SUN_KG * p_s**2 / (4 * np.pi**2)) ** (1 / 3)
    return a_m / AU_M


def equilibrium_temp_k(star_teff_k: float, star_radius_rsun: float, a_au: float,
                       albedo: float = 0.3) -> float:
    """Planet equilibrium temperature with full heat redistribution."""
    # T_eq = T* (1 - A)^(1/4) sqrt(R* / 2a): absorbed starlight = emitted heat
    return star_teff_k * (1 - albedo) ** 0.25 * np.sqrt(
        star_radius_rsun * R_SUN_M / (2 * a_au * AU_M))


def stellar_density_solar(mass_msun: float, radius_rsun: float) -> float:
    """Mean stellar density in solar units."""
    return mass_msun / radius_rsun**3


def expected_duration_d(period_d: float, rho_solar: float) -> float:
    """Expected duration of a central transit for a star of given mean density.

    Uses a/R* from the density and the period, then T ~ P / pi * asin(R*/a).
    """
    if not np.isfinite(rho_solar) or rho_solar <= 0:
        return float("nan")
    p_s = period_d * DAY_S
    # a / R* from the star's mean density (Kepler's third law divided by R*^3)
    a_over_r = (G_SI * rho_solar * RHO_SUN * p_s**2 / (3 * np.pi)) ** (1 / 3)
    if a_over_r <= 1:
        return float("nan")
    return float(period_d / np.pi * np.arcsin(1 / a_over_r))
