# src/analyzer.py

import numpy as np
import astropy.constants as C
from config import (PHASE_WIDTH, STAR_RADIUS_SOLAR, PLANET_RADIUS_JUP,
                    STAR_MASS_SOLAR, STAR_TEFF, FEATURE_NAMES)

def analyze_transit(lc_flat, bls_output):
    """
    Phase-folds, measures transit depth, Rp/Rs,
    physical parameters, and extracts ML features.
    """
    best_period = bls_output["best_period"]
    t0          = bls_output["t0"]
    duration    = bls_output["duration"]
    t           = lc_flat.time.value

    # Phase fold
    folded = lc_flat.fold(period=best_period, epoch_time=t0)
    phase  = folded.phase.value
    flux   = folded.flux.value

    # In/out masks
    in_transit  = np.abs(phase) < PHASE_WIDTH
    out_transit = ~in_transit

    f_in  = np.median(flux[in_transit])
    f_out = np.median(flux[out_transit])

    # Physical parameters
    depth      = (f_out - f_in) / f_out
    rp_over_rs = np.sqrt(max(depth, 0))

    # Literature comparison
    star_r             = STAR_RADIUS_SOLAR * C.R_sun
    planet_r           = PLANET_RADIUS_JUP * C.R_jup
    rp_over_rs_lit     = (planet_r / star_r).decompose().value
    error_pct          = abs(rp_over_rs - rp_over_rs_lit) / rp_over_rs_lit * 100

    # Semi-major axis via Kepler's 3rd Law
    G      = C.G.value
    M_star = STAR_MASS_SOLAR * C.M_sun.value
    P_sec  = best_period * 86400
    a_m    = (G * M_star * P_sec**2 / (4 * np.pi**2)) ** (1/3)
    a_AU   = a_m / C.au.value

    # Equilibrium temperature
    albedo = 0.3
    T_eq   = STAR_TEFF * (1 - albedo)**0.25 * np.sqrt(
        STAR_RADIUS_SOLAR * C.R_sun.value / (2 * a_m)
    )

    # ML features
    std_in  = np.std(flux[in_transit])
    std_out = np.std(flux[out_transit])
    snr     = depth / std_out if std_out > 0 else 0

    left    = flux[(phase > -PHASE_WIDTH) & (phase < 0)]
    right   = flux[(phase > 0) & (phase < PHASE_WIDTH)]
    min_len = min(len(left), len(right))
    symmetry = (
        1 - np.mean(np.abs(left[:min_len] - right[:min_len][::-1]))
        if min_len > 0 else 0
    )

    n_transits = int((t[-1] - t[0]) / best_period)

    features = {
        "depth"        : depth,
        "duration"     : duration,
        "mean_flux_in" : f_in,
        "std_flux_in"  : std_in,
        "std_flux_out" : std_out,
        "snr"          : snr,
        "symmetry"     : symmetry,
        "sharpness"    : depth / duration if duration > 0 else 0,
        "n_transits"   : n_transits
    }

    physical = {
        "period"              : best_period,
        "depth"               : depth,
        "rp_over_rs"          : rp_over_rs,
        "rp_over_rs_literature": rp_over_rs_lit,
        "error_percent"       : error_pct,
        "semi_major_axis_AU"  : a_AU,
        "eq_temperature_K"    : T_eq
    }

    return folded, features, physical


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    from config import STAR_NAME
    from src.fetcher import fetch_lightcurve
    from src.preprocessor import preprocess
    from src.detector import run_bls

    lc                    = fetch_lightcurve(STAR_NAME)
    _, lc_flat, _         = preprocess(lc)
    bls_output            = run_bls(lc_flat)
    folded, features, physical = analyze_transit(lc_flat, bls_output)

    print("\n-- Physical Parameters ------------------")
    for k, v in physical.items():
        print(f"  {k:<25}: {v:.5f}")

    print("\n-- ML Features --------------------------")
    for k, v in features.items():
        print(f"  {k:<20}: {v:.6f}")