# src/detector.py

import numpy as np
from astropy.timeseries import BoxLeastSquares
from config import PERIOD_MIN, PERIOD_MAX, PERIOD_STEPS, TRANSIT_DURATIONS

def run_bls(lc_flat):
    """
    Runs BLS over multiple transit duration guesses.
    Returns best period, t0, duration, and full result object.
    """
    t       = lc_flat.time.value
    f       = lc_flat.flux.value
    periods = np.linspace(PERIOD_MIN, PERIOD_MAX, PERIOD_STEPS)

    best_power  = -np.inf
    best_result = None

    print("[Detector] Running BLS periodogram...")
    for dur in TRANSIT_DURATIONS:
        bls    = BoxLeastSquares(t, f)
        result = bls.power(periods, dur)
        peak   = np.max(result.power)
        print(f"[Detector] Duration {dur}d -> Peak power: {peak:.4f}")
        if peak > best_power:
            best_power  = peak
            best_result = result

    # Best parameters
    idx         = np.argmax(best_result.power)
    best_period = float(best_result.period[idx])
    t0          = float(best_result.transit_time[idx])
    duration    = float(best_result.duration[idx])

    # Top 3 candidates
    top3_idx     = np.argsort(best_result.power)[-3:][::-1]
    top3_periods = best_result.period[top3_idx]
    top3_powers  = best_result.power[top3_idx]

    print(f"[Detector] Best period  : {best_period:.5f} days")
    print(f"[Detector] Transit time : {t0:.4f} BTJD")
    print(f"[Detector] Duration     : {duration:.4f} days")

    return {
        "best_period" : best_period,
        "t0"          : t0,
        "duration"    : duration,
        "bls_result"  : best_result,
        "periods"     : periods,
        "top3_periods": top3_periods,
        "top3_powers" : top3_powers
    }


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    from config import STAR_NAME
    from src.fetcher import fetch_lightcurve
    from src.preprocessor import preprocess

    lc                         = fetch_lightcurve(STAR_NAME)
    _, lc_flat, _              = preprocess(lc)
    bls_output                 = run_bls(lc_flat)

    print(f"\nTop 3 candidate periods:")
    for i, (p, pw) in enumerate(zip(bls_output["top3_periods"],
                                     bls_output["top3_powers"])):
        print(f"  #{i+1}  {p:.5f} days  |  power: {pw:.4f}")