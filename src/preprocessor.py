# src/preprocessor.py

import numpy as np
from config import FLATTEN_WINDOW_DAYS, SIGMA_CLIP_VAL

def preprocess(lc):
    """
    Cleans and flattens a raw light curve.
    Returns raw normalized, cleaned flattened, and trend.
    """
    # Step 1: Normalize flux to median = 1
    lc_normalized = lc.normalize()

    # Step 2: Manual sigma clip (compatible with all astropy versions)
    flux_vals = lc_normalized.flux.value
    mean      = np.mean(flux_vals)
    std       = np.std(flux_vals)
    mask      = np.abs(flux_vals - mean) < SIGMA_CLIP_VAL * std
    lc_clipped = lc_normalized[mask]
    print(f"[Preprocessor] Removed {np.sum(~mask)} outlier cadences")

    # Step 3: Flatten — remove long-term stellar variability
    # Window is set in days so it stays longer than the transit at any cadence
    cadence       = np.median(np.diff(lc_clipped.time.value))
    window_length = int(FLATTEN_WINDOW_DAYS / cadence) | 1   # must be odd
    window_length = min(window_length, (len(lc_clipped) - 1) | 1)
    lc_flat, trend = lc_clipped.flatten(
        window_length=window_length,
        return_trend=True
    )
    print(f"[Preprocessor] Remaining cadences: {len(lc_flat)}")

    return lc_normalized, lc_flat, trend


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    from config import STAR_NAME
    from src.fetcher import fetch_lightcurve

    lc = fetch_lightcurve(STAR_NAME)
    lc_normalized, lc_flat, trend = preprocess(lc)

    print(f"\nNormalized flux mean : {lc_normalized.flux.value.mean():.6f}")
    print(f"Flattened flux mean  : {lc_flat.flux.value.mean():.6f}")
    print(f"Flattened flux std   : {lc_flat.flux.value.std():.6f}")