# src/fetcher.py

import lightkurve as lk
import os
import pickle
from config import MISSION, DATA_DIR


def fetch_lightcurve(star_name):
    """
    Downloads first available TESS sector for a given star.
    Caches result locally using pickle (safe version).
    """

    os.makedirs(DATA_DIR, exist_ok=True)

    cache_file = os.path.join(
        DATA_DIR, f"{star_name.replace(' ', '_')}_lc.pkl"
    )

    # ── LOAD FROM CACHE ─────────────────────────────────────
    if os.path.exists(cache_file):
        try:
            print(f"[Fetcher] Loading cached data for {star_name}")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            from lightkurve import LightCurve
            lc = LightCurve(
                time=data["time"],
                flux=data["flux"],
                flux_err=data["flux_err"]
            )
            return lc

        except Exception:
            print("[Fetcher] Cache corrupted. Re-downloading...")
            os.remove(cache_file)

    # ── DOWNLOAD DATA ───────────────────────────────────────
    print(f"[Fetcher] Searching {MISSION} for {star_name}...")
    search = lk.search_lightcurve(star_name, mission=MISSION)

    if len(search) == 0:
        raise ValueError(f"No data found for {star_name} in {MISSION}")

    print(f"[Fetcher] Found {len(search)} sector(s). Downloading sector 1 only...")
    lc = search[0].download()
    lc = lc.remove_nans()

    print(f"[Fetcher] Done. Total cadences: {len(lc)}")

    # ── CONVERT TO SAFE DATA (NO ASTROPY OBJECTS) ───────────
    data = {
        "time": lc.time.value,
        "flux": lc.flux.value,
        "flux_err": lc.flux_err.value
    }

    # ── SAVE CACHE SAFELY ───────────────────────────────────
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"[Fetcher] Cached to {cache_file}")
    except Exception as e:
        print(f"[Fetcher] Warning: Could not cache data: {e}")

    return lc


# ── Quick test ─────────────────────────────────────────────
if __name__ == "__main__":
    from config import STAR_NAME

    lc = fetch_lightcurve(STAR_NAME)

    print(f"\nTime range : {lc.time.value[0]:.2f} -> {lc.time.value[-1]:.2f} BTJD")
    print(f"Cadences   : {len(lc)}")
    print(f"Flux mean  : {lc.flux.value.mean():.4f}")