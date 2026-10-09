# config.py

STAR_NAME         = "HD 209458"
MISSION           = "TESS"

# Preprocessing
FLATTEN_WINDOW_DAYS = 1.0   # must be several times longer than the transit
SIGMA_CLIP_VAL    = 5.0

# BLS Detection
PERIOD_MIN        = 0.5
PERIOD_MAX        = 15.0
PERIOD_STEPS      = 5000
TRANSIT_DURATIONS = [0.1, 0.15]

# Phase folding
PHASE_WIDTH       = 0.02

# Stellar parameters (HD 209458)
STAR_RADIUS_SOLAR = 1.2
PLANET_RADIUS_JUP = 1.359
STAR_MASS_SOLAR   = 1.148
STAR_TEFF         = 6065

# ML
TEST_SIZE         = 0.2
RANDOM_STATE      = 42
N_AUGMENTATIONS   = 40

FEATURE_NAMES = [
    "depth", "duration", "mean_flux_in",
    "std_flux_in", "std_flux_out", "snr",
    "symmetry", "sharpness", "n_transits"
]

# Paths
DATA_DIR    = "data/"
OUTPUT_DIR  = "outputs/"
MODEL_PATH  = "models/xgb_model.pkl"