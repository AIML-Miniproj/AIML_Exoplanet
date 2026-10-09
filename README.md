# Exoplanet Transit Detection Pipeline

Detects exoplanetary transits from TESS photometric data using BLS algorithm + XGBoost classifier.

## Pipeline
1. **Fetch** (`src/fetcher.py`) — downloads the first TESS sector for a star via lightkurve/MAST and caches it in `data/`
2. **Preprocess** (`src/preprocessor.py`) — normalizes flux, sigma-clips outliers, flattens stellar variability
3. **Detect** (`src/detector.py`) — Box Least Squares periodogram over 0.5–15 days
4. **Analyze** (`src/analyzer.py`) — phase-folds at the best period, measures depth, Rp/Rs, semi-major axis, equilibrium temperature, and extracts ML features
5. **Classify** (`src/ml_model.py`) — trains an XGBoost transit-vs-noise classifier on augmented samples and scores the candidate
6. **App** (`app.py`) — Streamlit UI showing light curves, periodogram, folded transit, and results

All tunable parameters live in `config.py`.

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

Each stage can also be run on its own from the project root, e.g. `python -m src.detector`.

## Stack
- lightkurve (TESS data)
- astropy (BLS, constants)
- XGBoost (transit classifier)
- Streamlit (frontend)
