# Architecture

## 1. Audit of the starting point (Phase 0)

### Original flow

```
fetcher.py      first search result only, arrays pickled under data/<name>_lc.pkl
  -> preprocessor.py   normalize -> symmetric mean±5σ clip -> lightkurve flatten
  -> detector.py       astropy BLS, 5000 linear periods 0.5–15 d, durations 0.1/0.15 d
  -> analyzer.py       fold, in-transit = |phase| < 0.02, depth, Rp/Rs, a, T_eq, 9 features
  -> ml_model.py       XGBoost trained on the target's own features (+ wrong-period folds)
  -> app.py            Streamlit, free-text star name, 4 technical tabs, trains on every click
```

On the cached HD 209458 data this finds P = 3.52531 d (catalog 3.52475 d) and
Rp/Rs = 0.114 against 0.116 computed from the radii in the config.

### Known problems, checked against the code and real data

| # | Status | Finding |
|---|---|---|
| P1 | **Confirmed** | Positives are the target's own 9 features plus 5% noise; negatives are the same star folded at wrong periods. The model is rebuilt on every click and scores the very sample it was built from, so "Transit Detected, ~97%" is guaranteed for any star. |
| P2 | **Confirmed, plus a correction** | Only `search[0]` is downloaded. Which product that is depends on how MAST sorts the results, not on a policy: for HD 209458 it is a **20-second** SPOC product (117 k cadences), for Vega and Sirius a **30-minute** FFI product. So cadence silently differs between targets. |
| P3 | **Confirmed, severe** | The clip is symmetric and runs *before* detrending. On HD 209458, 5σ is 1.32% flux while the transit is ~1.4% deep: **all 2 886 removed cadences are in-transit, i.e. 66% of all in-transit points are deleted.** Depth is only recovered from what is left of ingress/egress. Deep eclipsing binaries lose their eclipses entirely. |
| P4 | Confirmed | `lc_normalized` is returned but only plotted. Every measurement uses the flattened curve, so stellar variability is lost. |
| P5 | Confirmed | 5000 linearly spaced periods: far too coarse at long periods relative to the required phase precision, wastefully fine at short periods. Only 2 durations (2.4 h and 3.6 h). |
| P6 | Confirmed | In-transit window is ±0.02 in phase. At P = 3.5 d that is ±1.7 h (about right for HD 209458), at P = 15 d it is ±7.2 h, much wider than any real transit, so depth is diluted. |
| P7 | Confirmed | `n_transits = baseline / period` ignores gaps. |
| P8 | Confirmed | Star radius, mass, Teff and the "literature" planet radius are HD 209458 constants. For any other star the physical numbers and the "error vs literature" are meaningless (e.g. 87.7% "error" for Vega). |
| P9 | Confirmed | `mean_flux_in` = 1 − depth to within noise. `sharpness` = depth / searched duration is a property of the search grid, not the signal. |
| P10 | Confirmed, plus a detail | Arrays are pickled; FITS are not kept; the cache key is the free-text name, so "HD 209458" and "HD209458" download twice. |
| P11 | Confirmed | Free-text input, `st.exception` shows full tracebacks, technical tabs only. |
| P12 | Confirmed | `print` logging, no CLI, no tests, no lint or CI. Logging is fixed in Phase 0. |

Other findings
- The flatten window was 201 *cadences* (67 min on 20 s data, shorter than the 3 h transit). Already changed to a 1-day window before Phase 0.
- The BLS ignores flux uncertainties.
- Feature extraction is copy-pasted between `analyzer.py` and `ml_model.py`.

## 2. Target layout

```
app.py                  Streamlit entry (UI rebuilt in Phase 6)
config.py               typed dataclass config, CONFIG instance, run presets
src/
  pipeline.py           analyze_target(target, preset, progress_cb) -> TargetReport
  report.py             TargetReport dataclass, JSON/row export
  physics.py            planet radius, orbit size, temperature, expected duration
  log.py                logging setup
  ingest/               resolver (TIC lookup), fetcher (multi-sector), catalogs
  preprocess/           per-sector cleaning, stitching, detrending
  detect/               BLS search (+ optional TLS)
  vetting/              transit, multi-sector, variability and stellar features
  triage/               labels, dataset, training, model loading
  recycler/             routing to planet / EB / variable / noise modules (Phase 5)
scripts/                offline jobs: labels, features, training
tests/                  offline tests on synthetic light curves (conftest.py)
docs/                   this file, MODEL_CARD.md
```

Rules
- The UI never imports science code directly: it calls `src.pipeline.analyze_target`
  and renders the returned `TargetReport`.
- Progress is reported through a callback `progress_cb(stage: str, fraction: float)`,
  so the UI and the CLI share the same logic.
- Network access sits behind small functions that tests can replace; unit tests use
  synthetic light curves only. Network tests are marked `@pytest.mark.network`
  and are skipped by default.
