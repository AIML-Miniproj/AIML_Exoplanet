# MASTER PROMPT — Multi-Sector Exoplanet Pipeline + "False Positive Recycler"

> **How to use:** put this file in the repo root (e.g. `docs/MASTER_PROMPT.md`), open Claude Code in the repo, and say:
> `Read docs/MASTER_PROMPT.md and execute Phase 0. Stop at the end of the phase and report.`
> Then run each later phase the same way. Don't run everything in one go.

---

## 0. Role and working rules

You are a senior astro-ML engineer and product-minded frontend developer extending an existing Python project, **AIML_Exoplanet**. You will refactor and extend it in **phases**. Rules:

1. **Read the whole repo first** (`config.py`, `app.py`, `src/*.py`, `README.md`, `requirements.txt`) before changing anything.
2. **Work one phase at a time.** At the end of each phase: run the tests, print a short report (what changed, files touched, test results, open issues), and **stop**. Do not start the next phase until I say so.
3. **Keep it runnable.** After every phase, `streamlit run app.py` must still start and analyse a star end to end, even if some panels show "coming soon".
4. **No silent fabrication.** If a catalog column name, API, or class label isn't what this prompt says, check it against the live docs or the returned data. Then adapt and tell me. Never hard-code a guessed value.
5. **No network in unit tests.** Tests use synthetic light curves. Network-dependent code goes behind functions that can be mocked, plus a separate `@pytest.mark.network` integration test.
6. **Reproducible:** set seeds, pin versions in `requirements.txt` once working, and cache every download.
7. **Cross-platform:** this must run on Windows and Linux. Use `pathlib`, and use `joblib` for parallelism (not fork-only tricks).
8. Prefer small, readable modules with docstrings and type hints. Use the `logging` module instead of `print`.
9. **No literature references anywhere in the repo.** That means no citations, paper names, or arXiv links in code, docstrings, comments, README, model card, the UI, or phase reports. Explain methods in plain words instead.
10. **The UI is for a non-expert.** Assume the user has never heard of a light curve. Every screen must make sense without reading the README (see §5).

---

## 1. Project context (current state)

Current pipeline (single TESS sector, single star):

`fetcher.py` (first sector only, pickle cache) → `preprocessor.py` (normalize, symmetric 5σ clip, lightkurve flatten 1 d) → `detector.py` (astropy BLS, linear period grid 0.5–15 d, 2 durations) → `analyzer.py` (phase fold, depth, Rp/Rs, a, T_eq, 9 features) → `ml_model.py` (XGBoost) → `app.py` (Streamlit, free-text star input, 4 technical tabs).

### Known problems you must fix

| # | Problem | Where | Why it matters |
|---|---|---|---|
| P1 | **Circular ML training.** Positives = the target's own features plus 5% noise; negatives = the same star folded at wrong periods. The model is retrained on every click and basically always says "Transit". | `ml_model.build_dataset`, `app.py` | Classifier has no real meaning. Replace with an offline model trained on catalog-labelled stars (Phase 3–4). |
| P2 | Only `search[0]` is downloaded: one sector. | `fetcher.py` | Fewer transits, worse SNR, no long periods, no cross-sector checks. |
| P3 | **Symmetric** sigma clip (mean ± 5σ) | `preprocessor.py` | Deletes deep eclipses and transits. Clip **upper outliers only**. |
| P4 | Only the flattened curve is used downstream. | pipeline | Flattening destroys stellar variability, which the Variable module needs. Keep both `lc_norm` (stitched, normalized, not flattened) and `lc_flat`. |
| P5 | Linear period grid and only 2 durations | `detector.py` | Use a frequency-uniform grid (`BoxLeastSquares.autoperiod`) and a geometric duration grid. |
| P6 | `PHASE_WIDTH = 0.02` fixed in phase units | `analyzer.py`, `ml_model.py` | In-transit window must be duration-based (|t − t_mid| < 0.5·duration, in days). |
| P7 | `n_transits = baseline / period` | `analyzer.py` | Wrong with multi-sector gaps. Count transits that actually have data (`compute_stats(...)['per_transit_count']`). |
| P8 | Stellar params hard-coded for HD 209458 | `config.py`, `analyzer.py` | Look up R★, M★, Teff, logg, and contamination per target from the TIC. Show a literature comparison only when the target is a known planet (NASA Exoplanet Archive `pscomppars`). |
| P9 | `mean_flux_in` ≈ 1 − depth (redundant); `sharpness` is weakly defined | features | Replace with the vetting features in Phase 2. |
| P10 | Pickle cache of arrays | `fetcher.py` | Cache per-sector FITS via lightkurve's `download_dir` plus processed `.npz`/parquet. |
| P11 | Free-text star input; raw exceptions shown to users; emoji-heavy technical tabs only | `app.py` | Replace with the guided UX in §5. |
| P12 | `print` logging, no CLI, no CI, no lint | repo | Add `logging`, a CLI, ruff, and GitHub Actions (Phase 7). |

---

## 2. Goal of this upgrade

Three things (keep scope tight):

### A. Multi-sector data
Use **all available TESS sectors** for a target (with a user-selectable cap) and stitch them. Use multi-sector information as a feature source (per-sector depth consistency, per-sector period stability).

### B. "False Positive Recycler"
Change the question from "planet or noise?" to **triage → route → characterise**:

```
Target (picked from dropdown / search, or TIC)
  → resolve TIC + stellar params
  → multi-sector ingest + stitch   ──► lc_norm (variability kept)
  → detrend (wotan biweight)        ──► lc_flat
  → periodic signal search (BLS, optional TLS) on lc_flat
  → vetting feature extraction (lc_flat + lc_norm + stellar params + per-sector stats)
  → TRIAGE classifier  →  P(PLANET), P(EB), P(VARIABLE), P(NOISE)
  → RECYCLER ROUTER
        PLANET   → Planet module    (Rp, a, T_eq with uncertainties)
        EB       → EB module        (eclipse characterisation)
        VARIABLE → Variable module  (variability type + period)
        NOISE    → Systematics log  (why it was rejected; kept for retraining)
  → TargetReport (dataclass/JSON) → UI + batch CSV + downloadable summary
```

Nothing is thrown away. Every rejection from the planet path becomes a characterised object in another category.

### C. A frontend a beginner can use
Guided, plain-language, dropdown-driven, with instant demo results and an expert mode for details. Full spec in §5.

---

## 3. Target architecture

```
AIML_Exoplanet/
├── app.py                         # Streamlit entry: page config, nav, theme
├── pages/                         # Streamlit multipage (or st.navigation)
│   ├── 1_Home.py
│   ├── 2_Analyze_a_Star.py
│   ├── 3_Scan_Many_Stars.py
│   ├── 4_Explore_Results.py
│   └── 5_How_It_Works.py
├── ui/
│   ├── components.py              # verdict card, step indicator, glossary tooltips, empty/error states
│   ├── plots.py                   # Plotly figures with annotations + plain-language captions
│   ├── explain.py                 # feature/SHAP → plain-English sentences
│   ├── text.py                    # ALL user-facing copy + glossary in one place
│   └── state.py                   # session_state helpers, URL query params, recent history
├── .streamlit/config.toml         # theme (works in light and dark), no "deploy" clutter
├── config.py                      # dataclass-based config
├── requirements.txt
├── README.md
├── docs/
│   ├── ARCHITECTURE.md
│   └── MODEL_CARD.md              # plain-language: data, labels, splits, metrics, limitations (no citations)
├── src/
│   ├── pipeline.py                # analyze_target(target, preset, progress_cb) -> TargetReport
│   ├── report.py                  # TargetReport dataclasses + to_json/to_row/to_html
│   ├── cli.py                     # python -m src.cli analyze "HD 209458" --preset quick
│   ├── ingest/{resolver,fetcher,catalogs}.py
│   ├── preprocess/preprocessor.py
│   ├── detect/{bls,tls}.py
│   ├── vetting/{transit_features,sector_features,variability_features,stellar_features}.py
│   ├── triage/{labels,dataset,train,model}.py
│   ├── recycler/{router,planet_module,eb_module,variable_module,systematics_log}.py
│   └── physics.py
├── scripts/
│   ├── build_labels.py
│   ├── build_features.py          # parallel, resumable, --limit-per-class
│   ├── train_triage.py
│   ├── train_variable.py
│   ├── build_examples.py          # builds the dropdown example list + precomputed demo results
│   └── batch_scan.py
├── data/
│   ├── examples/                  # COMMITTED: examples.json + small precomputed demo reports
│   └── raw/ processed/ labels/ logs/ (gitignored)
├── models/                        # small trained models may be committed if < ~5 MB, else gitignored + instructions
├── tests/
│   ├── conftest.py                # synthetic light-curve factories
│   ├── test_preprocess.py  test_detect.py  test_vetting.py  test_router.py
│   ├── test_pipeline_offline.py
│   └── test_app_smoke.py          # streamlit.testing.v1.AppTest
└── .github/workflows/ci.yml
```

`ui/` must never import heavy science code directly. It calls `src.pipeline` and renders `TargetReport`. The pipeline reports progress through a callback, so the UI and the CLI share the same logic.

---

## 4. Phase plan

### Phase 0 — Audit, scaffolding, test harness
- Read everything. Write a short audit in `docs/ARCHITECTURE.md` (current flow + the P1–P12 table, confirmed or corrected).
- Create the new layout. Move code without changing behaviour yet.
- Convert `config.py` to a typed dataclass config.
- `tests/conftest.py` **synthetic generators**. Each produces multiple 27-day sectors with realistic gaps, as `lightkurve.LightCurve` objects:
  - `make_planet_lc(period, depth, duration, n_sectors, cadence, noise)`: U-shaped (trapezoid; optional `batman`)
  - `make_eb_lc(period, depth1, depth2, phase2, odd_even_diff, ellipsoidal_amp, ...)`: V-shaped primary + secondary eclipse
  - `make_variable_lc(kind="sinusoid"|"multiperiodic"|"spotted", period, amp)`
  - `make_noise_lc(...)`: white + red noise, gaps, spikes
- Replace `print` with `logging`. Add `pytest` and a smoke test on a synthetic planet.
- **Stop and report.**

### Phase 1 — Multi-sector ingest + preprocessing
**Resolver**: `resolve_target(name_or_tic) -> TargetInfo(tic_id, common_name, ra, dec, tmag, teff, rad, mass, logg, rho, contratio, gaia_id, ...)` via `astroquery.mast.Catalogs` (TIC). NaNs get flagged, not crashed on. Cache as JSON. Optional Gaia DR3 `ruwe` (feature-flagged).

Also add `list_available_sectors(tic) -> list[SectorInfo]`. It is cheap (search only, no download), and the UI uses it for the pre-run preview.

**Fetcher**:
- Author/cadence policy (configurable): SPOC 120 s → TESS-SPOC FFI → QLP. Use one product per sector; never mix 20 s and 120 s for a sector.
- `download_all(download_dir=...)`, retry with backoff and timeouts, and log the sectors used.
- `MAX_SECTORS` / explicit `SECTORS`.
- Return a `LightCurveCollection` + `sector_info`.

**Preprocessor**:
- Per sector: `remove_nans()`, quality mask, **upper-only** outlier clip, normalize by sector median.
- Stitch → `lc_norm`.
- Detrend per sector with `wotan.flatten(method="biweight", window_length=..., break_tolerance=0.5, return_trend=True)` → `lc_flat`. The window must be at least 3× the longest searched duration.
- Iterative re-detrend with the transit masked after detection.
- Optional 10-min binning for the search step only.

**Tests**: EB eclipse depth survives preprocessing (regression for P3); gaps don't break detrending; `lc_norm` keeps a sinusoid's amplitude within 5%.

**Stop and report.**

### Phase 2 — Detection + vetting features
**Detection**:
- Durations `np.geomspace(0.02, 0.5, ~12)` d. Build the period grid with `autoperiod(..., maximum_period=min(PMAX, baseline/2))`. Warn on huge grids.
- Compute SDE, plus top-N distinct peaks with aliases removed.
- `compute_stats` → depth, depth_odd, depth_even, depth_half, depth_phased, harmonic_delta_log_likelihood, per_transit_count, transit_times.
- **Always evaluate 2P too** (EBs with similar eclipses get detected at P/2).
- Optional TLS (`USE_TLS`, off by default). Optional multi-signal search (mask and re-run, max 3 signals).

**Vetting features** (each documented in plain words in `MODEL_CARD.md`):
- *Transit/eclipse:*
  - `depth_ppm`, `depth_snr`, `duration_hr`, `period_d`, `sde`, `n_transits_observed`
  - `odd_even_depth_sigma`
  - `secondary_depth_ppm`, `secondary_snr` at phase 0.5, plus best secondary phase searched over 0.3–0.7
  - Trapezoid fit → `ingress_to_total_ratio` (V vs U shape), `trapezoid_chi2_vs_box_chi2`
  - `transit_asymmetry`, `harmonic_dll`
  - `implied_rp_rearth` and an `implied_rp_gt_2rjup` flag
  - `duration_ratio_expected` (observed duration vs the duration expected from stellar density and period)
- *Multi-sector:* `sector_depth_cv`, `sector_depth_chi2`, `n_sectors`, `n_sectors_with_transit`, `max_sector_depth_ratio`, optional `crowdsap_min` and `crowdsap_range`.
- *Variability (from `lc_norm`):*
  - Lomb–Scargle top-3 periods, powers, and FAP
  - `ls_period_matches_bls` (also at 0.5× and 2×)
  - `rms_ppm`, `amp_p2p_ppm`, `skew`, `kurtosis`
  - ACF period
  - Fourier `R21`, `R31`, `phi21`, `phi31`
  - `ls_period_sector_std`
  - Phase-curve `cos 2φ` (ellipsoidal) and `cos φ` (reflection) amplitudes
- *Stellar:* `teff`, `rad`, `logg`, `tmag`, `contratio`, optional `ruwe`.

**Tests**: odd/even separates EB vs planet; secondary SNR is high for the EB; the shape ratio separates V vs U; LS recovers the variable's period.

**Stop and report.**

### Phase 3 — Labelled dataset (real labels, no circular training)
`data/labels/labels.parquet`: `tic_id, label, source, catalog_period, catalog_epoch, catalog_depth, notes`.

| Class | Source | Rule |
|---|---|---|
| **PLANET** | NASA Exoplanet Archive `toi` table (TAP) | `tfopwg_disp in ("CP","KP")` |
| **EB** | TESS EB catalog (MAST HLSP `tess-ebs` / VizieR `J/ApJS/258/16`) | All entries; TOI `FP` only if the TIC is also in the EB catalog |
| **VARIABLE** | Gaia DR3 `vari_classifier_result` (via TIC `GAIA` id) and/or TESS-SVC HLSP (`tess-svc`) | Non-`ECL` classes; exclude any TIC in the EB catalog or TOI list |
| **NOISE** | TOI `FA` + random quiet SPOC 2-min TICs not in any list above | |
| *held out* | TOI `PC`/`APC` | **Never train.** Used for the demo / "unconfirmed candidates" dropdown. |

- Verify column names and Gaia class strings from live data.
- Conflict priority: CP/KP > EB catalog > Gaia variable > FA/random. Log every conflict.
- v1 uses SPOC 2-min only. `--limit-per-class` defaults to **25** (academic scale, ~100 stars total; see §11).
- Reuse the catalogs and labels already in `data/`. Do not re-download catalogs.
- Training-star selection prefers Tmag 8–12, ≥2 SPOC 2-min sectors, and (for EB/PLANET) a catalog period of 0.5–15 d.

**Features**: run the **actual pipeline** on each labelled TIC (parallel, resumable, failures logged with reasons). Add a **recovery check**: a detected period matching the catalog within 1% (or at 0.5×/2×/3×) counts as `recovered`. Features always come from the pipeline, never from the catalog ephemeris. Report the recovery rate per class.

**Stop and report** (class counts, recovery rates, failure reasons).

### Phase 4 — Triage classifier
- 4-class XGBoost with class weights. **5-fold `StratifiedGroupKFold` by TIC**, reporting mean ± std across folds. With ~25 stars per class there is no separate held-out test set.
- Baselines: logistic regression and random forest on the same splits.
- Metrics:
  - Per-class precision, recall, F1, and PR-AUC; macro-F1
  - Normalized confusion matrix
  - Planet recall at a fixed EB contamination rate
- Probability calibration (isotonic or sigmoid) + reliability plot.
- SHAP `TreeExplainer`: a global summary, plus a per-candidate top-5 used by `ui/explain.py`.
- Save `models/triage_xgb.json` (native format) + `models/triage_meta.json` (features, class order, thresholds, date, label hash, metrics).
- `docs/MODEL_CARD.md` in plain language, with no citations. State honestly that flux-only vetting cannot rule out background eclipsing binaries, and that the dataset is small (academic scale).
- **Remove all training from `app.py`.** The app only loads the model.

**Stop and report** (metrics table in the report).

### Phase 5 — Recycler modules
- **Router**:
  - `route(probas, features) -> RoutingDecision(primary_class, confidence_label, secondary_class, reasons[])`
  - If max P < `UNCERTAIN_THRESHOLD`, the result is `NEEDS_REVIEW` and the top-2 modules both run.
  - Physics overrides are logged as reasons: implied Rp > 2.5 R_jup → not a planet; secondary SNR > 7 → EB unless the phase curve explains it.
  - Confidence labels: High (≥0.8), Medium (0.5–0.8), Low (<0.5). Configurable.
- **Planet module**: moved physics, Monte Carlo uncertainties from the stellar-parameter errors, and a size-category label (Earth-size / super-Earth / Neptune-size / Jupiter-size). Known-planet lookup → "matches known planet X".
- **EB module**:
  - True period (P vs 2P); primary/secondary depths, phases, and widths
  - `e·cosω` and `e·sinω` approximations (labelled as approximate)
  - Temperature-ratio estimate (assumptions stated)
  - Morphology class (detached / semi-detached / contact): trained on the EB catalog's `morph` value, with a rule-based fallback
  - Eclipse-timing O−C across sectors, with a "possible third body" flag
  - Catalog cross-match
- **Variable module** (uses `lc_norm`): coarse classes `ROTATION/SPOTTED`, `DELTA_SCUTI`, `GAMMA_DOR`, `RR_LYRAE`, `CEPHEID`, `LONG_PERIOD/IRREGULAR`, `OTHER`. Trained on Gaia labels mapped to these classes, with the mapping in the model card. Outputs the period, amplitude, and stability across sectors.
- **Systematics log**: NOISE and NEEDS_REVIEW results are appended to `data/logs/recycler_log.parquet`.
- Every module also fills a `plain_summary: str` and `key_facts: list[(label, value, unit, tooltip)]` in `TargetReport`, which the UI renders directly.
- Tests: router unit tests; EB `e·cosω` recovery; variable period recovery.

**Stop and report.**

### Phase 6 — Frontend overhaul (spec in §5)
Build everything in §5. Also:
- `scripts/build_examples.py`:
  - The demo set is **exactly 4 stars**: one confirmed planet (HD 209458 b), one TOI false positive that is an eclipsing binary, one variable star, and one unconfirmed TOI candidate. Require good data (≥2 SPOC sectors, bright, obvious signal).
  - Runs the full pipeline on each and saves `data/examples/examples.json` (display name, TIC, group, one-line description, sector count) plus `data/examples/demo/<tic>.json` precomputed reports. Keep the total small (< 10 MB) and committed.
  - If one of the 4 demo stars is misclassified, keep it and explain it honestly. Don't hide mistakes.
- `test_app_smoke.py` uses `streamlit.testing.v1.AppTest`: every page loads, picking a demo example renders a verdict with no network, and a bad TIC input shows a friendly error instead of a traceback.

**Stop and report** (include screenshots or a description of each page).

### Phase 7 — Batch, CLI, polish, CI
- `scripts/batch_scan.py` + `src/cli.py` (`analyze`, `batch`, `examples` subcommands). Both use `pipeline.analyze_target`.
- Results cache keyed by `(tic, config_hash, model_version)`, so re-opening a star is instant.
- `ruff` config; GitHub Actions running ruff + offline pytest on Linux and Windows.
- README rewrite: a 3-line "what this is", screenshots, quickstart, then "for developers" (build labels → features → train → build examples). No references section.
- **Final acceptance checklist** (run it and paste the results):
  - [ ] `pytest -q` passes offline; CI is green.
  - [ ] A fresh user can open the app, pick "HD 209458 b" from the dropdown, and see a result in under 3 s (demo cache) with zero typing.
  - [ ] Live run of HD 209458 → **PLANET**, P ≈ 3.5247 d, Rp ≈ 1.3–1.4 R_jup.
  - [ ] 3 catalog EBs → **EB** with the correct (not half) period. 3 Gaia variables → **VARIABLE** with the period within 1%.
  - [ ] Triage metrics are in `MODEL_CARD.md` and shown in plain language on the How It Works page.
  - [ ] Batch of 20 TOI candidates completes from the UI and CSV download works.
  - [ ] A wrong star name, a star with no TESS data, and a MAST timeout each show a friendly message with a suggested next step.

---

## 5. Frontend / UX specification

**Audience:** a curious person with no astronomy or ML background, such as a hackathon judge, a classmate, or a recruiter. **Secondary:** an expert who wants the plots and numbers. Design for the first and let the second opt in.

### 5.1 Global
- **Multipage app** with sidebar navigation: *Home · Analyze a Star · Scan Many Stars · Explore Results · How It Works*.
- **Beginner / Expert toggle** in the sidebar (default Beginner, remembered in session):
  - *Beginner*: plain-language verdict, key facts, 2–3 annotated charts, and "why" sentences.
  - *Expert*: adds the technical tabs (periodogram, odd/even, secondary, per-sector depths, LS, raw probabilities, SHAP values, full feature table).
- **All user-facing text lives in `ui/text.py`.** Short sentences, no jargon without a tooltip. Avoid emoji clutter; at most one small icon per section header.
- **Glossary tooltips** (`help=` on widgets, plus a hover/expander component) for: light curve, transit, eclipse, eclipsing binary, variable star, period, sector, TIC ID, confidence, false positive.
- **Consistent class colours** everywhere (Planet, EB, Variable, Noise/Review). Use a colourblind-safe palette that works in both light and dark theme, and always pair colour with a text label.
- **Responsive layout**: no horizontal scroll on narrow screens; use columns that stack.
- **Shareable state**: the selected star and preset go into URL query params (`?tic=…&preset=quick`), so a link opens straight to that result.
- **Errors**: never show a raw traceback by default. Show a friendly message plus a suggested action ("This star has no TESS data. Try one of these examples."). Put technical details in a collapsed "Show technical details" expander.
- **Model missing**: if no trained model exists, show a setup card with exact commands and keep demo examples working.

### 5.2 Home page
- Hero: one sentence on what the tool does ("Find planets around other stars, and figure out what's really going on when it isn't a planet").
- A 3-step visual: **Pick a star → We analyse its brightness → You get a verdict.**
- A 30-second explainer with a tiny animated or annotated example: a light curve with a dip and the label "a planet passing in front of its star blocks a little light".
- Four clickable cards, one per outcome (Planet / Eclipsing binary / Variable star / Noise). Each has a one-line explanation and a "See an example" button that jumps to Analyze with that demo preloaded.
- A primary button: **"Try an example"**.

### 5.3 Analyze a Star page (the main flow)
**Step indicator at the top**: 1 Choose star → 2 Settings → 3 Results.

**Step 1: Choose a star (no typing required)**
- A **grouped dropdown** (`st.selectbox`, searchable) populated from `data/examples/examples.json`, with groups: *Confirmed planets · Eclipsing binaries · Variable stars · Unconfirmed candidates · Tricky cases*. Each option shows the display name and a short hint, e.g. "HD 209458 b: hot Jupiter, 3.5-day orbit".
- A second option, *"Search all TESS candidates"*: a searchable selectbox over the cached TOI list (TOI number, TIC, and host name if any), loaded once and cached.
- A third option, *"I know the TIC ID / star name"*: a text input with validation, shown only in an expander. On submit, resolve it and show "Found: TIC 420814525 (HD 209458)" before running, or a friendly "not found" with suggestions.
- After a star is picked, show a **star card**: name, TIC, brightness, temperature as a colour swatch ("Sun-like", "cooler than the Sun"), size vs the Sun, and **how many TESS sectors are available**, with a small timeline of which sectors observed it.
- A **"Random example"** button.

**Step 2: Settings (sensible defaults, optional)**
- A **preset segmented control**:
  - *Quick* (up to 3 sectors, BLS only, ~30–60 s)
  - *Standard* (up to 10 sectors, default)
  - *Thorough* (all sectors + TLS + multi-signal search)
- Show the **estimated runtime and download size** before running, based on the sector count and cadence.
- Advanced settings in a collapsed expander, each with a `help=` tooltip: period range slider, sector multiselect, detrend window, enable TLS, max signals.
- One big **"Analyze"** button. Disable it while running.

**Step 3: Results**
- **Live progress** while running (`st.status` or a progress bar with stage names in plain words): "Downloading 6 sectors of TESS data (3/6)…", "Cleaning the data…", "Searching for repeating dips…", "Checking whether it's really a planet…", "Writing your report…". Allow cancel where feasible.
- Demo examples load instantly from the precomputed reports and show the badge "Precomputed example: click Re-run to analyse live".
- **Verdict card** (top, large):
  - Class label with its colour, plus a confidence label (High/Medium/Low) and a simple meter. Raw probabilities appear only in Expert mode.
  - A one-sentence plain summary from the module's `plain_summary`, e.g. "This looks like a planet about 1.4× the size of Jupiter that circles its star every 3.5 days."
  - For a recycled object: "Not a planet, but still interesting: this is a pair of stars eclipsing each other every 2.1 days."
  - NEEDS_REVIEW: "We're not sure. It could be X or Y; here's why."
- **"Why do we think so?"**: 3–5 plain-English reasons generated by `ui/explain.py` from SHAP + features. Examples: "The dips are U-shaped like a planet's, not V-shaped like a star's"; "Alternating dips have the same depth"; "No second, smaller dip halfway between, which eclipsing stars usually show"; "The dip depth stays the same across all 6 sectors".
- **Catalog check** when the star is known: "NASA's catalog lists this as a confirmed planet: we agree ✔" or "…: we disagree, see Tricky cases". Present this honestly.
- **Key facts** grid from `key_facts`, with units and tooltips. Planet: size, orbit period, distance from star, temperature, size category with a size comparison graphic vs Earth/Neptune/Jupiter. EB: period, eclipse depths, binary type. Variable: type, period, amplitude.
- **Charts** (Plotly, interactive, each with a one-line caption and a "What am I looking at?" expander):
  1. Full light curve, with sectors shaded and labelled and detected dips marked.
  2. "All dips stacked on top of each other" (phase fold) with the binned curve and the model overlay.
  3. Module-specific chart: planet = size comparison; EB = primary vs secondary eclipse side by side; variable = periodogram with the main period highlighted.
- **Expert tabs** (Expert mode only): Detection (periodogram + SDE + top peaks), Vetting (odd/even, secondary, trapezoid fit, per-sector depths), Variability (LS, ACF), Model (probabilities, SHAP waterfall, full feature table), Data (sector table, cadence, author).
- **Actions**: Download report (a single self-contained HTML summary with charts), Download data (JSON + CSV), Copy share link, Analyze another star.
- **Recent analyses** in the sidebar (session history) for one-click return.

### 5.4 Scan Many Stars page
- Input options:
  - Preset lists: "20 unconfirmed TESS candidates", "10 known eclipsing binaries", "Mixed demo set"
  - Upload a CSV (a downloadable **template** is provided, and the upload is validated with clear row-level errors)
  - Multiselect from the examples
- Preset selector (Quick/Standard), estimated total time, and a hard cap on star count with a warning.
- Progress per star (table updating live: queued / running / done / failed + reason).
- Results table: class (coloured chip), confidence, period, size or type, and catalog agreement. Filters by class and confidence, and sorting. Clicking a row opens that star on the Analyze page.
- A summary chart: counts per class (bar) and confidence distribution.
- Download CSV.

### 5.5 Explore Results page
- Browse everything analysed so far (cache + recycler log): filters by class, confidence, and NEEDS_REVIEW.
- A **"Most interesting"** section: unconfirmed candidates the model rates as planet with high confidence, plus NEEDS_REVIEW cases.
- A recycler visual: a Sankey or bar showing how candidates flowed (signals found → planet / EB / variable / noise).

### 5.6 How It Works page
- Plain-language walkthrough of the pipeline as a simple diagram with one sentence per step.
- What "false positive" means and why recycling them is useful.
- How the model was trained (which kinds of stars, how many, from public NASA/ESA catalogs; just names of data sources, no citations).
- How good it is: the metrics in plain words ("Out of 100 real planets, it correctly flags about N"), plus the confusion matrix.
- Limitations, honestly stated: flux-only, can't rule out background stars, trained on 2-min data, etc.
- FAQ: "What's a TIC ID?", "Why does it take a minute?", "Why might it be wrong?"

### 5.7 Performance
- `st.cache_resource` for the model and catalogs; `st.cache_data` for resolver, sector lists, and reports.
- Lazy-load expert charts. Downsample long light curves for plotting (e.g. ≤ 20k points, with binned overlay); never downsample for analysis.
- Run analysis so the UI stays responsive (progress callback; the pipeline must not block rendering of already-finished sections where feasible).

---

## 6. Dependencies to add
`astroquery`, `wotan`, `pyarrow`, `shap`, `tqdm`, `pytest`, `plotly` (already), `ruff`.
Optional: `transitleastsquares`, `batman-package`, `exovetter` (cross-check metrics only).
Pin versions once Phase 7 passes.

## 7. Config defaults (in config, not code)
`AUTHOR_PRIORITY=["SPOC","TESS-SPOC","QLP"]`, presets `{quick: max_sectors=3, tls=False, max_signals=1; standard: max_sectors=10, tls=False, max_signals=2; thorough: max_sectors=None, tls=True, max_signals=3}`, `DETREND_METHOD="biweight"`, `DETREND_WINDOW_D=0.75`, `CLIP_SIGMA_UPPER=5`, `PERIOD_MIN_D=0.5`, `PERIOD_MAX_D=30`, `DURATIONS_D=geomspace(0.02,0.5,12)`, `MAX_GRID_SIZE=40000`, `SDE_MIN=7`, `UNCERTAIN_THRESHOLD=0.5`, `CONF_HIGH=0.8`, `LIMIT_PER_CLASS=25`, `BATCH_MAX_STARS=50`, `PLOT_MAX_POINTS=20000`, `N_JOBS=2` (never more than 3), `RANDOM_STATE=42`.

Training / feature-build mode: `MAX_SECTORS=2`, 10-minute binning for the search step only, `PERIOD_MAX_D=15`, per-star timeout 120 s.

## 8. Things NOT to do
- Don't train on TOI PCs/APCs. Don't split by row; split by TIC.
- Don't retrain any model inside the Streamlit request path.
- Don't use the catalog ephemeris to build training features.
- Don't claim the system rules out background eclipsing binaries.
- Don't put citations or reference lists in code, docs, the UI, or reports.
- Don't show raw tracebacks, raw probabilities, or unexplained jargon in Beginner mode.
- Don't cherry-pick: the Tricky cases group and honest catalog disagreements stay visible.
- No modules outside this scope (no RV, JWST, habitability, etc.).

## 9. Report format at the end of each phase
```
## Phase N report
Changed files: ...
New files: ...
Tests: X passed / Y failed (summary)
Key results / metrics: ...
Deviations from the prompt (and why): ...
Open issues / questions for me: ...
Next phase preview: ...
```

## 10. Stretch ideas (only if I ask later)
Pixel-level centroid tests; a 1D-CNN second triage model; an active-learning "help us label" page fed from `recycler_log.parquet`; Kepler labels for cross-mission training; deployment to Streamlit Community Cloud or Hugging Face Spaces.

## 11. Scope and resource limits (academic project)
- **Training set:** 25 stars per class (PLANET, EB, VARIABLE, NOISE), about 100 stars in total.
- **Demo set:** exactly 4 stars (see Phase 6).
- **Never allocate the uncapped BLS period grid.** Compute its size from a formula and build the capped grid directly.
- **CPU:** at most 3 parallel workers (default 2).
- **Per-star timeout:** 120 s. On a timeout, skip the star, log the reason "timeout", and continue.
- **Progress:** process stars in chunks of 20 and print a progress line after each chunk (done / failed / avg s per star / ETA).
- **Long jobs:** if a job is estimated to take more than 30 minutes, stop and ask first. Never launch a background job and wait without reporting.
- Run a 5-star pilot before any full feature build, and report its time, peak RAM and failures.
