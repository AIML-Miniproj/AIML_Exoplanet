<div align="center">

# 🪐 Exoplanet Atlas

### Exoplanet Detection & False-Positive Recycling in TESS Light Curves<br>using XGBoost and Physics-Based Vetting

*Finds transiting planets in NASA TESS data, and when a signal is **not** a planet,<br>works out what it really is instead of throwing it away.*

[![CI](https://github.com/AIML-Miniproj/AIML_Exoplanet/actions/workflows/ci.yml/badge.svg)](https://github.com/AIML-Miniproj/AIML_Exoplanet/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-189FDD)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-8A2BE2)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-148%20passing-2EA44F)
![Macro F1](https://img.shields.io/badge/macro--F1-0.76%20±%200.10-54D8E8)

[**Overview**](#-overview) · [**The app**](#-the-app) · [**Architecture**](#-system-architecture) · [**Pipeline**](#-the-pipeline) · [**Results**](#-results) · [**Run it**](#-getting-started) · [**Model card**](docs/MODEL_CARD.md)

<img src="docs/screenshots/home.png" alt="Exoplanet Atlas home page" width="100%">

</div>

---

## 📑 Table of contents

1. [Overview](#-overview)
2. [Highlights](#-highlights)
3. [The app](#-the-app)
4. [System architecture](#-system-architecture)
5. [The pipeline](#-the-pipeline)
6. [Dataset](#-dataset)
7. [Features](#-features)
8. [Model and training](#-model-and-training)
9. [Results](#-results)
10. [Explainable AI](#-explainable-ai)
11. [Technology stack](#-technology-stack)
12. [Project structure](#-project-structure)
13. [Getting started](#-getting-started)
14. [Rebuilding everything](#-rebuilding-everything-from-scratch)
15. [MLOps, testing and CI](#-mlops-testing-and-ci)
16. [Deployment](#-deployment)
17. [Limitations](#-limitations)
18. [Future scope](#-future-scope)
19. [References](#-references)
20. [Credits and acknowledgements](#-credits-and-acknowledgements)

---

## 🔭 Overview

**The problem.** Most known exoplanets were found with the *transit method*: when a
planet crosses the face of its star, the star dims slightly, once per orbit. NASA's
**TESS** satellite records the brightness of hundreds of thousands of stars in 27-day
*sectors*, so it sees these repeating dips constantly. But a dip is not proof of a
planet. **Eclipsing binaries** (two stars eclipsing each other), **variable stars**
(pulsations, starspots) and **instrumental noise** all produce look-alike dips, and a
large share of TESS candidates turn out to be such false positives. Vetting them by
eye does not scale.

**What this project does.** An end-to-end pipeline takes any star, downloads its
multi-sector TESS light curves, cleans them without damaging transits, searches for
periodic dips with **Box Least Squares**, and describes the strongest signal with
**56 physically meaningful features**. A calibrated **four-class XGBoost** model
(planet, eclipsing binary, variable star, noise) then makes a call, which a
**physics router** can overrule. Instead of discarding non-planets, every signal is
**recycled** into a dedicated module that reports what it really is: a planet's size
and orbit, a binary's true period and configuration, or a variable star's type.

**Why it matters.** Deep-learning vetters reach high accuracy but need thousands of
labelled examples and answer only *planet or not*. This project shows that a small,
carefully labelled dataset (**100 stars**) combined with domain knowledge gives a
useful, **multi-class and explainable** triage tool. Every verdict comes with
calibrated probabilities and plain-English reasons.

> **Problem statement:** design and implement a machine-learning pipeline that takes
> the TESS light curve of any star, detects periodic dips, classifies the strongest
> signal as planet, eclipsing binary, variable star or noise with calibrated
> confidence, explains the decision, and characterises the object instead of
> discarding non-planet signals.

---

## ✨ Highlights

| | |
|---|---|
| 🛰️ **Real NASA data** | Multi-sector, 2-minute TESS light curves straight from the MAST archive |
| 🧹 **Transit-safe cleaning** | Upper-side-only outlier clipping and masked biweight detrending keep every dip at full depth |
| 🔍 **Robust search** | Box Least Squares over 0.5–30 days with refinement, prewhitening for fast pulsators |
| 📐 **56 physics features** | Dip shape, odd/even, secondary eclipse, sector consistency, variability, stellar parameters |
| 🤖 **Calibrated XGBoost** | Four classes, Platt-calibrated probabilities, honest grouped cross-validation |
| ⚖️ **Physics router** | Size limit, secondary-eclipse rule and a 50 % review threshold can overrule the model |
| ♻️ **False-positive recycling** | Planet, eclipsing-binary and variable-star modules characterise every signal |
| 💬 **Explainable** | A plain-English *"Why?"* for every verdict, built from physics checks and SHAP |
| 🌌 **Polished web app** | Five pages: Home, Learn, Analyzer, Demo Stars and How It Works |
| 🐳 **MLOps-ready** | 148 offline tests, GitHub Actions CI on Ubuntu and Windows, Docker, Streamlit Cloud |

---

## 🖥️ The app

The interface has five pages. The **Analyzer** runs the pipeline; the other four
explain the science in plain language. All six demo stars open instantly from
precomputed reports, and *Re-run live analysis* downloads fresh TESS data (about 1–2
minutes).

<table>
<tr>
<td width="50%"><img src="docs/screenshots/analyzer.png" alt="Analyzer"><br><b>Analyzer</b>: pick a demo star in the sidebar, press <i>Analyze</i>; the selected star's card shows its class, period and depth.</td>
<td width="50%"><img src="docs/screenshots/results.png" alt="Results tab"><br><b>Results</b>: verdict with confidence, class probabilities, the <i>"Why?"</i> list, NASA catalog check, findings and model evaluation.</td>
</tr>
<tr>
<td><img src="docs/screenshots/lightcurves.png" alt="Light curves tab"><br><b>Light Curves</b>: raw versus cleaned light curve, sectors shaded.</td>
<td><img src="docs/screenshots/bls.png" alt="BLS tab"><br><b>BLS Detection</b>: periodogram with the best period and its SDE.</td>
</tr>
<tr>
<td><img src="docs/screenshots/phasefold.png" alt="Phase-folded tab"><br><b>Phase-Folded</b>: TOI 184.01 folded on its true period, with primary vs secondary and odd vs even eclipses.</td>
<td><img src="docs/screenshots/demo_stars.png" alt="Demo stars page"><br><b>Demo Stars</b>: the six held-out stars with their real verdicts; each card opens the Analyzer.</td>
</tr>
<tr>
<td><img src="docs/screenshots/learn.png" alt="Learn page"><br><b>Learn</b>: a plain-language glossary of every idea the app uses, with real demo stars as examples.</td>
<td><img src="docs/screenshots/how_it_works.png" alt="How it works page"><br><b>How It Works</b>: the pipeline told as the story of one star, with diagrams and the results.</td>
</tr>
</table>

---

## 🏗️ System architecture

<p align="center"><img src="docs/architecture.svg" alt="System architecture" width="100%"></p>

| Layer | Responsibility | Main code |
|---|---|---|
| **Presentation** | Page routing, the Stitch-designed pages, the Analyzer with its four tabs | `app.py`, `ui/stitch_pages.py`, `ui/stitch/`, `ui/theme.py`, `ui/plots.py`, `ui/explain.py`, `ui/views.py` |
| **Pipeline core** | One star end to end: ingest → preprocess → detect → vetting → triage → router → recycler → report | `src/pipeline.py` and the `src/` packages |
| **Models and data** | Trained classifier with calibration, evaluation summary, six precomputed demo reports | `models/`, `data/examples/` |
| **External archives** | TESS light curves, TIC stellar parameters, TOI list, TESS EB catalog, Gaia DR3 variability | via `lightkurve`, `astroquery`, `requests` (cached locally) |
| **Offline training** | Labels, star selection, feature building, training, demo reports, evaluation | `scripts/` |

The Analyzer works in two modes: **instant** (reads a precomputed report and the saved
evaluation, works offline) and **live** (calls `src.pipeline.analyze_target`, which
runs the whole pipeline on fresh TESS data).

---

## 🔬 The pipeline

<p align="center"><img src="docs/pipeline.svg" alt="Pipeline flowchart" width="100%"></p>

| # | Stage | What happens | Module |
|---|---|---|---|
| 1 | **Collect** | Resolve the star to a TIC ID, list its sectors, download SPOC 2-minute PDCSAP light curves, cache them as Parquet | `src/ingest/` |
| 2 | **Clean** | Quality mask, per-sector median normalisation, upper-side-only 5σ clip, sliding biweight detrend (window ≥ 0.75 d, re-run with dips masked), stitch sectors | `src/preprocess/` |
| 3 | **Search** | Box Least Squares on the flattened, 10-min-binned curve: 0.5–30 days, 12 durations, coarse grid then refinement; SDE ≥ 7; prewhitened copy for fast variables | `src/detect/` |
| 4 | **Measure** | 56 features: trapezoid shape, odd/even, secondary eclipse, per-sector depth, Lomb–Scargle, autocorrelation, Fourier, stellar parameters | `src/vetting/` |
| 5 | **Decide** | Calibrated four-class XGBoost gives a probability for each class | `src/triage/` |
| 6 | **Double-check** | Physics router can overrule the classifier (see below) | `src/recycler/router.py` |
| 7 | **Recycle** | Planet, EB and variable modules characterise the signal; noise goes to a systematics log | `src/recycler/` |

### How the router decides

```mermaid
flowchart TD
    A([Calibrated class probabilities]) --> B{Implied size<br/>&gt; 2.5 Jupiter radii?}
    B -- yes --> NP[Not a planet<br/>next most likely class]
    B -- no --> C{Clear second eclipse<br/>at phase ≈ 0.5, SNR ≥ 7?}
    C -- yes --> EB[Eclipsing binary<br/>→ EB module]
    C -- no --> D{Top probability<br/>below 50 %?}
    D -- yes --> R[Needs review<br/>top two modules shown]
    D -- no --> K[Classifier's answer stands]
    K --> P[Planet module<br/>size · orbit · temperature ± errors]
    K --> E2[EB module<br/>true period P or 2P · eclipses · type]
    K --> V[Variable module<br/>type · period · amplitude vs Gaia]
    K --> N[Noise<br/>systematics log]

    classDef q fill:#0D1424,stroke:#EFBF62,color:#F2F5FA
    classDef planet fill:#0B1120,stroke:#54D8E8,color:#54D8E8
    classDef eb fill:#0B1120,stroke:#EFBF62,color:#EFBF62
    classDef var fill:#0B1120,stroke:#EC4899,color:#EC4899
    classDef rev fill:#0B1120,stroke:#A78BFA,color:#A78BFA
    classDef base fill:#0B1120,stroke:#4D9FFF,color:#F2F5FA
    class B,C,D q
    class P planet
    class EB,E2 eb
    class V var
    class R rev
    class A,K,NP,N base
```

The router **never discards a possible planet**: if no class reaches 50 %, the star is
kept as *Needs review* and the findings of the two most likely modules are shown side
by side.

---

## 🗂️ Dataset

The dataset is **assembled by the pipeline itself** from public NASA and ESA archives,
not taken from a ready-made CSV. This avoids the circular training of the original
prototype, where a star's own features were used as both positive and negative
examples.

| Item | Details |
|---|---|
| Light curves | TESS SPOC 2-minute PDCSAP flux from MAST, via Lightkurve |
| Planet labels | NASA Exoplanet Archive TOI table, dispositions CP and KP, period 0.5–15 d |
| Eclipsing-binary labels | TESS Eclipsing Binary catalog (Prša et al. 2022), period 0.5–15 d |
| Variable-star labels | Gaia DR3 variability classification, cross-matched to TIC by position |
| Noise labels | TOI false alarms (FA) plus random 2-minute targets in none of the lists |
| Stellar parameters | TESS Input Catalog: Teff, radius, log g, Tmag, contamination ratio |
| Size | **100 stars, 25 per class**, brightness-matched; 6 more held out for the demo |
| Domain | Tmag 8–12, two sectors per star (the two closest in time) |
| Split | No fixed split: 5-fold `StratifiedGroupKFold` grouped by TIC ID |

Labels are built by `scripts/build_labels.py`. When a star appears in several catalogs
a fixed priority is applied (confirmed planet > unconfirmed candidate > EB catalog >
Gaia variable > false alarm / random) and every conflict is logged. Unconfirmed
candidates are never used for training; they are kept as demo targets.
`scripts/select_stars.py` then draws 25 stars per class with **matched brightness**, so
no class can be recognised simply by how bright it is.

---

## 📐 Features

The 56 model features (full list in `models/triage_meta.json`) fall into five groups.
Values that grow with the amount of data (raw SNR, number of dips) are replaced by
per-transit or ratio versions, so the model behaves the same with 2 or 10 sectors.

| Group | Examples | What it captures |
|---|---|---|
| **Dip shape** | `depth_ppm`, `duty_cycle`, `ingress_to_total_ratio`, `duration_ratio_expected`, `implied_rp_rearth` | U- versus V-shaped dips; whether the duration and size fit an object orbiting *this* star |
| **Two-star tests** | `odd_even_depth_ratio`, `secondary_to_primary_ratio`, `ellip_amp_ppm`, `refl_amp_ppm` | Alternating eclipse depths, a second eclipse half an orbit later, ellipsoidal and reflection signals |
| **Consistency** | `sector_depth_cv`, `frac_sectors_with_transit`, `red_noise_beta` | Same depth in every sector, correlated noise on the dip timescale |
| **Variability** | `ls_period_1..3`, `ls_power_1..3`, `acf_period`, `acf_peak_height`, `fourier_R21`, `R31`, `phi21`, `phi31`, `skew`, `kurtosis` | Smooth stellar cycles (pulsation, rotation) found by Lomb–Scargle, autocorrelation and Fourier decomposition |
| **Star** | `teff`, `rad`, `logg`, `tmag`, `contratio` | The host star itself, from the TESS Input Catalog |

Missing values are left as `NaN` for XGBoost, which learns a default direction at each
split. The baselines get median imputation plus missing-value indicators. No PCA is
applied, because it would destroy the per-feature explanations.

---

## 🤖 Model and training

Three models were trained on **identical folds** (`StratifiedGroupKFold`, 5 folds,
seed 42). Grouping by TIC ID guarantees that no star appears on both sides of a split,
so every star is predicted exactly once by a model that never saw it.

| Model | Role | Key parameters |
|---|---|---|
| **XGBoost** *(selected)* | Main four-class classifier, native missing-value handling, exact TreeSHAP | `multi:softprob`, 400 trees, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=2`, `reg_lambda=1.0` |
| Random forest | Non-linear baseline | 500 trees, `min_samples_leaf=2`, balanced subsample weights |
| Logistic regression | Linear baseline | `C=0.5`, balanced weights, `StandardScaler`, median imputation + indicators |

- **Conservative hyperparameters** (shallow trees, low learning rate, subsampling, L2):
  a grid search on 100 stars would mostly fit noise in the validation folds.
- **Calibration:** raw XGBoost scores are over-confident, so a per-class sigmoid
  (**Platt scaling**) is fitted on out-of-fold predictions, itself cross-validated.
  "70 % planet" then means roughly 7 in 10 such cases are planets.
- **Saving:** the final model is refitted on all 100 stars and saved as XGBoost JSON
  together with a metadata file holding the feature list, calibration curves,
  thresholds, training date and a hash of the label file.

---

## 📊 Results

### Model comparison (5-fold grouped CV, macro averages)

| Model | Accuracy | Precision | Recall | **F1** | PR-AUC |
|---|:-:|:-:|:-:|:-:|:-:|
| Logistic regression | 0.65 | 0.63 | 0.65 | 0.63 | 0.76 |
| Random forest | 0.71 | 0.76 | 0.71 | 0.71 | 0.83 |
| **XGBoost (calibrated)** | **0.77** | **0.79** | **0.77** | **0.76 ± 0.10** | 0.82 |

### Per-class scores of the selected model (mean ± std over the folds)

| Class | Precision | Recall | F1 | PR-AUC |
|---|:-:|:-:|:-:|:-:|
| 🪐 Planet | 0.81 ± 0.14 | 0.76 ± 0.17 | 0.77 ± 0.11 | 0.89 ± 0.08 |
| 🌗 Eclipsing binary | 0.94 ± 0.13 | 0.76 ± 0.22 | 0.82 ± 0.11 | 0.92 ± 0.07 |
| ✨ Variable star | 0.80 ± 0.12 | 0.80 ± 0.14 | 0.80 ± 0.13 | 0.78 ± 0.14 |
| 〰️ Noise | 0.61 ± 0.18 | 0.76 ± 0.33 | 0.66 ± 0.24 | 0.68 ± 0.20 |

**Key observations**

- **No planet was ever called an eclipsing binary.** The main confusion is between
  shallow planets and noise (6 planets called noise, 4 noise stars called planet).
- With a threshold that lets through only **5 % of eclipsing binaries**, **96 % of real
  planets** are still kept.
- XGBoost's lead comes mainly from the variable-star class (F1 0.80 vs 0.72 and 0.49):
  variability features interact non-linearly with dip features.

### Pipeline and recycler checks

| Check | Result |
|---|---|
| Period recovered by BLS: planets | **25 / 25** at the catalog period |
| Period recovered by BLS: eclipsing binaries | **25 / 25** (15 at the true period, 10 at half) |
| Clear periodicity found: variable stars | **25 / 25** (false-alarm probability < 1 %) |
| EB true period after the P-vs-2P rule | **23 / 25** on unseen binaries, 22 / 25 on training binaries |
| Variable-star type vs Gaia | **18 / 25** (RR Lyrae 4/4, Cepheid 4/4, rotation 6/8) |
| EB configuration (detached / semi-detached / contact) | 31 / 50; every miss is one step too "close" |

### The six demo stars (never used for training)

| Star | True nature | Verdict | What it demonstrates |
|---|---|---|---|
| **HD 209458 b** | Confirmed planet | Planet (98 %) | The textbook case: 3.5248 d (catalog 3.5247 d), radius ≈ 1.35 R<sub>J</sub> |
| **TOI 184.01** | Eclipsing binary | Eclipsing binary | Recycling: the classifier leaned planet; the secondary-eclipse rule corrected it |
| **HIP 36213** | RR Lyrae variable | Variable (70 %) | 0.6284 d, typed RR Lyrae, agrees with Gaia |
| **TOI 476.01** | Unconfirmed candidate | Needs review | Honest uncertainty: planet 47 % vs variable 20 %, both modules shown |
| **TOI 1385.01** | Eclipsing binary | Eclipsing binary | Known limitation: 4.52 d, half the true 9.05 d (equal eclipses) |
| **TOI 862.01** | Candidate on a variable star | Needs review | A small dip buried under the star's own variability |

---

## 💬 Explainable AI

Every verdict in the Results tab comes with a short **"Why?"** list in plain English:

1. **Physics checks first**, because they can overrule the model (for example *"A
   clear second eclipse was found halfway through the orbit, which planets don't
   produce"*).
2. **Then the features that pushed the model most**, from exact **TreeSHAP**
   contributions, turned into sentences such as *"The dips are U-shaped like a
   planet's, not V-shaped like a star's."*
3. **A NASA catalog check**: the verdict and the period are compared with the TOI list,
   the TESS EB catalog or Gaia, and disagreements are shown, not hidden.

Globally, the most influential features are the **autocorrelation peak height**,
**flux skewness**, the **ingress-to-total ratio** (U vs V shape) and the **observed vs
expected duration**. These match the physical intuition that shape and timing
separate planets from stars, which suggests the model has not learned a dataset
artefact.

---

## 🧰 Technology stack

| Area | Tools |
|---|---|
| Language | Python 3.13 (all versions pinned in `requirements.txt`) |
| Astronomy | Astropy (BLS, Lomb–Scargle), Lightkurve, Astroquery, Wotan |
| Data | NumPy, Pandas, PyArrow (Parquet), SciPy |
| Machine learning | XGBoost, scikit-learn, SHAP, Joblib |
| Visualisation | Plotly (interactive), Matplotlib (offline evaluation plots) |
| Web app | Streamlit 1.56, HTML/Tailwind pages designed in Google Stitch |
| Quality | pytest, Ruff |
| DevOps | Git, GitHub, GitHub Actions, Docker, Streamlit Community Cloud |

---

## 📁 Project structure

```
AIML_Exoplanet/
├── app.py                  Streamlit entry point: page routing + the Analyzer
├── config.py               all settings in one typed dataclass
├── src/
│   ├── ingest/             star lookup, TESS download + cache, catalogs
│   ├── preprocess/         cleaning, detrending, stitching sectors
│   ├── detect/             BLS search (optional TLS cross-check)
│   ├── vetting/            transit, variability and per-sector features
│   ├── triage/             labels, feature runner, training, model
│   ├── recycler/           router + planet / EB / variable / noise modules
│   ├── pipeline.py         one star end to end (analyze_target)
│   └── demo_cache.py       precomputed demo reports
├── ui/
│   ├── theme.py            Analyzer look (CSS), verdict card, star card
│   ├── plots.py            Plotly figures
│   ├── views.py            plot views (folds, eclipses)
│   ├── explain.py          plain-English reasons from routing + SHAP
│   ├── stitch_pages.py     serves the Stitch pages, fills Demo Stars with real data
│   └── stitch/             Home, Learn, Demo Stars, How It Works (HTML)
├── static/stitch/          page images (credits in CREDITS.md)
├── scripts/                build labels / features / model / examples / evaluation, screenshots
├── models/                 triage_xgb.json, triage_meta.json, model_evaluation.json
├── data/examples/          the six precomputed demo reports
├── tests/                  offline test suite (synthetic light curves + app smoke tests)
├── docs/                   model card, architecture notes, diagrams, screenshots
├── Dockerfile · docker-compose.yml · .dockerignore
└── .github/workflows/ci.yml
```

---

## 🚀 Getting started

### 1. Local run

Requires **Python 3.12 or 3.13**.

<details open>
<summary><b>Linux / macOS</b></summary>

```bash
git clone https://github.com/AIML-Miniproj/AIML_Exoplanet.git
cd AIML_Exoplanet
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
</details>

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
git clone https://github.com/AIML-Miniproj/AIML_Exoplanet.git
cd AIML_Exoplanet
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
</details>

Open <http://localhost:8501>. The app starts on **Home**; go to **Analyzer** (or click
a card on **Demo Stars**), pick a star and press **Analyze**. Tick *Re-run live
analysis* to download fresh TESS data (needs internet). The trained model and its
evaluation are committed in `models/`, so everything works straight after cloning.

### 2. Run with Docker

```bash
docker build -t exoplanet-atlas .
docker run -p 8501:8501 exoplanet-atlas
# or
docker compose up --build
```

The image (`python:3.13-slim`) contains the pinned dependencies, the trained model and
the demo reports, runs as a non-root user and has a health check; Compose keeps the
TESS download cache in a named volume.

### 3. Tests and linting

```bash
python -m pytest -q             # offline suite; network tests are deselected by default
python -m pytest -m network     # the few tests that query NASA's archives
ruff check .
```

---

## 🧪 Rebuilding everything from scratch

Downloaded data (`data/`, except `data/examples/`) and evaluation plots (`outputs/`)
are not in git; the trained model (`models/triage_xgb.json`, `models/triage_meta.json`,
which holds the calibration) and `models/model_evaluation.json` are. To rebuild
everything, run these in order. Times are for a laptop with 2 worker processes; scripts
that would run longer than 10 minutes stop and ask for `--confirm-long`.

| Step | Command | Time |
|---|---|---|
| 1. Labels from public catalogs (TOI list, TESS EB catalog, Gaia variables) | `python scripts/build_labels.py` | a few minutes the first time (catalog downloads, then cached) |
| 2. Pick 25 stars per class | `python scripts/select_stars.py` | under a minute |
| 3. Run the pipeline on every training star | `python scripts/build_features.py` | ~8 min with light curves cached; over an hour when every light curve must be downloaded |
| 4. Train and cross-validate | `python scripts/train_triage.py` | about a minute |
| 5. Precomputed demo reports | `python scripts/build_examples.py` | ~3 min |
| 6. Evaluation summary for the app (`models/model_evaluation.json`) | `python scripts/save_evaluation.py` | seconds |
| 7. *(optional)* README screenshots | `python scripts/take_screenshots.py` | ~1 min (needs Playwright) |

Optional checks: `scripts/eval_eb_period.py` (EB true-period accuracy),
`scripts/eval_eb_module.py`, `scripts/eval_recycler_rules.py`,
`scripts/check_verdicts.py`. The fresh EB set used by the period check comes from
`select_stars.py` and is built with
`python scripts/build_features.py --labels data/labels/fresh_eb_set.parquet --feature-set fresh_eb`.
All tunable settings (worker count, per-star timeout, period range, thresholds) live
in `config.py`.

---

## ⚙️ MLOps, testing and CI

| Practice | How it is done here |
|---|---|
| **Code versioning** | Git + GitHub; work arrives through pull requests into `main` |
| **Data versioning** | Raw data is not in git, but every model stores a hash of the label file it was trained on; the dataset can be rebuilt deterministically with the numbered scripts and a fixed seed |
| **Experiment tracking** | File-based: `models/triage_meta.json` (parameters, per-fold metrics, calibration, thresholds, top SHAP features) and `models/model_evaluation.json` |
| **Model versioning** | Model and metadata are committed together, so each git revision pins one exact model |
| **Automated testing** | 148 offline tests on synthetic light curves (preprocessing, detection, vetting, labels, triage, recycler, pipeline) plus app smoke tests for every page and demo star |
| **Continuous integration** | GitHub Actions runs Ruff and the test suite on **Ubuntu and Windows** for every push and pull request |
| **Containerisation** | `Dockerfile` + `docker-compose.yml` |
| **Monitoring** | Noise and needs-review signals are appended to a systematics log for later inspection |

---

## ☁️ Deployment

The app is set up for **Streamlit Community Cloud**, deployed straight from this repository:

| Setting | Value |
|---|---|
| Repository | `AIML-Miniproj/AIML_Exoplanet` |
| Branch | `main` |
| Main file | `app.py` |
| Python | 3.12 (Advanced settings) |

Streamlit Cloud installs `requirements.txt`, reads `.streamlit/config.toml` (dark
theme, static file serving for the page images) and redeploys automatically on every
push to `main`. For an organisation-owned repository, an organisation owner must first
grant the Streamlit OAuth app access (GitHub → Settings → Applications → Streamlit →
*Grant*).

---

## ⚠️ Limitations

- **Small dataset.** 100 stars show that the method works, not that it can be relied
  on blindly; scores could shift by about ±0.1 on new stars.
- **Brightness only.** A fainter eclipsing binary blended with the target on the sky
  cannot be ruled out; that needs pixel-level or ground-based follow-up.
- **Equal-eclipse binaries.** When both eclipses look alike, the period comes out at
  half the true value (TOI 1385.01).
- **Starspot periods.** Spots evolve within weeks, so a rotation period can come out at
  a multiple of the true one.
- **Noise vs variable.** These classes overlap most; noise has the lowest F1, partly
  because Gaia screening was only possible for brighter stars.
- **Training domain.** 2-minute data, Tmag 8–12, periods up to 15 days. Outside this
  range, treat results with care.

Within these limits the system is suitable as a **first-pass triage and teaching
tool**, not as a replacement for expert vetting.

---

## 🔮 Future scope

- **More data:** scale from 100 to thousands of labelled stars with the existing label
  scripts, allowing a true held-out test set.
- **Deep learning:** add a 1-D CNN on global and local phase-folded views and ensemble
  it with the feature model.
- **Pixel-level vetting:** centroid shifts from target pixel files to rule out blended
  binaries.
- **Better recyclers:** eclipse-timing variations for third bodies, smarter
  equal-eclipse handling, a trained variable-star sub-classifier.
- **Wider domain:** 200-second and 30-minute full-frame-image light curves, fainter
  stars, longer periods.
- **MLOps:** MLflow tracking and registry, DVC data versioning, scheduled retraining
  as new TESS sectors and TOI dispositions arrive.
- **Serving:** a REST API (e.g. FastAPI), batch processing of whole sectors and drift
  monitoring.

---

## 📚 References

1. G. R. Ricker et al., "Transiting Exoplanet Survey Satellite," *J. Astron. Telesc. Instrum. Syst.*, 1(1), 014003, 2015.
2. G. Kovács, S. Zucker and T. Mazeh, "A box-fitting algorithm in the search for periodic transits," *A&A*, 391, 369–377, 2002.
3. S. D. McCauliff et al., "Automatic classification of Kepler planetary transit candidates," *ApJ*, 806(1), 6, 2015.
4. C. J. Shallue and A. Vanderburg, "Identifying exoplanets with deep learning," *AJ*, 155(2), 94, 2018.
5. M. Ansdell et al., "Scientific domain knowledge improves exoplanet transit classification with deep learning," *ApJL*, 869(1), L7, 2018.
6. L. Yu et al., "Identifying exoplanets with deep learning. III. Automated triage and vetting of TESS candidates," *AJ*, 158(1), 25, 2019.
7. H. P. Osborn et al., "Rapid classification of TESS planet candidates with convolutional neural networks," *A&A*, 633, A53, 2020.
8. D. J. Armstrong, J. Gamper and T. Damoulas, "Exoplanet validation with machine learning: 50 new validated Kepler planets," *MNRAS*, 504(4), 5327–5354, 2021.
9. M. Hippke et al., "Wōtan: Comprehensive time-series detrending in Python," *AJ*, 158(4), 143, 2019.
10. N. R. Lomb, *Ap&SS*, 39, 447–462, 1976; J. D. Scargle, *ApJ*, 263, 835–853, 1982.
11. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 785–794, 2016.
12. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS 30*, 4765–4774, 2017.
13. J. Platt, "Probabilistic outputs for support vector machines…," *Advances in Large Margin Classifiers*, MIT Press, 1999.
14. A. Prša et al., "TESS eclipsing binary stars. I.," *ApJS*, 258(1), 16, 2022.
15. L. Eyer et al., "Gaia Data Release 3: Summary of the variability processing and analysis," *A&A*, 674, A13, 2023.

The full reference list is in the project report; model details are in
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and design notes in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 🙏 Credits and acknowledgements

- **Data:** NASA TESS mission, the Mikulski Archive for Space Telescopes (MAST), the
  NASA Exoplanet Archive (TOI table), the TESS Eclipsing Binary catalog and ESA Gaia
  DR3.
- **Images:** NASA Image and Video Library (V838 Monocerotis light echo:
  NASA/ESA/H. E. Bond; PIA26601: NASA/JPL-Caltech; TESS launch: NASA/Tony Gray) and
  images from the Google Stitch export. Full credits in
  [`static/stitch/CREDITS.md`](static/stitch/CREDITS.md).
- **Page design:** "Exoplanet Atlas" interface designed in Google Stitch.

**Mini project:** Third Year B.Tech (Computer Engineering), MKSSS's Cummins College of
Engineering for Women, Pune · Academic Year 2026–27.
**Contributors:** [@mahi-mahalle](https://github.com/mahi-mahalle) ·
[@Dkhatke](https://github.com/Dkhatke)

<details>
<summary><b>Cite this project</b></summary>

```bibtex
@misc{exoplanet_atlas_2026,
  title  = {Exoplanet Detection and False-Positive Recycling in TESS Light Curves
            using XGBoost and Physics-Based Vetting},
  author = {Mahalle, Mahi and contributors},
  year   = {2026},
  note   = {Mini project, Cummins College of Engineering for Women, Pune},
  url    = {https://github.com/AIML-Miniproj/AIML_Exoplanet}
}
```
</details>

<div align="center"><sub>Built with real NASA data · explained in plain language · honest about its limits</sub></div>
