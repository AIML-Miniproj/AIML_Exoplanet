# Model card: triage classifier

<!-- METRICS_START -->
## At a glance

- **An academic-scale model.** It was trained on **400 stars: 100 planets,
  100 eclipsing binaries, 100 variable stars and 100 "noise" stars** (run 5; the
  first version used 25 per class). Every number below comes from that set, so
  it still carries some uncertainty (roughly +/- 5 percentage points), and
  differences between models of a few points mean little.
- **How it was tested.** No star is ever scored by a model that saw it during
  training. The 400 stars are split into 5 groups (each group keeps the class
  balance, and a star is never split across groups); each group is scored by a
  model trained on the other 4. Results are the average over the 5 groups, with
  the spread between groups shown as +/-. There is no separate test set yet;
  with 400 stars one is now feasible (see the README's next steps).
- **Training data.** 2 sectors per star (the two closest together in time),
  SPOC 2-minute light curves only, star brightness Tmag 8-12.

## How good is it?

5-fold cross-validation, grouped by star (mean +/- spread across folds):

| Model | Overall score (macro-F1) | Accuracy | Planet F1 | EB F1 | Variable F1 | Noise F1 |
|---|---|---|---|---|---|---|
| **XGBoost (used by the app, calibrated)** | 0.87 +/- 0.05 | 0.87 +/- 0.05 | 0.87 +/- 0.04 | 0.94 +/- 0.04 | 0.87 +/- 0.07 | 0.81 +/- 0.06 |
| Logistic regression (baseline) | 0.79 +/- 0.04 | 0.78 +/- 0.04 | 0.81 +/- 0.04 | 0.88 +/- 0.07 | 0.78 +/- 0.05 | 0.68 +/- 0.05 |
| Random forest (baseline) | 0.84 +/- 0.03 | 0.84 +/- 0.03 | 0.83 +/- 0.03 | 0.92 +/- 0.03 | 0.84 +/- 0.06 | 0.77 +/- 0.02 |
| SVM, RBF kernel (extra) | 0.80 +/- 0.04 | 0.80 +/- 0.04 | 0.81 +/- 0.06 | 0.85 +/- 0.06 | 0.80 +/- 0.04 | 0.72 +/- 0.06 |
| K-nearest neighbours, k = 9 (extra) | 0.71 +/- 0.08 | 0.72 +/- 0.08 | 0.76 +/- 0.08 | 0.79 +/- 0.10 | 0.71 +/- 0.07 | 0.58 +/- 0.13 |
| Ensemble: XGBoost + random forest (calibrated) | 0.87 +/- 0.05 | 0.86 +/- 0.05 | 0.88 +/- 0.03 | 0.94 +/- 0.06 | 0.85 +/- 0.06 | 0.80 +/- 0.06 |
| Ensemble: XGBoost + random forest + logistic regression (calibrated) | 0.84 +/- 0.05 | 0.84 +/- 0.05 | 0.84 +/- 0.04 | 0.91 +/- 0.06 | 0.84 +/- 0.08 | 0.78 +/- 0.05 |

For comparison, the first model (100 stars, 25 per class) scored macro-F1
0.76 +/- 0.10. Going to 400 stars improved every class and halved the spread
between folds; noise improved most (F1 0.66 -> 0.81).

**Which model?** XGBoost scores highest, now clearly above both baselines. The
soft-voting ensembles (the members' probabilities averaged, scored on the same
folds) do not beat it: XGBoost + random forest ties, and adding logistic
regression makes it worse. XGBoost is kept because it is as good as anything
tried and allows exact per-star explanations ("why do we think so?").

**Further experiments** (same folds; scripts write to `outputs/` only):

- *Two more classifier families* (`scripts/compare_classifiers.py`): SVM with an
  RBF kernel (0.80) and k-nearest neighbours (0.71), both with median imputation
  and standardisation. KNN suffers most from the 56 dimensions and calls 32 noise
  stars "planet"; SVM ranks planets well (keeps 91% at 5% binary leakage) but its
  class decisions are weaker.
- *Other ensembles* (`scripts/compare_ensembles.py`): weighted soft voting with
  weights chosen by nested CV 0.864, hard voting 0.852 (3 models) and 0.832 (5),
  seed-bagged XGBoost 0.866. The models share most errors (44 of XGBoost's 52 are
  also random-forest errors), so voting has little to correct.
- *Hyperparameter tuning* (`scripts/tune_xgboost.py`): RandomizedSearchCV over 8
  settings, 40 tries, nested 5 x 3 CV: 0.839 vs 0.871 for the hand-chosen
  settings; the "best" settings differ from fold to fold (fitting noise).
- *Feature selection* (`scripts/select_features.py`): SHAP ranking and recursive
  elimination, k chosen by nested CV: 0.868 vs 0.866 with all 56 features. About
  40 features carry nearly all the information; all are kept.

**In plain words** (every star scored once by a model that had not seen it):

| Truly a... | called Planet | called EB | called Variable | called Noise |
|---|---|---|---|---|
| Planet (100) | **87** | 0 | 4 | 9 |
| Eclipsing binary (100) | 3 | **91** | 4 | 2 |
| Variable star (100) | 1 | 0 | **88** | 11 |
| Noise (100) | 8 | 3 | 7 | **82** |

- Out of 100 real planets, it flags **87** and never mistakes one for an eclipsing binary; 9 are called "noise" (mostly shallow transits) and 4 "variable".
- When the planet threshold is set so that only about **1 in 20 eclipsing binaries** slips through as "planet", it keeps about **88%** of real planets (+/- 16% between folds; 91% pooled over all 400 stars).
- Noise remains the weakest class; shallow planets vs noise and variables vs noise are the most common mix-ups.
- Probabilities are calibrated (a sigmoid curve per class, itself checked with the same 5 groups), so "70% planet" means roughly 7 in 10 such cases are planets.

Plots: `outputs/triage_confusion.png`, `outputs/triage_reliability.png`,
`outputs/triage_shap_summary.png`.

## Data sources (public catalogs, names only)

| Class | Source | Rule |
|---|---|---|
| Planet | NASA Exoplanet Archive TESS Objects of Interest (TOI) table | confirmed or known planets (CP, KP), period 0.5-15 days: 64 CP + 36 KP |
| Eclipsing binary | TESS eclipsing binary catalog | all entries with period 0.5-15 days: 100 |
| Variable star | Gaia DR3 variability classification | short-timescale classes: 31 rotating/spotted, 20 delta Scuti/gamma Doradus, 16 RR Lyrae, 18 Cepheids, 15 other; long-period/irregular variables excluded; not in the TOI list or the EB catalog |
| Noise | TOI false alarms (FA) + random 2-minute targets in none of the lists above | 39 FA + 61 random |

The 400 were drawn from a candidate list capped at 400 per class (built from
the catalogs in October 2026; TESS sectors 1-107 with SPOC 2-minute data).
Eligible stars before that cap: 948 planets, 3,268 binaries, 7,645 variables.

Unconfirmed TOI candidates (PC/APC) are never used for training. Stars whose
catalog entry is flagged as split, duplicate or artifact are excluded.

## How well does the pipeline find the right period? (recovery)

The period the pipeline finds on its own, compared with the catalog value:

| Class | Stars | Period recovered | Notes |
|---|---|---|---|
| Planet | 100 | **89 / 100** (88 at the catalog period, 1 at half) | the larger sample includes shallower planets than the first 25 (which were all recovered) |
| Eclipsing binary | 100 | **97 / 100** | 73 at the true period, 24 at half of it (two similar eclipses look like one dip repeating twice as often); see the orbital-period check below |
| Variable star | 100 | **100 / 100** show a clear repeating pattern (false-alarm chance below 1%) | only 4 have a catalog period to compare; 2 match |
| Noise | 100 | no catalog period | 15 / 100 still pass the dip-detection threshold |

## Limitations (read these)

- **Academic-scale dataset.** 400 stars give much steadier numbers than the
  first 100, but real-world performance can still differ from the +/- above,
  and there is no separate held-out test set yet.
- **Light only.** It looks at brightness measurements only. **It cannot rule out
  that a dip comes from a fainter eclipsing binary close to the target on the
  sky**; that needs pixel-level or ground-based follow-up.
- **Weak check on "noise" stars.** Random noise stars were checked against the
  Gaia list of known variables only for stars brighter than Gaia G = 11. Fainter
  variable stars may be among the "noise" examples, which makes noise and
  variable harder to separate.
- **Starspot periods.** Spots grow and fade over a few weeks, so the shape of a
  spotted star's light curve changes between rotations. Over two sectors the
  measured period can come out at a multiple of the true rotation period. The one
  training rotator with a catalog period (3.41 days) was measured at 6.38 days.
- **Eclipsing-binary orbital period.** It is decided by a simple rule
  (alternating dips differ -> double; a second eclipse is visible -> keep;
  neither -> keep, flagged "ambiguous"). On 25 binaries never used for training
  or design it was right for **23 of 25** with up to 10 sectors (earlier binary
  set); with only 2 sectors it is right for 16 of 25 new fresh binaries. Treat
  a binary flagged "ambiguous" as possibly twice as long.
- **Trained on 2-minute data from 2 sectors, Tmag 8-12.** Fainter stars, other
  cadences, or very long periods (over 15 days) are outside what it has seen.
  Features that grow with the amount of data are not used by the model, so runs
  with more sectors should give consistent answers (checked below).
- **Stars with strong, fast brightness changes can hide a planet.** If a star's
  own variability repeats on about the same timescale as the cleaning window
  (about a day), part of it survives the cleaning. The dip search therefore also
  tries a copy with up to 3 strong, fast sine waves removed (prewhitening), and
  uses it only when it gives a clearly stronger detection, so a binary whose
  light curve is itself almost a sine wave is not erased. This helps with enough
  data, but not always: TOI 862.01 (below) is found with 10 sectors, not with 2.
- **Shallow planets can be mistaken for noise.** In the latest training run, 9
  of the 100 planets were called "noise" (6 of 25 in the 100-star run). These are mostly
  planets with shallow dips, which look much like the small, irregular dips in
  noisy light curves. The router therefore never sends a weak planet signal
  straight to "noise": if "planet" is among the two most likely answers and the
  model is unsure, the star is marked "needs review" instead.
- **Weak second signals.** With the standard settings the search looks for a
  second signal after removing the first. On HD 209458 it reports a weak one
  (14.7 days, about 260 ppm, just above the detection threshold) that is
  probably not real. Treat second signals near the threshold with caution.

## Checks on real stars

**Same verdict with more or less data?**

| Star | Sectors used | Verdict | Period found |
|---|---|---|---|
| HD 209458 (planet) | 1 | Planet (Medium, 74%) | 3.5245 d |
| HD 209458 (planet) | 2 (all TESS has) | Planet (High, 98%) | 3.5248 d |
| TOI 862.01 (candidate) | 2 | Variable (High, 87%) | 1.109 d (the star's own variability) |
| TOI 862.01 (candidate) | 10 | Variable (Low, 39%; planet 23%) | 3.3102 d (the catalog transit) |

HD 209458 keeps its verdict and becomes more confident with more data.
TOI 862.01 keeps the label "variable", but the evidence underneath changes:
with 2 sectors the transit is not found; with 10 it is (after the star's fast
variability is removed for the search), yet the variability still dominates
the verdict. It is shown in the app as a tricky case.

**Demo stars** (the app's standard settings, up to 10 sectors):

| Star | What it is | Verdict | Period found | Catalog period |
|---|---|---|---|---|
| HD 209458 b | Confirmed planet | **Planet** (High, 99%) | 3.5248 d | 3.5247 d |
| TOI 184.01 | Candidate later shown to be an eclipsing binary | **Eclipsing binary** (High, 88%, from the classifier itself; the 100-star model leaned planet and needed the second-eclipse rule) | 4.8170 d | 4.8171 d |
| HIP 36213 | RR Lyrae pulsating star (not in training) | **Variable** (High, 98%) | 0.6284 d | 0.6284 d |
| TOI 476.01 | Unconfirmed candidate | **Planet** (Medium, 61%; variable 27%) | 3.0630 d | 3.0630 d |
| *Tricky case:* TOI 862.01 | Unconfirmed candidate on a fast-varying star | Needs review (Variable 46%, Noise 26%; a repeating dip found every 3.3 days) | 3.3102 d | 3.3102 d |
| *Tricky case:* TOI 1385.01 | Candidate later shown to be an eclipsing binary | Eclipsing binary (Medium, 79%) | 4.523 d | 9.0457 d: its two eclipses look alike, so the period found is half the true one |

None of the six shows flares. The table "Same verdict with more or less data?"
above is from the 100-star model and has not been rerun.

**Eclipsing-binary orbital period** (binary rule vs catalog, match within 1%).
Measured with up to 10 sectors per star on the *earlier* binary sets (100-star
run); not yet re-measured on the 400-star sets (`scripts/eval_eb_module.py`):

| Set | Correct | Found at true period / at half |
|---|---|---|
| 25 binaries never used for training or design | **23 / 25** | 20 / 4 (1 at another ratio) |
| 25 training binaries (*seen while designing the rule*) | 22 / 25 | 15 / 10 |

Quick check on the new sets with only the 2 training sectors per star
(`scripts/eval_eb_period.py`; fewer sectors, so more binaries are found at half
the period and the rule has less to go on): **16 / 25** on 25 new fresh
binaries, **90 / 100** on the 100 training binaries.

## After the classifier: the recycler modules

The classifier's answer goes to a router and then to one module per class.
These modules use simple, stated rules, not trained models (they were designed
when there were 25 examples per class, too few to learn sub-types). Checked
against catalog labels on the earlier (100-star run) sets:

| Module | What it decides | Agreement with the catalog |
|---|---|---|
| Eclipsing binary | Orbital period (detected or twice it) | 23 / 25 on binaries never used in design; 22 / 25 on the training binaries (see below) |
| Eclipsing binary | Type: detached / semi-detached / contact | 31 / 50 (training + fresh binaries). Every miss is one step too "close": 9 detached called semi-detached, 10 semi-detached called contact; never two steps off |
| Variable star | Type: rotation, delta Scuti / gamma Doradus, RR Lyrae, Cepheid, other | 18 / 25: RR Lyrae 4/4, Cepheid 4/4, rotation 6/8, delta Scuti / gamma Doradus 4/7, other 0/2 |

These checks use the data the app itself would load (up to 10 sectors per star).

**Orbital period of a binary.** Whether a second eclipse is visible is decided
from the module's own measurement (every point of every sector the run loaded,
compared with the brightness just around it), not from the coarser measurement
used by the classifier. Accuracy stayed the same (fresh 23/25, training 22/25),
but far fewer binaries are left as "can't tell which period" (15 -> 3 of 50).

*Tried and dropped:* comparing alternate eclipses (depth, width and shape of odd
vs even eclipses) to decide the period. It did not help: 45 -> 44 correct out of
50 (it acted on any of three tests, so chance differences could trigger it). It
is still measured and saved in the report (JSON, `modules.eb.details`), for
information only.

*Side effect on TOI 1385.01 (demo, tricky case).* Before this change the period
was reported as ambiguous ("4.5 or 9.0 days"); after it, a confident 4.52 d, half
of the catalog's 9.0457 d. The classifier's feature looks for a second dip at
exactly phase 0.5 and finds none there (SNR -0.5), so the rule said "can't tell".
The module's own measurement searches every phase and finds a shallow dip (0.10%
deep, SNR 59) at phase 0.46, which the rule reads as "second eclipse seen: keep
the detected period". The two real eclipses are almost equally deep, so at
4.52 d they fold on top of each other and the alternate-eclipse comparison shows
no clear difference either. The rule was not changed for this star: it is the
equal-eclipse limitation, and the app states it next to the result.

**Binary type** is judged from the module's own measurements at the true period
(fitted eclipse width; smooth light change between eclipses on the light curve
with stellar variability kept). This agrees slightly better with the catalog than
the earlier version (30 -> 32 of 50) but keeps a systematic lean towards "closer"
types: treat "semi-detached" and "contact" as possibly one step too close.

The router never sends a possible planet straight to "noise": when the model is
unsure (top probability below 50%) the star is marked "needs review" and both
likely modules run. Unsure and noise results are logged for later review.
Eclipse-timing analysis (looking for a third body) is not included yet.

## Flares (reported for every star)

Flares are sudden brightenings from magnetic outbursts. They are measured by
`src/recycler/flare_module.py` on the light curve *before* cleaning (the
cleaning step clips them as outliers), reported next to the verdict, and **not
given to the classifier**. A flare needs at least 3 consecutive 2-minute points
clearly above the star's quiet level, an impulsive shape (rise within 20 min and
at most half the decay time, jumping up from the light just before it), must not
sit at a detected eclipse or transit, and must not come back every cycle of the
star's own variation or orbit (which would make it a pulsation or a binary's
bump). Stars whose own brightness swings by more than 10% are reported as
"can't tell" rather than guessed. Energies (equivalent duration x the star's
blackbody brightness in the TESS band) are approximate.

**On the 400 training stars** (2 sectors each), 3 are flare stars: a 3,400 K red
dwarf (TIC 421951793, 5 flares up to +7%, about 10^33 erg) and two K dwarfs
(TIC 156074324, an eclipsing binary, and TIC 93125144). None of the 100 planet
hosts, and none of the RR Lyrae, Cepheid or delta Scuti stars, is called a flare
star (12 of 16 RR Lyrae are reported as pulsation, 8 Cepheids as too variable).
The rules were tightened while looking at these stars (eclipse egress, Cepheid
rises, delta Scuti peaks and a binary's bump were first picked up as flares),
so this is a design set, not an independent test.

**Injection-recovery** (6,208 artificial flares, 4-minute rise, added to the
real light curves of 40 planet hosts; height in units of each light curve's
point-to-point scatter):

| Height | 5-min decay | 15-min decay | 40-min decay |
|---|---|---|---|
| 5x scatter (~0.6%) | 11% | 54% | 68% |
| 8x (~1%) | 54% | 89% | 87% |
| 15x (~1.9%) | 81% | 98% | 96% |
| 30x (~3.8%) | 93% | 98% | 94% |

Short flares are the hardest because a flare must span 3 consecutive points
(which keeps single-point cosmic rays out). Flare *bunching* at the rotation
period is not used against a star: on spotted stars flares can come from
active regions that rotate into view.
<!-- METRICS_END -->

## What the model does

Given the measurements the pipeline makes on a star's light curve, the model
estimates how likely the strongest repeating signal is to be:

| Class | Meaning |
|---|---|
| **PLANET** | A planet passing in front of its star |
| **EB** | An eclipsing binary: two stars passing in front of each other |
| **VARIABLE** | A star whose brightness changes on its own (pulsation, starspots, ...) |
| **NOISE** | Nothing real: instrument effects, scattered light, or random noise |

It only looks at brightness measurements plus a few catalog properties of the
star. **It cannot rule out that the dip comes from a different, fainter eclipsing
binary that happens to sit close to the target on the sky.** Ruling that out needs
pixel-level or ground-based follow-up.

## Features (all computed by the pipeline, never copied from a catalog)

**Display-only features.** Some measurements grow with the amount of data: a
star observed for 10 sectors has more dips, so its signal-to-noise ratios,
likelihood sums and false-alarm probabilities look stronger than the same star
observed for 2. Training uses 2 sectors per star but the app may use up to 10,
so these are shown in reports but **not given to the model** (marked *display
only* below). The model gets versions that do not depend on the amount of data
instead: significance per dip (divided by the square root of the number of dips)
and ratios.

### The dip itself
| Feature | Plain-language meaning |
|---|---|
| `period_d` | Time between dips, in days |
| `duration_hr` | How long each dip lasts (from the search grid), in hours |
| `duty_cycle` | Dip duration divided by the period: the share of each orbit spent in the dip. The search never tries more than 0.2 (a longer "dip" is not a dip) |
| `depth_ppm` | How much light is lost during the dip, in parts per million |
| `depth_snr` | Depth divided by its uncertainty: how clearly the dip stands out (*display only*) |
| `depth_snr_per_transit` | The same, per dip: `depth_snr` divided by the square root of the number of dips |
| `sde` | How much the best period stands out from all other trial periods (*display only*: grows with more dips) |
| `single_event_fraction` | Share of the evidence coming from the single strongest dip; near 1 means one glitch, not a repeating signal (*display only*) |
| `single_event_excess` | That share times the number of dips: 1 when all dips are equally strong, much larger when one dip dominates |
| `n_transits_observed` | Number of dips actually covered by data (*display only*) |
| `odd_even_depth_sigma` | Do alternating dips have different depths? Large values mean the "planet" is really two different stars eclipsing at twice the period. Measured from the scatter of individual dip depths (*display only*) |
| `odd_even_sigma_per_transit` | The same test, per dip |
| `odd_even_depth_ratio` | Shallower of odd/even depth divided by the deeper one (1 = identical) |
| `odd_even_depth_sigma_2p` | The same test when folding at twice the period (*display only*; per-dip version `odd_even_sigma_2p_per_transit`) |
| `secondary_depth_ppm`, `secondary_snr` | Is there a second, smaller dip exactly halfway between the main dips? Eclipsing stars usually show one; planets almost never do at this precision (`secondary_snr` is *display only*; per-dip version `secondary_snr_per_transit`) |
| `secondary_to_primary_ratio` | Depth of that second dip divided by the main dip's depth: near 0 for a planet, large for two stars |
| `secondary_best_phase`, `secondary_best_snr`, `secondary_best_depth_ppm` | The strongest second dip anywhere between 30% and 70% of the way around the orbit (eccentric binaries do not put it at exactly halfway). `secondary_best_snr` is *display only*; the model gets `secondary_best_snr_per_transit` and `secondary_best_to_primary_ratio` |
| `ingress_to_total_ratio` | Shape of the dip from a trapezoid fit: near 0 means a flat bottom (U-shape, typical of planets), 0.5 means a V-shape (typical of grazing stars) |
| `t14_hr` | Total dip duration from the trapezoid fit |
| `trapezoid_chi2_vs_box_chi2` | How much better a sloped-edge trapezoid fits than a sharp box |
| `transit_asymmetry` | Difference between the first and second half of the dip |
| `harmonic_dll` | Whether a dip fits better than a smooth sine wave (compressed to a log scale; *display only*) |
| `harmonic_dll_per_transit` | The same, per dip |
| `implied_rp_rearth` | Size of the object if it were a dark planet, in Earth radii (uses the star's catalog radius) |
| `implied_rp_gt_2rjup` | 1 if that size is above twice Jupiter's, too big for a planet |
| `duration_ratio_expected` | Observed duration divided by the duration expected for a planet around this star at this period (from the star's density). Far from 1 suggests the dip is not on this star or not a planet |
| `red_noise_beta` | How much more the data scatter on the dip's timescale than pure random noise would; describes the noise |

### Across sectors
| Feature | Plain-language meaning |
|---|---|
| `frac_sectors_with_transit` | Share of observed sectors that contain at least one dip |
| `sector_depth_cv` | Spread of the dip depth between sectors, relative to the mean depth |
| `sector_depth_chi2` | Whether depth changes between sectors more than the noise allows; light from a neighbouring star changes with each sector's aperture |
| `max_sector_depth_ratio` | Deepest sector depth divided by the typical one (*display only*: a maximum over more sectors only gets larger) |
| `crowdsap_min`, `crowdsap_range` | Pipeline estimate of how much of the light in the aperture comes from the target (1 = all of it). `crowdsap_range` is *display only* |
| `n_sectors`, `n_sectors_with_transit` | *Display only* (they depend on the run preset) |

### Star variability (from the light curve with variability kept)
| Feature | Plain-language meaning |
|---|---|
| `rms_ppm`, `amp_p2p_ppm` | How much the brightness scatters overall, and the spread between its 5th and 95th percentile |
| `skew`, `kurtosis` | Whether the brightness spends more time dipping or brightening, and how peaked the distribution is |
| `ls_period_1..3`, `ls_power_1..3` | The three strongest smooth, repeating patterns (Lomb-Scargle periodogram) and their strengths |
| `ls_log10_fap` | How unlikely the strongest pattern is to be chance (log scale; *display only*, also used for the variable recovery check) |
| `var_period_d` | The variable star's period after checking the periodogram's period P against 2P: the light curve is folded at 2P and the two halves compared. 2P is kept only if the halves clearly differ (for example two dips of different depth) |
| `var_period_halves_diff` | How different those two halves are, relative to the full brightness range |
| `ls_period_matches_bls` | 1 if the smooth pattern has the same period as the dips (or half/double): typical of close binaries |
| `acf_period`, `acf_peak_height` | Period and strength of repetition from the autocorrelation, good for starspots |
| `fourier_R21`, `fourier_R31`, `fourier_phi21`, `fourier_phi31` | Shape of the main brightness cycle (relative size and timing of its harmonics); pulsating stars have characteristic values |
| `ls_period_sector_std` | How much the main period changes from sector to sector |
| `ellip_amp_ppm`, `ellip_snr` | Brightness change twice per orbit, caused by a star stretched by a close companion (`ellip_snr` is *display only*) |
| `refl_amp_ppm` | Brightness change once per orbit, from light reflected by the companion |

### The star
| Feature | Plain-language meaning |
|---|---|
| `teff` | Surface temperature (K) |
| `rad` | Radius (Sun = 1) |
| `logg` | Surface gravity (log scale): tells dwarfs from giants |
| `tmag` | Brightness in the TESS band (smaller = brighter) |
| `contratio` | Catalog estimate of contaminating light from neighbours |
| `ruwe` | Gaia astrometric quality, a hint of unresolved companions (off by default; dropped when missing for every star) |

Missing values (for example a star with no catalog radius) are left empty; the
model handles them natively.
