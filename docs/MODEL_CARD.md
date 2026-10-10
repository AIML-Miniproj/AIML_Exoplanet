# Model card: triage classifier

<!-- METRICS_START -->
## At a glance

- **A small, academic-scale model.** It was trained on **100 stars: 25 planets,
  25 eclipsing binaries, 25 variable stars and 25 "noise" stars**. Every number
  below comes from that small set, so it carries large uncertainty (roughly
  +/- 10-15 percentage points), and differences between models of a few points
  mean nothing.
- **How it was tested.** No star is ever scored by a model that saw it during
  training. The 100 stars are split into 5 groups (each group keeps the class
  balance, and a star is never split across groups); each group is scored by a
  model trained on the other 4. Results are the average over the 5 groups, with
  the spread between groups shown as +/-. There is no separate test set: at
  this size, holding out 20 stars would leave too few to learn from.
- **Training data.** 2 sectors per star (the two closest together in time),
  SPOC 2-minute light curves only, star brightness Tmag 8-12.

## How good is it?

5-fold cross-validation, grouped by star (mean +/- spread across folds):

| Model | Overall score (macro-F1) | Accuracy | Planet F1 | EB F1 | Variable F1 | Noise F1 |
|---|---|---|---|---|---|---|
| **XGBoost (used by the app, calibrated)** | 0.67 +/- 0.14 | 0.69 +/- 0.11 | 0.84 +/- 0.11 | 0.75 +/- 0.15 | 0.60 +/- 0.12 | 0.51 +/- 0.38 |
| Logistic regression (baseline) | 0.65 +/- 0.09 | 0.66 +/- 0.08 | 0.78 +/- 0.13 | 0.68 +/- 0.11 | 0.57 +/- 0.11 | 0.58 +/- 0.23 |
| Random forest (baseline) | 0.67 +/- 0.10 | 0.68 +/- 0.10 | 0.78 +/- 0.14 | 0.71 +/- 0.16 | 0.60 +/- 0.13 | 0.59 +/- 0.23 |

All three models perform about the same; XGBoost is kept because it is best on
planets and eclipsing binaries, the two classes the app is mainly about.

**In plain words** (every star scored once by a model that had not seen it):

| Truly a... | called Planet | called EB | called Variable | called Noise |
|---|---|---|---|---|
| Planet (25) | **22** | 0 | 0 | 3 |
| Eclipsing binary (25) | 1 | **18** | 5 | 1 |
| Variable star (25) | 0 | 4 | **15** | 6 |
| Noise (25) | 5 | 1 | 5 | **14** |

- Out of 25 real planets, it correctly flags **22** and never mistakes one for an eclipsing binary.
- When the planet threshold is set so that only about **1 in 20 eclipsing binaries** slips through as "planet", it still keeps about **92%** of real planets (+/- 11%).
- Variable stars and noise are the weakest classes: they are often confused with each other, because a quiet variable star and a noisy flat star can look alike over two sectors.
- Probabilities are calibrated (a sigmoid curve per class, itself checked with the same 5 groups), so "70% planet" means roughly 7 in 10 such cases are planets, within the limits of 100 stars.

Plots: `outputs/triage_confusion.png`, `outputs/triage_reliability.png`,
`outputs/triage_shap_summary.png`.

## Data sources (public catalogs, names only)

| Class | Source | Rule |
|---|---|---|
| Planet | NASA Exoplanet Archive TESS Objects of Interest (TOI) table | confirmed or known planets (CP, KP), period 0.5-15 days |
| Eclipsing binary | TESS eclipsing binary catalog | all entries with period 0.5-15 days |
| Variable star | Gaia DR3 variability classification | short-timescale classes (rotation/spots, delta Scuti/gamma Doradus, RR Lyrae, Cepheid, other); long-period/irregular variables excluded; not in the TOI list or the EB catalog |
| Noise | TOI false alarms (FA) + random 2-minute targets in none of the lists above | |

Unconfirmed TOI candidates (PC/APC) are never used for training. Stars whose
catalog entry is flagged as split, duplicate or artifact are excluded.

## How well does the pipeline find the right period? (recovery)

The period the pipeline finds on its own, compared with the catalog value:

| Class | Stars | Period recovered | Notes |
|---|---|---|---|
| Planet | 25 | **25 / 25** at the catalog period | |
| Eclipsing binary | 25 | **25 / 25** | 15 at the true period, 10 at half of it (two similar eclipses look like one dip repeating twice as often) |
| Variable star | 25 | **25 / 25** show a clear repeating pattern (false-alarm chance below 1%) | only 1 has a catalog period to compare; it does not match (see limitations) |
| Noise | 25 | no catalog period | 2 / 25 still pass the dip-detection threshold |

## Limitations (read these)

- **Small dataset.** 100 stars is enough to show the method works, not to rely
  on it. Expect real-world performance to differ by more than the +/- above.
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
- **Eclipsing-binary true period.** The rule that decides between the detected
  period and twice it is right for 15 of the 25 training binaries. It fails
  mostly when no second eclipse is visible: it then assumes two equal eclipses
  (doubles the period), but most such binaries in the sample were already found
  at their true period. Treat a binary's period as uncertain by a factor of 2
  when the report flags it as ambiguous.
- **Trained on 2-minute data from 2 sectors, Tmag 8-12.** Fainter stars, other
  cadences, or very long periods (over 15 days) are outside what it has seen.
  Features that grow with the amount of data are not used by the model, so runs
  with more sectors should give consistent answers (checked below).
- **Stars with strong, fast brightness changes can hide a planet.** If a star's
  own variability repeats on about the same timescale as the cleaning window
  (about a day), part of it survives the cleaning and the dip search locks onto
  it instead of a shallow transit (see TOI 862.01 below).

## Checks on real stars

**Same verdict with more or less data?**

| Star | Sectors used | Verdict | Period found |
|---|---|---|---|
| HD 209458 (planet) | 1 | Planet (Medium, 75%) | 3.5245 d |
| HD 209458 (planet) | 2 (all TESS has) | Planet (High, 97%) | 3.5248 d |
| TOI 862.01 (candidate) | 2 | Variable (Medium, 78%) | 1.109 d |
| TOI 862.01 (candidate) | 10 | Noise (Medium, 61%) | 1.109 d |

HD 209458 keeps its verdict; more data raises the confidence, as it should.
TOI 862.01 does **not**: in both runs the search missed the catalog transit
(3.31 days, 0.5% deep) and found the star's own variability instead (a
repeating pattern near 1.1-1.2 days, too weak to count as a detection), and
the verdict on that non-transit signal changed with the amount of data.

**The 4 demo stars** (the app's standard settings, up to 10 sectors):

| Star | What it is | Verdict | Period found | Catalog period |
|---|---|---|---|---|
| HD 209458 b | Confirmed planet | **Planet** (High, 97%) | 3.5248 d | 3.5247 d |
| TOI 1385.01 | Candidate later shown to be an eclipsing binary | **Eclipsing binary** (High, 80%) | 4.523 d (true orbit 9.046 d after the binary check) | 9.0457 d |
| HIP 36213 | RR Lyrae pulsating star | **Eclipsing binary** (Medium, 51%), Variable second (32%): **wrong** | 0.6284 d | 0.6284 d |
| TOI 862.01 | Unconfirmed candidate | **Noise** (Medium, 61%) | 1.109 d | 3.310 d |

HIP 36213 is a real mistake: the training set holds only 3 large-amplitude
pulsating stars, and a pulsation whose dimming repeats exactly with the
detected period looks binary-like to the model. It is kept in the demo and
shown as a tricky case.
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
