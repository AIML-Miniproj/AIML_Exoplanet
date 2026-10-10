# Model card: triage classifier

<!-- METRICS_START -->
*Metrics are filled in after training (see `scripts/train_triage.py`).*
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

### The dip itself
| Feature | Plain-language meaning |
|---|---|
| `period_d` | Time between dips, in days |
| `duration_hr` | How long each dip lasts (from the search grid), in hours |
| `depth_ppm` | How much light is lost during the dip, in parts per million |
| `depth_snr` | Depth divided by its uncertainty: how clearly the dip stands out |
| `sde` | How much the best period stands out from all other trial periods |
| `single_event_fraction` | Share of the evidence coming from the single strongest dip; near 1 means one glitch, not a repeating signal |
| `n_transits_observed` | Number of dips actually covered by data (reported, not used by the model) |
| `odd_even_depth_sigma` | Do alternating dips have different depths? Large values mean the "planet" is really two different stars eclipsing at twice the period. Measured from the scatter of individual dip depths |
| `odd_even_depth_ratio` | Shallower of odd/even depth divided by the deeper one (1 = identical) |
| `odd_even_depth_sigma_2p` | The same test when folding at twice the period |
| `secondary_depth_ppm`, `secondary_snr` | Is there a second, smaller dip exactly halfway between the main dips? Eclipsing stars usually show one; planets almost never do at this precision |
| `secondary_best_phase`, `secondary_best_snr`, `secondary_best_depth_ppm` | The strongest second dip anywhere between 30% and 70% of the way around the orbit (eccentric binaries do not put it at exactly halfway) |
| `ingress_to_total_ratio` | Shape of the dip from a trapezoid fit: near 0 means a flat bottom (U-shape, typical of planets), 0.5 means a V-shape (typical of grazing stars) |
| `t14_hr` | Total dip duration from the trapezoid fit |
| `trapezoid_chi2_vs_box_chi2` | How much better a sloped-edge trapezoid fits than a sharp box |
| `transit_asymmetry` | Difference between the first and second half of the dip |
| `harmonic_dll` | Whether a dip fits better than a smooth sine wave (compressed to a log scale) |
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
| `max_sector_depth_ratio` | Deepest sector depth divided by the typical one |
| `crowdsap_min`, `crowdsap_range` | Pipeline estimate of how much of the light in the aperture comes from the target (1 = all of it) |
| `n_sectors`, `n_sectors_with_transit` | Reported but not used by the model (they depend on the run preset) |

### Star variability (from the light curve with variability kept)
| Feature | Plain-language meaning |
|---|---|
| `rms_ppm`, `amp_p2p_ppm` | How much the brightness scatters overall, and the spread between its 5th and 95th percentile |
| `skew`, `kurtosis` | Whether the brightness spends more time dipping or brightening, and how peaked the distribution is |
| `ls_period_1..3`, `ls_power_1..3` | The three strongest smooth, repeating patterns (Lomb-Scargle periodogram) and their strengths |
| `ls_log10_fap` | How unlikely the strongest pattern is to be chance (log scale) |
| `ls_period_matches_bls` | 1 if the smooth pattern has the same period as the dips (or half/double): typical of close binaries |
| `acf_period`, `acf_peak_height` | Period and strength of repetition from the autocorrelation, good for starspots |
| `fourier_R21`, `fourier_R31`, `fourier_phi21`, `fourier_phi31` | Shape of the main brightness cycle (relative size and timing of its harmonics); pulsating stars have characteristic values |
| `ls_period_sector_std` | How much the main period changes from sector to sector |
| `ellip_amp_ppm`, `ellip_snr` | Brightness change twice per orbit, caused by a star stretched by a close companion |
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
