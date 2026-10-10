"""Combine all vetting features into one flat, ordered dictionary."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.detect.bls import Signal
from src.preprocess.preprocessor import Preprocessed
from src.vetting.sector_features import sector_features
from src.vetting.stellar_features import stellar_features
from src.vetting.transit_features import transit_features
from src.vetting.variability_features import variability_features

TRANSIT_FEATURES = [
    "period_d", "duration_hr", "duty_cycle", "depth_ppm", "depth_snr", "sde",
    "single_event_fraction",
    "n_transits_observed",
    "odd_even_depth_sigma", "odd_even_depth_ratio", "odd_even_depth_sigma_2p",
    "secondary_depth_ppm", "secondary_snr", "secondary_best_phase", "secondary_best_snr",
    "secondary_best_depth_ppm", "ingress_to_total_ratio", "t14_hr",
    "trapezoid_chi2_vs_box_chi2", "transit_asymmetry", "harmonic_dll",
    "implied_rp_rearth", "implied_rp_gt_2rjup", "duration_ratio_expected", "red_noise_beta",
    "depth_snr_per_transit", "secondary_snr_per_transit", "secondary_best_snr_per_transit",
    "odd_even_sigma_per_transit", "odd_even_sigma_2p_per_transit", "harmonic_dll_per_transit",
    "single_event_excess", "secondary_to_primary_ratio", "secondary_best_to_primary_ratio",
]
SECTOR_FEATURES = [
    "n_sectors", "n_sectors_with_transit", "frac_sectors_with_transit", "sector_depth_cv",
    "sector_depth_chi2", "max_sector_depth_ratio", "crowdsap_min", "crowdsap_range",
]
VARIABILITY_FEATURES = [
    "rms_ppm", "amp_p2p_ppm", "skew", "kurtosis",
    "ls_period_1", "ls_power_1", "ls_period_2", "ls_power_2", "ls_period_3", "ls_power_3",
    "ls_log10_fap", "ls_period_matches_bls", "var_period_d", "var_period_halves_diff",
    "var_period_halves_chi2", "acf_period", "acf_peak_height",
    "fourier_R21", "fourier_R31", "fourier_phi21", "fourier_phi31", "ls_period_sector_std",
    "ellip_amp_ppm", "ellip_snr", "refl_amp_ppm",
]
STELLAR_FEATURES = ["teff", "rad", "logg", "tmag", "contratio", "ruwe"]

FEATURE_NAMES = TRANSIT_FEATURES + SECTOR_FEATURES + VARIABILITY_FEATURES + STELLAR_FEATURES

# Features that grow with the amount of data (how many sectors the run preset
# downloaded), not with what the star is: counts, signal-to-noise ratios,
# likelihood sums, false-alarm probabilities, and maxima/ranges over sectors.
# They stay in reports for display but are not fed to the classifier, so a model
# trained on 2 sectors gives the same answer on a 10-sector run. The
# "*_per_transit" and "single_event_excess" features replace them.
DATA_AMOUNT_DEPENDENT = {
    "n_sectors", "n_sectors_with_transit", "n_transits_observed",
    "depth_snr", "sde", "single_event_fraction",
    "odd_even_depth_sigma", "odd_even_depth_sigma_2p",
    "secondary_snr", "secondary_best_snr", "harmonic_dll",
    "max_sector_depth_ratio", "crowdsap_range",
    "ls_log10_fap", "ellip_snr", "var_period_halves_chi2",
}
MODEL_FEATURES = [f for f in FEATURE_NAMES if f not in DATA_AMOUNT_DEPENDENT]


def extract_features(pre: Preprocessed, signal: Signal, star=None,
                     search_cadence_d: float | None = None,
                     ) -> tuple[dict[str, float], dict[str, Any]]:
    """All features for one signal, plus plot/diagnostic details."""
    feats: dict[str, float] = {}
    details: dict[str, Any] = {}

    f, d = transit_features(pre.lc_flat, signal, star, search_cadence_d)
    feats.update(f)
    details.update(d)
    f, d = sector_features(pre.lc_flat, signal, pre.stats.get("sectors"),
                           beta=d.get("red_noise_beta", 1.0))
    feats.update(f)
    details.update(d)
    f, d = variability_features(pre.lc_norm, signal)
    feats.update(f)
    details.update(d)
    feats.update(stellar_features(star))

    ordered = {k: float(feats.get(k, np.nan)) if feats.get(k) is not None else np.nan
               for k in FEATURE_NAMES}
    return ordered, details
