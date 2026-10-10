"""Small helpers shared by the vetting feature modules."""

from __future__ import annotations

import numpy as np

NAN = float("nan")


def phase_time(t: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Time from the nearest predicted mid-transit, in days (range -P/2..P/2)."""
    return (t - t0 + 0.5 * period) % period - 0.5 * period


def phase(t: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Orbital phase 0..1 with the primary dip at 0."""
    return ((t - t0) / period) % 1.0


def box_depth(flux: np.ndarray, in_mask: np.ndarray, out_mask: np.ndarray,
              beta: float = 1.0) -> tuple[float, float, int]:
    """Depth (baseline minus in-box mean), its error and the in-box count.

    The white-noise error is multiplied by ``beta`` (see ``red_noise_factor``).
    """
    n_in, n_out = int(in_mask.sum()), int(out_mask.sum())
    if n_in < 2 or n_out < 5:
        return NAN, NAN, n_in
    base = float(np.median(flux[out_mask]))
    sd = float(np.std(flux[out_mask]))
    depth = base - float(np.mean(flux[in_mask]))
    err = beta * sd / np.sqrt(n_in) * np.sqrt(1 + n_in / n_out)
    return depth, err, n_in


def red_noise_factor(t: np.ndarray, flux: np.ndarray, width_d: float,
                     mask: np.ndarray | None = None) -> float:
    """How much more the flux scatters on the ``width_d`` timescale than white noise predicts.

    Bins the (masked) flux into ``width_d`` chunks and compares the scatter of the
    chunk means with sigma / sqrt(points per chunk). Returns at least 1.
    """
    keep = np.ones(t.size, bool) if mask is None else mask
    t, f = t[keep], flux[keep]
    if t.size < 50 or width_d <= 0:
        return 1.0
    idx = np.floor((t - t.min()) / width_d).astype(np.int64)
    _, inv, cnt = np.unique(idx, return_inverse=True, return_counts=True)
    means = np.bincount(inv, f) / cnt
    good = cnt >= max(3, 0.5 * np.median(cnt))
    if good.sum() < 10:
        return 1.0
    sigma = np.std(f)
    expected = sigma / np.sqrt(np.median(cnt[good]))
    observed = np.std(means[good])
    return float(max(1.0, observed / expected)) if expected > 0 else 1.0


def per_transit_depths(t: np.ndarray, flux: np.ndarray, period: float, t0: float,
                       duration: float, min_points: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Depth of each individual transit (local baseline minus in-transit mean).

    Returns (epoch numbers, depths) for transits with enough in-transit and
    nearby out-of-transit points.
    """
    epoch = np.round((t - t0) / period).astype(np.int64)
    dt = np.abs(t - (t0 + epoch * period))
    inside = dt < 0.4 * duration
    near = (dt > 0.75 * duration) & (dt < 3 * duration)
    epochs, depths = [], []
    for ep in np.unique(epoch[inside]):
        mi = inside & (epoch == ep)
        mo = near & (epoch == ep)
        if mi.sum() >= min_points and mo.sum() >= 2 * min_points:
            epochs.append(ep)
            depths.append(np.median(flux[mo]) - np.mean(flux[mi]))
    return np.asarray(epochs), np.asarray(depths)


def bin_xy(x: np.ndarray, y: np.ndarray, edges: np.ndarray,
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean y in each x bin; returns centres, means, standard errors and counts."""
    idx = np.digitize(x, edges) - 1
    nb = len(edges) - 1
    ok = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[ok], minlength=nb).astype(float)
    s = np.bincount(idx[ok], y[ok], minlength=nb)
    s2 = np.bincount(idx[ok], y[ok] ** 2, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / cnt
        var = s2 / cnt - mean**2
        err = np.sqrt(np.maximum(var, 0) / np.maximum(cnt - 1, 1))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, mean, err, cnt


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b and np.isfinite(a) and np.isfinite(b) else NAN


def signed_log10(x: float) -> float:
    """sign(x) * log10(1 + |x|): compresses huge likelihood values."""
    return float(np.sign(x) * np.log10(1 + abs(x))) if np.isfinite(x) else NAN
