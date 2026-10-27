"""Optional Transit Least Squares search (``cfg.use_tls``, off by default).

TLS fits a realistic transit shape (rounded bottom, gradual edges) instead of a
box, which helps for small planets. It is slower and needs the optional
``transitleastsquares`` package.
"""

# ── In plain words ───────────────────────────────────────────────────────────
# OPTIONAL second opinion on the dip search (off by default: cfg.use_tls).
# BLS looks for a flat-bottomed box. A real transit has a rounded bottom (the star's
# edge is darker than its centre, "limb darkening") and sloped edges. TLS (Transit
# Least Squares, Hippke & Heller 2019) searches with that realistic shape, which can
# find smaller planets, but it is much slower and needs an extra package
# (pip install transitleastsquares). Its answer is returned in the same Signal shape
# as BLS and stored in the report for comparison only; it does not change the verdict.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging

import lightkurve as lk
import numpy as np

from config import CONFIG, Config
from src.detect.bls import Signal

log = logging.getLogger(__name__)


# True if the optional package is installed (the app then may run TLS).
def tls_available() -> bool:
    try:
        import transitleastsquares  # noqa: F401
    except ImportError:
        return False
    return True


def search_tls(lc: lk.LightCurve, cfg: Config = CONFIG, star=None) -> Signal:
    """Run TLS and return a Signal in the same shape as the BLS one."""
    try:
        from transitleastsquares import transitleastsquares
    except ImportError as e:
        raise ImportError("TLS is enabled but 'transitleastsquares' is not installed. "
                          "Run: pip install transitleastsquares") from e

    t = np.asarray(lc.time.value, float)
    f = np.asarray(lc.flux.value, float)
    e = np.asarray(lc.flux_err.value, float)
    kwargs = {}
    # Tell TLS the star's size and mass, so its transit shape and expected length
    # match this star.
    if star is not None and np.isfinite(getattr(star, "rad", np.nan)):
        kwargs.update(R_star=star.rad, M_star=star.mass if np.isfinite(star.mass) else 1.0)
    # Same period range as BLS: 0.5 days up to half the data span (two dips needed).
    model = transitleastsquares(t, f, e)
    r = model.power(period_min=cfg.period_min_d,
                    period_max=min(cfg.period_max_d, (t.max() - t.min()) / 2),
                    use_threads=1, show_progress_bar=False, **kwargs)
    # TLS reports the flux at the dip's bottom (e.g. 0.99); depth = 1 - that = 1%.
    depth = 1 - float(r.depth)
    log.info("TLS: P=%.5f d, SDE=%.1f", r.period, r.SDE)
    return Signal(period=float(r.period), t0=float(r.T0), duration=float(r.duration),
                  depth=depth, depth_err=depth / max(float(r.snr), 1e-6),
                  power=float(r.SDE), sde=float(r.SDE), method="tls")
