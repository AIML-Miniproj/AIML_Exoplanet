"""Logging setup shared by the app, the CLI and the scripts."""

from __future__ import annotations

import logging
import sys
import warnings

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once. Safe to call repeatedly."""
    root = logging.getLogger()
    if not any(getattr(h, "_exo_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        handler._exo_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)
    # astroquery installs its own Logger subclass on import; creating its logger
    # first (via getLogger below) would break that, so import it before touching it.
    try:
        import astroquery  # noqa: F401
    except ImportError:
        pass
    # Third-party libraries are noisy at INFO.
    for name in ("astroquery", "lightkurve", "matplotlib", "numba", "urllib3", "shap"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # lightkurve warns about an optional submodule on every import.
    warnings.filterwarnings("ignore", message=".*tpfmodel submodule is not available.*")
