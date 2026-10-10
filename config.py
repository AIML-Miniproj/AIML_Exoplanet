"""Typed project configuration.

All tunable values live here. Code reads them through the ``CONFIG`` instance
(or a modified copy made with ``dataclasses.replace``), never via literals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Preset:
    """A named bundle of run settings shown to the user as Quick/Standard/Thorough."""

    name: str
    max_sectors: int | None
    use_tls: bool
    max_signals: int
    period_max_d: float | None = None          # None keeps the config value
    search_bin_minutes: float | None = None    # None keeps the config value
    sector_choice: str | None = None           # None keeps the config value


PRESETS: dict[str, Preset] = {
    "quick": Preset("quick", max_sectors=3, use_tls=False, max_signals=1),
    "standard": Preset("standard", max_sectors=10, use_tls=False, max_signals=2),
    "thorough": Preset("thorough", max_sectors=None, use_tls=True, max_signals=3),
}

# Feature-build mode for the training set: small and fast. Not shown in the UI.
TRAINING_PRESET = Preset("training", max_sectors=2, use_tls=False, max_signals=1,
                         period_max_d=15.0, search_bin_minutes=10.0, sector_choice="closest")


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. Everything under ``data/`` except ``examples/`` is gitignored."""

    root: Path = ROOT
    data: Path = ROOT / "data"
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    labels: Path = ROOT / "data" / "labels"
    logs: Path = ROOT / "data" / "logs"
    cache: Path = ROOT / "data" / "cache"
    examples: Path = ROOT / "data" / "examples"
    models: Path = ROOT / "models"
    outputs: Path = ROOT / "outputs"

    def ensure(self) -> None:
        for p in (self.raw, self.processed, self.labels, self.logs,
                  self.cache, self.models, self.outputs):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)

    # ── Target defaults ──────────────────────────────────────
    star_name: str = "HD 209458"
    mission: str = "TESS"

    # ── Ingest ───────────────────────────────────────────────
    author_priority: tuple[str, ...] = ("SPOC", "TESS-SPOC", "QLP")
    spoc_exptime_s: float = 120.0          # never mix 20 s and 120 s within a sector
    max_sectors: int | None = 10
    sectors: tuple[int, ...] | None = None  # explicit sector list overrides max_sectors
    sector_choice: str = "earliest"         # "earliest" or "closest" (neighbouring in time)
    download_retries: int = 3
    download_backoff_s: float = 5.0
    download_timeout_s: int = 300
    query_timeout_s: int = 90               # TIC / product-search calls
    sector_source: str = "search"           # "search" (all authors) or "spoc_index" (training)
    spoc_index_file: str = "spoc_lc_index.parquet"  # under data/labels/raw; batch runs use a subset
    resolver_cache_days: float = 30.0
    sector_list_cache_days: float = 7.0
    use_gaia_ruwe: bool = False

    # ── Preprocessing ────────────────────────────────────────
    clip_sigma_upper: float = 5.0
    clip_window_d: float = 0.02             # running-median window for the outlier clip
    detrend_method: str = "biweight"
    detrend_window_d: float = 0.75
    break_tolerance_d: float = 0.5
    search_bin_minutes: float | None = 10.0

    # ── Detection ────────────────────────────────────────────
    period_min_d: float = 0.5
    period_max_d: float = 30.0
    durations_d: tuple[float, ...] = tuple(np.geomspace(0.02, 0.5, 12).round(5))
    frequency_factor: float = 1.0
    max_grid_size: int = 40_000             # coarser grid + local refinement beyond this
    n_refine_peaks: int = 3                 # strongest coarse peaks re-searched at full resolution
    sde_min: float = 7.0
    max_single_event_fraction: float = 0.8  # one dip carrying >80% of the evidence is not periodic
    n_top_peaks: int = 5
    use_tls: bool = False
    max_signals: int = 1

    # ── Labels / training ────────────────────────────────────
    random_state: int = 42
    limit_per_class: int = 25               # academic scale: ~100 training stars in total
    n_jobs: int = 2                         # parallel workers; clamped to max_n_jobs
    max_n_jobs: int = 3                     # hard ceiling, whatever is requested
    star_timeout_s: float = 120.0           # one star's pipeline run is killed after this
    chunk_size: int = 20                    # stars per progress report
    max_job_minutes: float = 30.0           # longer estimated runs need explicit confirmation
    test_fraction: float = 0.2
    cv_folds: int = 5
    eb_contamination_target: float = 0.05

    # ── Routing / UI ─────────────────────────────────────────
    uncertain_threshold: float = 0.5
    conf_high: float = 0.8
    batch_max_stars: int = 50
    plot_max_points: int = 20_000

    def search_durations(self) -> np.ndarray:
        """Durations actually searched: BLS needs every duration < the shortest period."""
        d = np.asarray(self.durations_d, dtype=float)
        return d[d < self.period_min_d]

    def detrend_window(self) -> float:
        """Detrend window in days: at least 3x the longest searched duration."""
        return max(self.detrend_window_d, 3.0 * float(self.search_durations().max()))

    def with_preset(self, preset: str | Preset) -> Config:
        """Return a copy with a run preset applied."""
        from dataclasses import replace

        if isinstance(preset, str):
            p = TRAINING_PRESET if preset == TRAINING_PRESET.name else PRESETS[preset]
        else:
            p = preset
        out = replace(self, max_sectors=p.max_sectors, use_tls=p.use_tls,
                      max_signals=p.max_signals)
        if p.period_max_d is not None:
            out = replace(out, period_max_d=p.period_max_d)
        if p.search_bin_minutes is not None:
            out = replace(out, search_bin_minutes=p.search_bin_minutes)
        if p.sector_choice is not None:
            out = replace(out, sector_choice=p.sector_choice)
        return out

    def workers(self, requested: int | None = None) -> int:
        """Number of parallel workers to use: never more than ``max_n_jobs``."""
        n = requested if requested is not None else self.n_jobs
        return max(1, min(n, self.max_n_jobs)) if n > 0 else 1


CONFIG = Config()
