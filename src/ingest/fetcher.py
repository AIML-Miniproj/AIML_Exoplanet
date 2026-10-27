"""Find and download TESS light curves for every available sector of a star.

Policy: one product per sector, chosen by ``cfg.author_priority`` (SPOC 120 s,
then TESS-SPOC full-frame, then QLP). SPOC 20 s products are never used, so a
sector never mixes cadences. Raw FITS files are kept under ``data/raw/tess``
and each sector is also stored as a small ``.npz`` for fast re-reads.
"""

# ── In plain words ───────────────────────────────────────────────────────────
# DATASET: the light curves themselves (the raw measurements).
# What: for one star, finds which TESS sectors have SPOC 2-minute data, picks which
#   to use (training: the 2 sectors closest in time; the app: up to 10), downloads
#   them from NASA's MAST archive and returns them as lightkurve LightCurve objects.
# Where and format:
#     data/raw/tess/tess...-s0001-<TIC>-...-s_lc.fits
#         the original NASA file, one per star per sector (~2 MB each).
#         FITS = the standard astronomy file format: a header plus a table of
#         time, brightness (PDCSAP flux), its error and a quality flag per reading,
#         one reading every 2 minutes (~19,000 per 27-day sector).
#     data/processed/tic_<TIC>/s0001_SPOC_120.npz
#         a compact copy of just the columns we use (NumPy's zipped-array format),
#         so a second run does not need to re-read the big FITS file.
# A corrupt download is deleted and fetched again next time.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.time import Time

from config import CONFIG, Config
from src.errors import DataDownloadError, NoDataError
from src.ingest import cache

log = logging.getLogger(__name__)

MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
_SECTOR_RE = re.compile(r"Sector\s+(\d+)")
_META_KEYS = ("CROWDSAP", "FLFRCSAP", "TESSMAG", "TEFF", "RADIUS", "CAMERA", "CCD")


@dataclass
class SectorInfo:
    """One downloadable light-curve product for one sector."""

    sector: int
    author: str
    exptime_s: float
    filename: str
    data_uri: str
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        return f"s{self.sector:04d}_{self.author}_{int(round(self.exptime_s))}"


# ── Network (replaced in tests) ──────────────────────────────

def _set_query_timeout(seconds: int) -> None:
    from astroquery.mast import conf

    conf.timeout = seconds


@lru_cache(maxsize=1)
def _spoc_index(path: str) -> dict[int, list[tuple[int, str]]]:
    df = pd.read_parquet(path)
    out: dict[int, list[tuple[int, str]]] = {}
    for tic, sec, fn in zip(df["tic_id"], df["sector"], df["filename"]):
        out.setdefault(int(tic), []).append((int(sec), str(fn)))
    return out


def _index_products(tic_id: int, cfg: Config) -> list[dict[str, Any]]:
    """SPOC 2-minute products from the bulk-script index (no archive search)."""
    path = cfg.paths.labels / "raw" / cfg.spoc_index_file
    if not path.exists() and cfg.spoc_index_file == "spoc_lc_index.parquet":
        from src.ingest.catalogs import fetch_spoc_lc_index
        fetch_spoc_lc_index(cfg)
    return [{"sector": sec, "author": "SPOC", "exptime_s": 120.0, "filename": fn,
             "data_uri": f"mast:TESS/product/{fn}", "size_bytes": 0,
             "target_name": str(tic_id)}
            for sec, fn in _spoc_index(str(path)).get(int(tic_id), [])]


def _search_products(tic_id: int) -> list[dict[str, Any]]:
    """All TESS light-curve products for a TIC, as plain dicts."""
    _set_query_timeout(CONFIG.query_timeout_s)
    res = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
    rows = []
    for r in res.table:
        m = _SECTOR_RE.search(str(r["mission"]))
        if not m:
            continue
        rows.append({
            "sector": int(m.group(1)), "author": str(r["author"]),
            "exptime_s": float(r["exptime"]), "filename": str(r["productFilename"]),
            "data_uri": str(r["dataURI"]), "size_bytes": int(r["size"] or 0),
            "target_name": str(r["target_name"]),
        })
    # Cone searches can include neighbours; keep the target's own products.
    own = [r for r in rows if r["target_name"].lstrip("0") == str(tic_id)]
    return own or rows


def _download_file(url: str, dest: Path, timeout_s: float) -> None:
    import requests

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    tmp.replace(dest)


# ── Product selection ────────────────────────────────────────

def choose_products(rows: list[dict[str, Any]], cfg: Config = CONFIG) -> list[SectorInfo]:
    """Pick one product per sector following the author priority."""
    by_sector: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_sector.setdefault(r["sector"], []).append(r)

    chosen = []
    for sector in sorted(by_sector):
        best, best_rank = None, None
        for r in by_sector[sector]:
            if r["author"] not in cfg.author_priority:
                continue
            if r["author"] == "SPOC" and abs(r["exptime_s"] - cfg.spoc_exptime_s) > 1:
                continue   # skip 20 s fast cadence
            rank = (cfg.author_priority.index(r["author"]), r["exptime_s"])
            if best_rank is None or rank < best_rank:
                best, best_rank = r, rank
        if best is not None:
            chosen.append(SectorInfo(**{k: best[k] for k in SectorInfo.__dataclass_fields__}))
    return chosen


def select_sectors(available: list[SectorInfo], max_sectors: int | None = None,
                   sectors: tuple[int, ...] | list[int] | None = None,
                   cfg: Config = CONFIG) -> list[SectorInfo]:
    """Apply an explicit sector list, or keep the best ``max_sectors``.

    ``cfg.sector_choice == "earliest"``: sectors from higher-priority authors win;
    among equals, the earliest sectors win.
    ``"closest"`` (training preset): the neighbouring sectors closest together in
    time, which keeps the period search small and its grid fine.
    """
    if sectors:
        wanted = set(int(s) for s in sectors)
        out = [s for s in available if s.sector in wanted]
        missing = wanted - {s.sector for s in out}
        if missing:
            log.warning("Requested sectors not available: %s", sorted(missing))
        return out
    if max_sectors is None or len(available) <= max_sectors:
        return list(available)
    if cfg.sector_choice == "closest":
        # The run of max_sectors neighbouring sectors spanning the least time
        # (consecutive when possible); ties go to the best author, then the earliest.
        by_sector = sorted(available, key=lambda s: s.sector)
        windows = [by_sector[i:i + max_sectors]
                   for i in range(len(by_sector) - max_sectors + 1)]
        return min(windows, key=lambda w: (w[-1].sector - w[0].sector,
                                           max(cfg.author_priority.index(s.author) for s in w),
                                           w[0].sector))
    ranked = sorted(available, key=lambda s: (cfg.author_priority.index(s.author), s.sector))
    return sorted(ranked[:max_sectors], key=lambda s: s.sector)


def list_available_sectors(tic_id: int, cfg: Config = CONFIG,
                           use_cache: bool = True) -> list[SectorInfo]:
    """Cheap search (no download): the product the policy would use for each sector."""
    if cfg.sector_source == "spoc_index":
        return choose_products(_index_products(tic_id, cfg), cfg)
    path = cfg.paths.cache / "sectors" / f"tic_{tic_id}.json"
    rows = cache.read_json(path, cfg.sector_list_cache_days) if use_cache else None
    if rows is None:
        try:
            rows = _search_products(tic_id)
        except Exception as e:  # noqa: BLE001
            raise DataDownloadError(f"MAST search for TIC {tic_id} failed: {e}") from e
        cache.write_json(path, rows)
    return choose_products(rows, cfg)


# ── Download + read ──────────────────────────────────────────

def _npz_path(tic_id: int, info: SectorInfo, cfg: Config) -> Path:
    return cfg.paths.processed / f"tic_{tic_id}" / f"{info.key}.npz"


def _save_npz(path: Path, lc: lk.LightCurve, info: SectorInfo) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: lc.meta.get(k) for k in _META_KEYS if lc.meta.get(k) is not None}
    meta.update({"SECTOR": info.sector, "AUTHOR": info.author, "EXPTIME": info.exptime_s})
    quality = np.asarray(lc["quality"].value if "quality" in lc.colnames else
                         np.zeros(len(lc)), dtype=np.int32)
    np.savez_compressed(path, time=lc.time.value, flux=np.asarray(lc.flux.value, float),
                        flux_err=np.asarray(lc.flux_err.value, float), quality=quality,
                        meta=json.dumps(meta, default=float))


def _load_npz(path: Path) -> lk.LightCurve:
    with np.load(path) as d:
        lc = lk.LightCurve(time=Time(d["time"], format="btjd", scale="tdb"),
                           flux=d["flux"], flux_err=d["flux_err"], quality=d["quality"])
        lc.meta.update(json.loads(str(d["meta"])))
    return lc


def _read_fits(path: Path, info: SectorInfo) -> lk.LightCurve:
    lc = lk.read(str(path), quality_bitmask="default")
    lc = lc.remove_nans()
    lc.meta.update({"SECTOR": info.sector, "AUTHOR": info.author, "EXPTIME": info.exptime_s})
    return lc


def download_sector(tic_id: int, info: SectorInfo, cfg: Config = CONFIG,
                    downloader: Callable[[str, Path, float], None] | None = None,
                    sleep: Callable[[float], None] = time.sleep) -> lk.LightCurve:
    """Return one sector's light curve, downloading with retries if not cached."""
    npz = _npz_path(tic_id, info, cfg)
    if npz.exists():
        try:
            return _load_npz(npz)
        except Exception as e:  # noqa: BLE001
            log.warning("Corrupt cache %s (%s); re-reading", npz.name, e)

    fits = cfg.paths.raw / "tess" / info.filename
    fits.parent.mkdir(parents=True, exist_ok=True)
    dl = downloader or _download_file
    last_err: Exception | None = None
    for attempt in range(cfg.download_retries):
        if fits.exists():
            break
        try:
            dl(MAST_DOWNLOAD_URL + info.data_uri, fits, cfg.download_timeout_s)
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("Sector %d download failed (attempt %d/%d): %s", info.sector,
                        attempt + 1, cfg.download_retries, e)
            if attempt + 1 < cfg.download_retries:
                sleep(cfg.download_backoff_s * 2**attempt)
    if not fits.exists():
        raise DataDownloadError(f"sector {info.sector}: {last_err}")

    try:
        lc = _read_fits(fits, info)
    except Exception as e:  # noqa: BLE001
        fits.unlink(missing_ok=True)   # corrupt download: force a fresh one next time
        raise DataDownloadError(f"sector {info.sector}: unreadable FITS ({e})") from e
    _save_npz(npz, lc, info)
    return lc


def fetch_lightcurves(tic_id: int, cfg: Config = CONFIG, max_sectors: int | None = -1,
                      sectors: tuple[int, ...] | list[int] | None = None,
                      progress_cb: Callable[[str, float], None] | None = None,
                      downloader: Callable[[str, Path, float], None] | None = None,
                      ) -> tuple[lk.LightCurveCollection, list[SectorInfo]]:
    """Download the selected sectors and return them with their SectorInfo.

    ``max_sectors=-1`` means "use the config value". Sectors that fail are logged
    and skipped; if none succeed a DataDownloadError is raised.
    """
    available = list_available_sectors(tic_id, cfg)
    if not available:
        raise NoDataError(f"TIC {tic_id} has no TESS light curves from {cfg.author_priority}")
    limit = cfg.max_sectors if max_sectors == -1 else max_sectors
    chosen = select_sectors(available, limit, sectors or cfg.sectors, cfg)
    if not chosen:
        raise NoDataError(f"TIC {tic_id}: none of the requested sectors are available")
    log.info("TIC %d: using sectors %s (of %d available)", tic_id,
             [f"{s.sector}:{s.author}" for s in chosen], len(available))

    lcs, used, errors = [], [], []
    for i, info in enumerate(chosen):
        if progress_cb:
            progress_cb(f"Downloading {len(chosen)} sectors of TESS data ({i + 1}/{len(chosen)})",
                        i / len(chosen))
        try:
            lc = download_sector(tic_id, info, cfg, downloader=downloader)
        except DataDownloadError as e:
            log.warning("Skipping sector %d: %s", info.sector, e)
            errors.append(str(e))
            continue
        if len(lc) < 100:
            log.warning("Skipping sector %d: only %d good cadences", info.sector, len(lc))
            continue
        lcs.append(lc)
        used.append(info)
    if not lcs:
        raise DataDownloadError("; ".join(errors) or f"TIC {tic_id}: no usable sectors")
    return lk.LightCurveCollection(lcs), used
