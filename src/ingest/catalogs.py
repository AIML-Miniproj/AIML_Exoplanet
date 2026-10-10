"""Public catalogs used for training labels. Every download is cached under data/labels/raw.

- TOI list (NASA Exoplanet Archive TAP, table ``toi``)
- TESS eclipsing binary catalog (VizieR J/ApJS/258/16)
- Gaia DR3 variability classifications (``gaiadr3.vari_classifier_result``)
- TESS 2-minute target lists, one CSV per sector
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG, Config

log = logging.getLogger(__name__)

TOI_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
TOI_COLUMNS = ("toi", "tid", "tfopwg_disp", "pl_orbper", "pl_tranmid", "pl_trandep",
               "pl_trandurh", "st_tmag", "sectors")
EB_CATALOG = "J/ApJS/258/16"
EB_COLUMNS = ["TIC", "Tmag", "BJD0", "Per", "Morph", "Dp-pf", "Ds-pf", "Wp-pf", "Sectors"]
TARGET_LIST_URL = "https://tess.mit.edu/public/target_lists/2m/all_targets_S{:03d}_v1.csv"
BULK_SCRIPT_URL = ("https://archive.stsci.edu/missions/tess/download_scripts/sector/"
                   "tesscurl_sector_{}_lc.sh")
BTJD_OFFSET = 2457000.0


def _raw_dir(cfg: Config) -> Path:
    d = cfg.paths.labels / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cached(path: Path, fetch, refresh: bool) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch()
    df.to_parquet(path, index=False)
    log.info("Cached %d rows -> %s", len(df), path.name)
    return df


# ── TOI ──────────────────────────────────────────────────────

def _download_toi() -> pd.DataFrame:
    import requests

    q = f"select {','.join(TOI_COLUMNS)} from toi"
    r = requests.get(TOI_TAP_URL, params={"query": q, "format": "csv"}, timeout=180)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    missing = set(TOI_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(f"TOI table is missing expected columns {missing}")
    return df


def fetch_toi(cfg: Config = CONFIG, refresh: bool = False) -> pd.DataFrame:
    """All TOIs with disposition, period, epoch (converted to BTJD) and depth (ppm)."""
    df = _cached(_raw_dir(cfg) / "toi.parquet", _download_toi, refresh)
    df = df.rename(columns={"tid": "tic_id"})
    df["tic_id"] = df["tic_id"].astype("int64")
    df["tfopwg_disp"] = df["tfopwg_disp"].fillna("").astype(str).str.strip()
    df["epoch_btjd"] = df["pl_tranmid"] - BTJD_OFFSET
    return df


# ── EB catalog ───────────────────────────────────────────────

def _download_eb() -> pd.DataFrame:
    from astroquery.vizier import Vizier

    v = Vizier(row_limit=-1, columns=EB_COLUMNS)
    t = v.get_catalogs(EB_CATALOG)[0]
    df = t.to_pandas()
    missing = set(EB_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(f"EB catalog is missing expected columns {missing}")
    return df


def fetch_eb_catalog(cfg: Config = CONFIG, refresh: bool = False) -> pd.DataFrame:
    """TESS eclipsing binaries: TIC, period, epoch (already BTJD), morphology."""
    df = _cached(_raw_dir(cfg) / "tess_ebs.parquet", _download_eb, refresh)
    df = df.rename(columns={"TIC": "tic_id", "Per": "period", "BJD0": "epoch_btjd",
                            "Morph": "morph"})
    df["tic_id"] = df["tic_id"].astype("int64")
    return df


# ── Gaia DR3 variables ───────────────────────────────────────

GAIA_SYNC_ROW_CAP = 2000   # the archive's synchronous queries return at most this many rows


def _gaia_query(g_lo: float, g_hi: float, ra_lo: float = 0.0, ra_hi: float = 360.0) -> str:
    return f"""
    select v.source_id, v.best_class_name, v.best_class_score,
           g.ra, g.dec, g.pmra, g.pmdec, g.phot_g_mean_mag
    from gaiadr3.vari_classifier_result as v
    join gaiadr3.gaia_source as g on g.source_id = v.source_id
    where g.phot_g_mean_mag >= {g_lo} and g.phot_g_mean_mag < {g_hi}
      and g.ra >= {ra_lo} and g.ra < {ra_hi}
      and v.best_class_name not in ('ECL', 'EP')
    """


def _gaia_bins(gmax: float) -> list[tuple[float, float]]:
    """Starting magnitude bins, narrower where variables are denser (faint end)."""
    edges = [-5.0, 7.0, 8.0, 8.5, 9.0]
    edges += list(np.round(np.arange(9.1, 10.0001, 0.1), 3))
    edges += list(np.round(np.arange(10.04, gmax + 1e-6, 0.04), 3))
    edges = sorted({e for e in edges if e < gmax} | {float(gmax)})
    return list(zip(edges[:-1], edges[1:]))


def _download_gaia_bin(g_lo: float, g_hi: float, slice_dir: Path | None) -> list[pd.DataFrame]:
    """One magnitude bin, split further (in magnitude, then RA) until under the row cap."""
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    frames: list[pd.DataFrame] = []
    stack = [(g_lo, g_hi, 0.0, 360.0)]
    while stack:
        lo, hi, ra_lo, ra_hi = stack.pop()
        f = slice_dir / f"g{lo:.4f}_{hi:.4f}_ra{ra_lo:.2f}_{ra_hi:.2f}.parquet"             if slice_dir else None
        if f is not None and f.exists():
            frames.append(pd.read_parquet(f))
            continue
        df = Gaia.launch_job(_gaia_query(lo, hi, ra_lo, ra_hi)).get_results().to_pandas()
        if len(df) >= GAIA_SYNC_ROW_CAP:
            if hi - lo > 0.005:
                mid = round((lo + hi) / 2, 4)
                stack += [(lo, mid, ra_lo, ra_hi), (mid, hi, ra_lo, ra_hi)]
            else:
                mid = (ra_lo + ra_hi) / 2
                stack += [(lo, hi, ra_lo, mid), (lo, hi, mid, ra_hi)]
            continue
        df.columns = [c.lower() for c in df.columns]
        if f is not None:
            df.to_parquet(f, index=False)
        frames.append(df)
        log.info("Gaia G %.3f-%.3f RA %.0f-%.0f: %d rows", lo, hi, ra_lo, ra_hi, len(df))
    return frames


def _download_gaia(gmax: float, slice_dir: Path | None = None, n_threads: int = 4
                   ) -> pd.DataFrame:
    """Many small synchronous queries, run a few at a time.

    The archive's asynchronous queue can stall for long periods, while
    synchronous queries answer in seconds but are capped at 2000 rows. Each
    finished slice is cached in ``slice_dir`` so an interrupted download resumes.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(n_threads) as ex:
        parts = list(ex.map(lambda b: _download_gaia_bin(b[0], b[1], slice_dir),
                            _gaia_bins(gmax)))
    frames = [f for p in parts for f in p]
    return pd.concat(frames, ignore_index=True).drop_duplicates("source_id")


def fetch_gaia_variables(cfg: Config = CONFIG, gmax: float = 11.0,
                         refresh: bool = False) -> pd.DataFrame:
    """Bright Gaia DR3 variables except eclipsing binaries ('ECL') and transits ('EP')."""
    slices = _raw_dir(cfg) / "gaia_slices"
    slices.mkdir(exist_ok=True)
    return _cached(_raw_dir(cfg) / f"gaia_vari_simple_g{gmax:g}.parquet",
                   lambda: _download_gaia(gmax, slices), refresh)


def _download_gaia_periods(source_ids: list[int]) -> pd.DataFrame:
    """Periods from the class-specific Gaia tables (long-period variables give a frequency)."""
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    frames = []
    tables = (("vari_rrlyrae", "pf"), ("vari_cepheid", "pf"),
              ("vari_rotation_modulation", "best_rotation_period"),
              ("vari_long_period_variable", "1.0/frequency"))
    for i in range(0, len(source_ids), 400):
        ids = ",".join(str(int(x)) for x in source_ids[i:i + 400])
        for table, expr in tables:
            q = (f"select source_id, {expr} as period from gaiadr3.{table} "
                 f"where source_id in ({ids})")
            if "frequency" in expr:
                q += " and frequency > 0"
            res = Gaia.launch_job(q).get_results().to_pandas()
            res.columns = [c.lower() for c in res.columns]
            frames.append(res.assign(table=table))
    out = pd.concat(frames, ignore_index=True) if frames else         pd.DataFrame(columns=["source_id", "period", "table"])
    return out.dropna(subset=["period"]).drop_duplicates("source_id")


def fetch_gaia_periods(source_ids, cfg: Config = CONFIG) -> pd.DataFrame:
    """Gaia periods for the given sources, cached and extended on demand."""
    path = _raw_dir(cfg) / "gaia_periods.parquet"
    have = pd.read_parquet(path) if path.exists() else         pd.DataFrame(columns=["source_id", "period", "table", "queried"])
    asked = set(have["source_id"].astype("int64")) if len(have) else set()
    todo = sorted({int(x) for x in source_ids} - asked)
    if todo:
        got = _download_gaia_periods(todo).assign(queried=True)
        none = pd.DataFrame({"source_id": sorted(set(todo) - set(got["source_id"])),
                             "period": np.nan, "table": "", "queried": True})
        have = pd.concat([have, got, none], ignore_index=True)
        have["source_id"] = have["source_id"].astype("int64")
        have.to_parquet(path, index=False)
    ids = {int(x) for x in source_ids}
    return have[have["source_id"].isin(ids)].reset_index(drop=True)


# ── TESS 2-minute target lists ───────────────────────────────

def _download_target_list(sector: int) -> pd.DataFrame | None:
    import requests

    r = requests.get(TARGET_LIST_URL.format(sector), timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), comment="#")
    df.columns = [c.strip() for c in df.columns]
    return df[["TICID", "Tmag", "RA", "Dec"]]


def fetch_2min_targets(cfg: Config = CONFIG, max_sector: int = 150,
                       refresh: bool = False) -> pd.DataFrame:
    """Every TIC that got 2-minute cadence in any sector, with its sector count."""
    path = _raw_dir(cfg) / "tess_2min_targets.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    sector_dir = _raw_dir(cfg) / "target_lists"
    sector_dir.mkdir(exist_ok=True)
    frames, misses = [], 0
    for s in range(1, max_sector + 1):
        f = sector_dir / f"S{s:03d}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
        else:
            df = _download_target_list(s)
            if df is None:
                misses += 1
                if misses >= 3:
                    break
                continue
            df.to_parquet(f, index=False)
        misses = 0
        frames.append(df.assign(sector=s))
    allt = pd.concat(frames, ignore_index=True)
    out = (allt.groupby("TICID")
           .agg(tmag=("Tmag", "first"), ra=("RA", "first"), dec=("Dec", "first"),
                n_2min_sectors=("sector", "nunique"), first_sector=("sector", "min"),
                last_sector=("sector", "max"))
           .reset_index().rename(columns={"TICID": "tic_id"}))
    out["tic_id"] = out["tic_id"].astype("int64")
    out.to_parquet(path, index=False)
    log.info("2-min targets: %d unique TICs from sectors 1-%d", len(out), allt["sector"].max())
    return out


# ── SPOC 2-minute light-curve file index ─────────────────────

def _download_bulk_script(sector: int) -> pd.DataFrame | None:
    """Parse MAST's per-sector bulk download script into (tic, sector, filename) rows."""
    import re

    import requests

    r = requests.get(BULK_SCRIPT_URL.format(sector), timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    rx = re.compile(r"-o (tess\d+-s(\d{4})-(\d{16})-\d{4}-s_lc\.fits)")
    rows = [(int(m.group(3)), int(m.group(2)), m.group(1)) for m in rx.finditer(r.text)]
    return pd.DataFrame(rows, columns=["tic_id", "sector", "filename"])


def fetch_spoc_lc_index(cfg: Config = CONFIG, max_sector: int = 150,
                        refresh: bool = False) -> pd.DataFrame:
    """Every SPOC 2-minute light-curve file on MAST, from the bulk download scripts.

    Lets the training run skip one archive search per star (slow when many run in
    parallel). Covers SPOC 2-minute data only, which is what v1 training uses.
    """
    path = _raw_dir(cfg) / "spoc_lc_index.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    sdir = _raw_dir(cfg) / "bulk_scripts"
    sdir.mkdir(exist_ok=True)
    frames, misses = [], 0
    for s in range(1, max_sector + 1):
        f = sdir / f"S{s:03d}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
        else:
            df = _download_bulk_script(s)
            if df is None or df.empty:
                misses += 1
                if misses >= 3:
                    break
                continue
            df.to_parquet(f, index=False)
        misses = 0
        frames.append(df)
    idx = pd.concat(frames, ignore_index=True).drop_duplicates(["tic_id", "sector"])
    idx.to_parquet(path, index=False)
    log.info("SPOC 2-min index: %d files, %d stars, sectors %d-%d", len(idx),
             idx["tic_id"].nunique(), idx["sector"].min(), idx["sector"].max())
    return idx
