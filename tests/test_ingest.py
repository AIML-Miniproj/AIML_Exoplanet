"""Resolver and fetcher logic with all network calls replaced."""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from config import CONFIG, Paths
from src.errors import DataDownloadError, NoDataError, TargetNotFoundError
from src.ingest import fetcher, resolver
from src.ingest.fetcher import SectorInfo, choose_products, select_sectors


@pytest.fixture
def cfg(tmp_path):
    paths = Paths(root=tmp_path, data=tmp_path / "data", raw=tmp_path / "raw",
                  processed=tmp_path / "processed", labels=tmp_path / "labels",
                  logs=tmp_path / "logs", cache=tmp_path / "cache",
                  examples=tmp_path / "examples", models=tmp_path / "models",
                  outputs=tmp_path / "outputs")
    return replace(CONFIG, paths=paths, download_backoff_s=0.0)


TIC_ROW = {"ID": "420814525", "ra": 330.79, "dec": 18.88, "Tmag": 7.13, "Teff": 6052.0,
           "e_Teff": 100.6, "rad": 1.187, "e_rad": 0.045, "mass": 1.13, "e_mass": 0.15,
           "logg": 4.34, "rho": 0.676, "contratio": 0.002, "d": 48.3, "GAIA": "1779546757669063552",
           "HIP": "108859", "lumclass": "DWARF", "disposition": None, "dstArcSec": 0.003}


# ── Resolver ─────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [("TIC 420814525", 420814525), ("420814525", 420814525),
                                           ("tic-12", 12), ("HD 209458", None), ("", None)])
def test_parse_tic(text, expected):
    assert resolver.parse_tic(text) == expected


def test_resolve_by_name_and_cache(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(resolver, "_query_tic_by_name",
                        lambda name: calls.append(name) or [TIC_ROW])
    info = resolver.resolve_target("HD 209458", cfg)
    assert info.tic_id == 420814525 and info.common_name == "HD 209458"
    assert info.rad == pytest.approx(1.187) and info.flags == []
    # second call comes from the JSON cache, also under the TIC key
    assert resolver.resolve_target("hd209458", cfg).tic_id == 420814525
    monkeypatch.setattr(resolver, "_query_tic_by_id", lambda tic: pytest.fail("not cached"))
    assert resolver.resolve_target("TIC 420814525", cfg).teff == 6052.0
    assert len(calls) == 1


def test_resolve_flags_missing_values(cfg, monkeypatch):
    row = dict(TIC_ROW, rad=None, rho=None, mass=float("nan"), contratio="--")
    monkeypatch.setattr(resolver, "_query_tic_by_id", lambda tic: [row])
    info = resolver.resolve_target("TIC 1", cfg)
    assert math.isnan(info.rad) and math.isnan(info.rho)
    assert {"missing_rad", "missing_rho", "missing_mass", "missing_contratio"} <= set(info.flags)


def test_resolve_derives_density(cfg, monkeypatch):
    monkeypatch.setattr(resolver, "_query_tic_by_id", lambda tic: [dict(TIC_ROW, rho=None)])
    info = resolver.resolve_target("TIC 2", cfg)
    assert info.rho == pytest.approx(1.13 / 1.187**3)
    assert "rho_derived" in info.flags


def test_resolve_not_found(cfg, monkeypatch):
    monkeypatch.setattr(resolver, "_query_tic_by_id", lambda tic: [])
    with pytest.raises(TargetNotFoundError):
        resolver.resolve_target("TIC 3", cfg)

    def boom(name):
        raise ValueError("Could not resolve 'Notastar' to a sky position.")
    monkeypatch.setattr(resolver, "_query_tic_by_name", boom)
    with pytest.raises(TargetNotFoundError):
        resolver.resolve_target("Notastar", cfg)


def test_resolve_skips_artifact_rows(cfg, monkeypatch):
    rows = [dict(TIC_ROW, ID="1", disposition="ARTIFACT", dstArcSec=0.0),
            dict(TIC_ROW, ID="2", dstArcSec=1.0)]
    monkeypatch.setattr(resolver, "_query_tic_by_name", lambda name: rows)
    assert resolver.resolve_target("Some Star", cfg).tic_id == 2


# ── Product policy ───────────────────────────────────────────

def _row(sector, author, exptime):
    return {"sector": sector, "author": author, "exptime_s": exptime,
            "filename": f"f_{sector}_{author}_{exptime}.fits", "data_uri": "mast:x",
            "size_bytes": 1, "target_name": "1"}


def test_choose_products_policy():
    rows = [_row(1, "SPOC", 20), _row(1, "SPOC", 120), _row(1, "QLP", 1800),
            _row(2, "SPOC", 20), _row(2, "TESS-SPOC", 600), _row(2, "QLP", 600),
            _row(3, "QLP", 200), _row(3, "TARS", 200),
            _row(4, "TARS", 200)]
    chosen = {s.sector: (s.author, s.exptime_s) for s in choose_products(rows)}
    assert chosen == {1: ("SPOC", 120), 2: ("TESS-SPOC", 600), 3: ("QLP", 200)}


def test_select_sectors():
    avail = [SectorInfo(s, a, 120, "f", "u") for s, a in
             [(1, "QLP"), (2, "SPOC"), (5, "SPOC"), (9, "TESS-SPOC"), (12, "SPOC")]]
    assert [s.sector for s in select_sectors(avail, 3)] == [2, 5, 12]
    assert [s.sector for s in select_sectors(avail, None)] == [1, 2, 5, 9, 12]
    assert [s.sector for s in select_sectors(avail, 3, sectors=[1, 9, 99])] == [1, 9]


def test_select_closest_sectors_for_training():
    cfg = CONFIG.with_preset("training")
    assert cfg.sector_choice == "closest"

    def pick(sectors, k=2):
        avail = [SectorInfo(s, "SPOC", 120, "f", "u") for s in sectors]
        return [s.sector for s in select_sectors(avail, k, cfg=cfg)]

    assert pick([11, 91]) == [11, 91]                      # only two: keep both
    assert pick([11, 40, 41, 91]) == [40, 41]              # consecutive beats earliest
    assert pick([3, 30, 57, 58, 84, 85]) == [57, 58]       # ties: earliest consecutive pair
    assert pick([5, 9, 30, 32]) == [30, 32]                # smallest gap when none consecutive
    assert pick([1, 2, 5, 6, 7], k=3) == [5, 6, 7]
    # UI presets keep the old behaviour
    avail = [SectorInfo(s, "SPOC", 120, "f", "u") for s in (11, 40, 41)]
    assert [s.sector for s in select_sectors(avail, 2)] == [11, 40]


# ── Download with retries and caching ────────────────────────

def _fake_fits_writer(collection):
    """Downloader that 'downloads' by remembering which sector was asked for."""
    def dl(url, dest: Path, timeout):
        dest.write_bytes(b"fake")
    return dl


def test_fetch_retries_then_caches(cfg, monkeypatch, planet_factory):
    coll = planet_factory(n_sectors=2)
    rows = [_row(1, "SPOC", 120), _row(2, "SPOC", 120)]
    monkeypatch.setattr(fetcher, "_search_products", lambda tic: rows)
    by_sector = {lc.meta["SECTOR"]: lc for lc in coll}
    monkeypatch.setattr(fetcher, "_read_fits",
                        lambda path, info: by_sector[info.sector])
    attempts = {"n": 0}

    def flaky(url, dest, timeout):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise TimeoutError("MAST timed out")
        dest.write_bytes(b"fake")

    stages = []
    out, used = fetcher.fetch_lightcurves(1, cfg, max_sectors=None, downloader=flaky,
                                          progress_cb=lambda s, f: stages.append(s))
    assert [s.sector for s in used] == [1, 2] and len(out) == 2
    assert attempts["n"] == 4            # 2 failures + 2 successes
    assert "(1/2)" in stages[0]
    # second run: served from the .npz cache, no download and no FITS read
    monkeypatch.setattr(fetcher, "_read_fits", lambda *a: pytest.fail("should use npz"))
    out2, _ = fetcher.fetch_lightcurves(1, cfg, max_sectors=None,
                                        downloader=lambda *a: pytest.fail("cached"))
    np.testing.assert_allclose(out2[0].flux.value, out[0].flux.value)
    assert out2[0].meta["SECTOR"] == 1


def test_fetch_gives_up_after_retries(cfg, monkeypatch):
    monkeypatch.setattr(fetcher, "_search_products", lambda tic: [_row(1, "SPOC", 120)])

    def always_fail(url, dest, timeout):
        raise TimeoutError("down")

    with pytest.raises(DataDownloadError):
        fetcher.fetch_lightcurves(1, cfg, downloader=always_fail)


def test_fetch_no_data(cfg, monkeypatch):
    monkeypatch.setattr(fetcher, "_search_products", lambda tic: [_row(1, "TARS", 200)])
    with pytest.raises(NoDataError):
        fetcher.fetch_lightcurves(1, cfg)


# ── Live checks (opt-in) ─────────────────────────────────────

@pytest.mark.network
def test_live_resolve_and_sectors(cfg):
    info = resolver.resolve_target("HD 209458", cfg)
    assert info.tic_id == 420814525
    sectors = fetcher.list_available_sectors(info.tic_id, cfg)
    assert {56, 82} <= {s.sector for s in sectors}
    assert all(s.author == "SPOC" and s.exptime_s == 120 for s in sectors
               if s.sector in (56, 82))


def test_spoc_index_source_skips_archive_search(cfg, monkeypatch):
    import pandas as pd

    raw = cfg.paths.labels / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({"tic_id": [7, 7, 8], "sector": [3, 1, 2],
                  "filename": ["tess-s0003-7-s_lc.fits", "tess-s0001-7-s_lc.fits",
                               "tess-s0002-8-s_lc.fits"]}).to_parquet(raw / "spoc_lc_index.parquet")
    fetcher._spoc_index.cache_clear()
    monkeypatch.setattr(fetcher, "_search_products", lambda tic: pytest.fail("no search"))
    out = fetcher.list_available_sectors(7, replace(cfg, sector_source="spoc_index"))
    assert [(s.sector, s.author, s.exptime_s) for s in out] == [(1, "SPOC", 120), (3, "SPOC", 120)]
    assert out[0].data_uri == "mast:TESS/product/tess-s0001-7-s_lc.fits"
    fetcher._spoc_index.cache_clear()


def test_prefetch_tic_fills_cache_in_batches(cfg, monkeypatch):
    calls = []

    def batch(ids):
        calls.append(list(ids))
        return [dict(TIC_ROW, ID=str(i)) for i in ids]

    monkeypatch.setattr(resolver, "_query_tic_by_id", batch)
    assert resolver.prefetch_tic([11, 12, 13], cfg, chunk=2) == 3
    assert calls == [[11, 12], [13]]
    assert resolver.prefetch_tic([11, 12, 13], cfg) == 0          # all cached now
    monkeypatch.setattr(resolver, "_query_tic_by_id", lambda t: pytest.fail("cached"))
    assert resolver.resolve_target("TIC 12", cfg).tic_id == 12


def test_spoc_index_subset_keeps_only_needed_stars(cfg, monkeypatch):
    import pandas as pd

    from src.ingest.catalogs import write_spoc_index_subset

    raw = cfg.paths.labels / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"tic_id": [7, 7, 8, 9], "sector": [3, 1, 2, 5],
                  "filename": ["tess-s0003-7-s_lc.fits", "tess-s0001-7-s_lc.fits",
                               "tess-s0002-8-s_lc.fits", "tess-s0005-9-s_lc.fits"],
                  }).to_parquet(raw / "spoc_lc_index.parquet")
    name = write_spoc_index_subset([7, 9, 42], cfg)
    assert sorted(pd.read_parquet(raw / name)["tic_id"]) == [7, 7, 9]
    sub_cfg = replace(cfg, sector_source="spoc_index", spoc_index_file=name)
    fetcher._spoc_index.cache_clear()
    monkeypatch.setattr(fetcher, "_search_products", lambda tic: pytest.fail("no search"))
    assert [s.sector for s in fetcher.list_available_sectors(7, sub_cfg)] == [1, 3]
    assert fetcher.list_available_sectors(8, sub_cfg) == []     # not in the subset
    fetcher._spoc_index.cache_clear()
