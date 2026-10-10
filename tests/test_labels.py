"""Label rules on tiny hand-made catalogs (no network)."""

import numpy as np
import pandas as pd
import pytest

from src.triage.dataset import recovered
from src.triage.labels import build_labels, crossmatch_gaia_to_tic, label_hash


def _toi(rows):
    df = pd.DataFrame(rows, columns=["toi", "tic_id", "tfopwg_disp", "pl_orbper", "pl_trandep"])
    df["epoch_btjd"] = 1500.0
    df["st_tmag"] = 10.0
    return df


@pytest.fixture
def catalogs():
    toi = _toi([
        (1.01, 1, "CP", 3.0, 5000), (1.02, 1, "PC", 7.0, 300),    # CP wins over its PC
        (2.01, 2, "KP", 50.0, 4000),                              # period outside search
        (3.01, 3, "PC", 2.0, 800),                                # held out
        (4.01, 4, "FP", 1.2, 20000),                              # FP + EB catalog -> EB
        (5.01, 5, "FP", 4.0, 900),                                # FP not in EB catalog -> excluded
        (6.01, 6, "FA", 9.0, 100),                                # NOISE
        (7.01, 7, "FA", 3.0, 100),                                # FA but also Gaia variable -> excluded
        (12.01, 12, "CP", 2.5, 3000),                             # CP and EB catalog -> PLANET
    ])
    ebs = pd.DataFrame({"tic_id": [4, 8, 9, 12, 3], "period": [1.2, 2.0, np.nan, 2.5, 2.0],
                        "epoch_btjd": [1400.0] * 5, "morph": [0.3, 0.7, 0.5, 0.2, 0.1]})
    gaia_x = pd.DataFrame({"tic_id": [7, 10, 11, 8], "best_class_score": [0.9, 0.8, 0.3, 0.9],
                           "best_class_name": ["SOLAR_LIKE", "DSCT|GDOR|SXPHE", "RS", "RS"],
                           "coarse_class": ["ROTATION/SPOTTED", "DELTA_SCUTI/GAMMA_DOR",
                                            "ROTATION/SPOTTED", "ROTATION/SPOTTED"],
                           "gaia_period": [5.0, np.nan, 2.0, 2.0]})
    rng = np.random.default_rng(0)
    quiet = np.arange(100, 400)
    targets = pd.DataFrame({
        "tic_id": np.r_[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], quiet],
        "tmag": np.r_[[9.0] * 12, rng.uniform(5, 12.4, quiet.size)],
        "ra": 0.0, "dec": 0.0})
    return toi, ebs, gaia_x, targets


def test_class_rules_and_priorities(catalogs):
    ls = build_labels(*catalogs, limit_per_class=10, seed=1)
    lab = ls.labels.set_index("tic_id")
    assert lab.loc[1, "label"] == "PLANET" and lab.loc[1, "catalog_period"] == 3.0
    assert "3.000000" in lab.loc[1, "catalog_periods"]
    assert lab.loc[12, "label"] == "PLANET"            # CP beats EB catalog
    assert 2 not in lab.index                          # period outside range
    assert 3 not in lab.index                          # PC held out, even though in EB catalog
    assert set(ls.holdout["tic_id"]) == {3}
    assert lab.loc[4, "label"] == "EB" and lab.loc[4, "source"] == "ebcat+toi_fp"
    assert lab.loc[8, "label"] == "EB"                 # EB beats Gaia variable
    assert 5 not in lab.index and 9 not in lab.index   # FP-only, EB without period
    assert lab.loc[10, "label"] == "VARIABLE"
    assert 11 not in lab.index                         # low Gaia score
    assert 7 not in lab.index                          # variable that is also a TOI
    assert lab.loc[6, "label"] == "NOISE" and lab.loc[6, "source"] == "toi_fa"
    assert ls.summary["excluded"]["TOI FP not in EB catalog"] == 1


def test_conflicts_are_logged(catalogs):
    ls = build_labels(*catalogs, limit_per_class=10)
    conf = ls.conflicts.set_index("tic_id")
    assert {1, 3, 8, 12} <= set(conf.index)
    assert conf.loc[12, "resolution"] == "PLANET"
    assert conf.loc[3, "resolution"].startswith("held out")


def test_random_noise_fill_is_brightness_matched_and_unlisted(catalogs):
    ls = build_labels(*catalogs, limit_per_class=40, seed=3)
    noise = ls.labels[ls.labels["label"] == "NOISE"]
    rand = noise[noise["source"] == "random_2min"]
    # only pool stars near the reference brightness can be drawn (no replacement)
    assert 15 <= len(rand) <= 39
    assert rand["tic_id"].min() >= 100                 # never a catalogued star
    # reference stars are all Tmag 9, so random picks cluster near it
    assert rand["tmag"].between(8.5, 9.5).mean() > 0.9


def test_limit_and_hash_are_deterministic(catalogs):
    a = build_labels(*catalogs, limit_per_class=1, seed=5).labels
    b = build_labels(*catalogs, limit_per_class=1, seed=5).labels
    assert a["label"].value_counts().max() == 1
    assert label_hash(a) == label_hash(b)


def test_crossmatch_uses_proper_motion():
    targets = pd.DataFrame({"tic_id": [1, 2], "tmag": [8.0, 8.0], "ra": [10.0, 50.0],
                            "dec": [0.0, 0.0]})
    # a star 16 years of 500 mas/yr away from its epoch-2000 position (8 arcsec)
    gaia = pd.DataFrame({"source_id": [9], "ra": [10.0 + 8 / 3600], "dec": [0.0],
                         "pmra": [500.0], "pmdec": [0.0], "phot_g_mean_mag": [8.5],
                         "best_class_name": ["RS"], "best_class_score": [0.9]})
    x = crossmatch_gaia_to_tic(gaia, targets)
    assert list(x["tic_id"]) == [1]
    gaia["phot_g_mean_mag"] = 13.0                      # brightness mismatch
    assert crossmatch_gaia_to_tic(gaia, targets).empty


@pytest.mark.parametrize("period,cats,expected", [
    (3.0, [3.01], (True, 1.0)), (1.5, [3.0], (True, 0.5)), (9.0, [3.0], (True, 3.0)),
    (3.2, [3.0], (False, None)), (7.0, [2.0, 3.5], (True, 2.0)), (float("nan"), [3.0], (False, None)),
])
def test_recovery_check(period, cats, expected):
    ok, ratio = recovered(period, cats)
    assert ok == expected[0]
    if ok:
        assert ratio == expected[1]


def test_gaia_ids_keep_full_precision_and_get_periods(catalogs):
    from src.triage.labels import add_variable_periods

    toi, ebs, gaia_x, targets = catalogs
    big = 1779546757669063553                     # not representable as float64
    gaia_x = gaia_x.assign(source_id=[big, big + 2, big + 4, big + 6])
    lab = build_labels(toi, ebs, gaia_x, targets, limit_per_class=10).labels
    row = lab[lab["tic_id"] == 10].iloc[0]
    assert row["gaia_source_id"] == str(big + 2)
    periods = pd.DataFrame({"source_id": [big + 2], "period": [0.123]})
    lab2 = add_variable_periods(lab, periods)
    assert lab2.loc[lab2["tic_id"] == 10, "catalog_period"].iloc[0] == 0.123
    assert lab2.loc[lab2["tic_id"] == 10, "catalog_periods"].iloc[0] == "0.123000"
