"""Pick the training stars and propose demo candidates from the cached catalogs.

    python scripts/select_stars.py [--per-class 25]

No light curves are downloaded and no catalogs are re-fetched: everything
comes from data/labels/, apart from one batched TIC lookup to screen the chosen
stars' catalog entries (cached in data/labels/tic_screen.json). Writes
data/labels/training_set.parquet,
data/labels/demo_candidates.csv and data/labels/demo_set.csv, and prints them.

Training filters (all classes): Tmag 8-12 and at least 2 SPOC 2-minute sectors.
PLANET and EB also need a catalog period of 0.5-15 days. VARIABLE skips the
long-period/irregular class, whose changes are far slower than a 2-sector
window can show, and swaps 8 rotators for large-amplitude pulsators (4 RR Lyrae,
4 Cepheids) from the cached Gaia list. Last, every chosen star is checked (one
batched TIC query, cached): entries flagged SPLIT/DUPLICATE/ARTIFACT, and random
noise stars in the cached Gaia variable list, are replaced by the next eligible
star of the same class (replacements are checked too). Also writes a separate
set of 25 EBs, never used for training, for checking the EB period rule. Stars
whose features were already built are kept first. Demo candidates are never used for training.
"""

# ── In plain words ───────────────────────────────────────────────────────────
# DATASET STEP 2: pick the actual training stars from the candidate list.
# Run:    python scripts/select_stars.py --per-class 100
# Reads:  data/labels/labels.parquet and the cached catalogs in data/labels/raw/
# Does:   - keeps stars with TESS brightness 8-12 and at least 2 sectors of 2-minute data
#         - draws 100 per class in a fixed random order (seed 42), so it is repeatable
#         - swaps in RR Lyrae and Cepheid stars so the variable class has big pulsators too
#         - checks each star's catalog entry and replaces broken ones (split / duplicate)
#         - keeps the 6 demo stars OUT of training
# Writes: data/labels/training_set.parquet  the 400 training stars (+ which 2 sectors to use)
#         data/labels/fresh_eb_set.parquet  25 extra binaries, only for testing the period rule
#         data/labels/demo_set.csv          the 6 demo stars shown in the app
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import CONFIG  # noqa: E402

HD209458_TIC = 420814525
# Demo stars as (slot, TIC): HD 209458 b, TOI 184.01 (EB, TOI false positive),
# HIP 36213 (RR Lyrae), TOI 476.01 (unconfirmed candidate), and the "tricky
# cases" group, shown honestly rather than tuned away: TOI 862.01 (transit hidden
# under the star's own fast variability) and TOI 1385.01 (an eclipsing binary
# whose two eclipses look alike, so its period stays ambiguous).
DEMO_SET = [("planet", 420814525), ("eb_toi_fp", 300871545), ("variable", 441584047),
            ("toi_pc", 156987351), ("tricky", 309254930), ("tricky", 329864959)]
EXPECTED_TOI = {300871545: "FP", 156987351: "PC", 309254930: "PC", 329864959: "FP"}
N_RR, N_CEP = 4, 4         # large-amplitude pulsators per 25 training variables (scaled)
TRAIN_SECTORS = 2          # the training preset uses the 2 SPOC sectors closest in time
CLASSES = ("PLANET", "EB", "VARIABLE", "NOISE")
SLOW_VARIABLE_CLASS = "LONG_PERIOD/IRREGULAR"


def _cached_names() -> dict[int, str]:
    """TIC -> common name from the resolver cache (no network)."""
    names = {}
    for p in (CONFIG.paths.cache / "resolver").glob("tic_*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("common_name"):
            names[int(d["tic_id"])] = d["common_name"]
    return names


def _sector_table() -> pd.DataFrame:
    from src.ingest.fetcher import SectorInfo, select_sectors

    cfg = CONFIG.with_preset("training")

    def chosen(sectors: pd.Series) -> list[int]:
        avail = [SectorInfo(int(s), "SPOC", 120, "", "") for s in sorted(set(sectors))]
        return [s.sector for s in select_sectors(avail, TRAIN_SECTORS, cfg=cfg)]

    idx = pd.read_parquet(CONFIG.paths.labels / "raw" / "spoc_lc_index.parquet")
    g = idx.groupby("tic_id")["sector"]
    first = g.apply(chosen)
    return pd.DataFrame({
        "n_spoc": g.nunique(),
        "train_sectors": first.map(lambda s: ",".join(map(str, s))),
        "gap_sectors": first.map(lambda s: s[-1] - s[0] if len(s) > 1 else 0),
    })


def _built_tics() -> set[int]:
    d = CONFIG.paths.processed / "features"
    return {int(p.stem[4:]) for p in d.glob("tic_*.json")}


def _with_sectors(df: pd.DataFrame, sec: pd.DataFrame, tic_col: str = "tic_id") -> pd.DataFrame:
    out = df.join(sec, on=tic_col)
    out["n_spoc"] = out["n_spoc"].fillna(0).astype(int)
    return out


def demo_candidates(lab: pd.DataFrame, sec: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    raw = CONFIG.paths.labels / "raw"
    rows = []

    hd = sec.loc[HD209458_TIC] if HD209458_TIC in sec.index else None
    rows.append({"slot": "planet", "tic_id": HD209458_TIC, "name": "HD 209458 b",
                 "tmag": 7.13, "n_spoc": int(hd["n_spoc"]) if hd is not None else 0,
                 "period_d": 3.52475, "why": "fixed: classic hot Jupiter, deep U-shaped transit"})

    # EB that was a TOI false positive: TOI 'FP' and in the TESS EB catalog
    toi = pd.read_parquet(raw / "toi.parquet")
    ebs = pd.read_parquet(raw / "tess_ebs.parquet")
    ebs = ebs.assign(tic_id=ebs["TIC"].astype(np.int64)).drop_duplicates("tic_id")
    fp = toi[toi["tfopwg_disp"] == "FP"].merge(ebs[["tic_id", "Per", "Morph"]],
                                               left_on="tid", right_on="tic_id")
    fp = _with_sectors(fp, sec)
    fp = fp[(fp["st_tmag"] < 10.5) & (fp["n_spoc"] >= 2) & fp["Per"].between(0.5, 10)
            & (fp["pl_trandep"] > 3000) & (fp["Morph"] < 0.6)]
    fp = fp.sort_values(["pl_trandep"], ascending=False).drop_duplicates("tic_id")
    for _, r in fp.head(3).iterrows():
        rows.append({"slot": "eb_toi_fp", "tic_id": int(r["tic_id"]),
                     "name": names.get(int(r["tic_id"]), f"TOI {r['toi']}"),
                     "tmag": round(float(r["st_tmag"]), 2), "n_spoc": int(r["n_spoc"]),
                     "period_d": round(float(r["Per"]), 5),
                     "why": f"TOI {r['toi']} ruled a false positive and in the EB catalog; "
                            f"dip {r['pl_trandep'] / 1e4:.1f}%, detached (morph "
                            f"{r['Morph']:.2f})"})

    # Variable: short-timescale classes, bright, prefer a known period
    var = lab[(lab["label"] == "VARIABLE") & (lab["n_spoc"] >= 2) & (lab["tmag"] < 10.5)]
    var = var[var["var_class"].isin(["RR_LYRAE", "DELTA_SCUTI/GAMMA_DOR", "ROTATION/SPOTTED"])]
    var = var.assign(has_p=var["catalog_period"].between(0.05, 15),
                     rank=var["var_class"].map({"RR_LYRAE": 0, "DELTA_SCUTI/GAMMA_DOR": 1,
                                                "ROTATION/SPOTTED": 2}))
    var = var.sort_values(["has_p", "rank", "tmag"], ascending=[False, True, True])
    for _, r in var.drop_duplicates("var_class").head(3).iterrows():
        p = r["catalog_period"]
        rows.append({"slot": "variable", "tic_id": int(r["tic_id"]),
                     "name": names.get(int(r["tic_id"]), ""), "tmag": r["tmag"],
                     "n_spoc": int(r["n_spoc"]),
                     "period_d": round(float(p), 5) if np.isfinite(p) else np.nan,
                     "why": f"Gaia class {r['var_class']}" + (", Gaia period known"
                                                              if r["has_p"] else "")})

    # Unconfirmed TOI candidate (PC), held out from training
    pc = _with_sectors(pd.read_parquet(CONFIG.paths.labels / "holdout_candidates.parquet"), sec)
    pc = pc[(pc["tfopwg_disp"] == "PC") & (pc["tmag"] < 10) & (pc["n_spoc"] >= 3)
            & pc["catalog_period"].between(1, 10) & pc["catalog_depth"].between(3000, 15000)]
    pc = pc.sort_values(["n_spoc", "catalog_depth"], ascending=False)
    for _, r in pc.head(3).iterrows():
        rows.append({"slot": "toi_pc", "tic_id": int(r["tic_id"]),
                     "name": names.get(int(r["tic_id"]), f"TOI {r['toi']}"),
                     "tmag": round(float(r["tmag"]), 2), "n_spoc": int(r["n_spoc"]),
                     "period_d": round(float(r["catalog_period"]), 5),
                     "why": f"TOI {r['toi']} still a candidate; dip "
                            f"{r['catalog_depth'] / 1e4:.2f}%, many sectors"})
    out = pd.DataFrame(rows)
    return out.join(sec[["train_sectors", "gap_sectors"]], on="tic_id")


def eligible_pool(lab: pd.DataFrame, cls: str, exclude: set[int], seed: int) -> pd.DataFrame:
    """All stars of ``cls`` passing the training filters, in selection order."""
    built = _built_tics()
    g = lab[(lab["label"] == cls) & lab["tmag"].between(8, 12) & (lab["n_spoc"] >= 2)
            & ~lab["tic_id"].isin(exclude)]
    if cls in ("PLANET", "EB"):
        g = g[g["catalog_period"].between(0.5, 15)]
    if cls == "VARIABLE":
        g = g[g["var_class"] != SLOW_VARIABLE_CLASS]
        g = g.assign(_p=g["catalog_period"].between(0.05, 15))
    else:
        g = g.assign(_p=False)
    g = g.assign(_built=g["tic_id"].isin(built),
                 _r=np.random.default_rng(seed).random(len(g)))
    g = g.sort_values(["_built", "_p", "_r"], ascending=[False, False, True])
    return g.drop(columns=["_built", "_p", "_r"]).assign(
        reused_build=lambda d: d["tic_id"].isin(built))


def training_set(lab: pd.DataFrame, per_class: int, exclude: set[int],
                 seed: int) -> pd.DataFrame:
    picks = []
    for cls in CLASSES:
        g = eligible_pool(lab, cls, exclude, seed)
        if len(g) < per_class:
            print(f"WARNING: only {len(g)} {cls} stars pass the filters")
        picks.append(g.head(per_class))
    return pd.concat(picks, ignore_index=True)


def screen_tic_entries(tics, batch: int = 100) -> dict[int, dict[str, str]]:
    """TIC id -> {"problem": "" if the catalog entry is fine, "gaia": Gaia DR id}.

    Entries flagged SPLIT, DUPLICATE or ARTIFACT (or missing) are rejected by the
    resolver, so a training run on them always fails. One batched query per 100
    stars; answers are cached in data/labels/tic_screen_v2.json.
    """
    from src.ingest.resolver import _BAD_DISPOSITIONS, _query_tic_by_id

    path = CONFIG.paths.labels / "tic_screen_v2.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [int(t) for t in tics if str(int(t)) not in cache]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        rows: dict[int, list[dict]] = {}
        for r in _query_tic_by_id(chunk):
            rows.setdefault(int(r["ID"]), []).append(r)
        for t in chunk:
            entries = rows.get(t)
            if not entries:
                cache[str(t)] = {"problem": "no TIC entry", "gaia": ""}
                continue
            disp = {str(r.get("disposition") or "") for r in entries}
            bad = sorted(disp & _BAD_DISPOSITIONS)
            gaia = str(entries[0].get("GAIA") or "")
            cache[str(t)] = {"problem": f"TIC disposition {bad[0]}" if bad and disp <= set(bad)
                             else "", "gaia": "" if gaia in ("--", "None") else gaia}
        path.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    return {int(t): cache[str(int(t))] for t in tics}


def _gaia_variables() -> set[str]:
    path = CONFIG.paths.labels / "raw" / "gaia_vari_simple_g11.parquet"
    return set(pd.read_parquet(path)["source_id"].astype(np.int64).astype(str))


def problems(rows: pd.DataFrame, gaia_vari: set[str]) -> dict[int, str]:
    """Why each star is unusable ("" if fine): bad TIC entry, or a random noise star
    that the cached Gaia variable list contains (only covers Gaia G < 11)."""
    screen = screen_tic_entries(rows["tic_id"])
    out = {}
    for _, r in rows.iterrows():
        s = screen[int(r["tic_id"])]
        if s["problem"]:
            out[int(r["tic_id"])] = s["problem"]
        elif (r["label"] == "NOISE" and r["source"] == "random_2min"
              and s["gaia"] in gaia_vari):
            out[int(r["tic_id"])] = "in the Gaia variable list"
        else:
            out[int(r["tic_id"])] = ""
    return out


def enforce_quality(train: pd.DataFrame, lab: pd.DataFrame, exclude: set[int], seed: int
                    ) -> tuple[pd.DataFrame, list[tuple[pd.Series, pd.Series, str]], int]:
    """Replace unusable stars by the next eligible, usable star of the same class.

    Runs after every other selection step, so replacements are checked too.
    Returns the set, the swaps, and how many random noise stars were purity-checked.
    """
    gaia_vari = _gaia_variables()
    bad = problems(train, gaia_vari)
    swaps = []
    for idx in [i for i, t in train["tic_id"].items() if bad[int(t)]]:
        old = train.loc[idx]
        pool = eligible_pool(lab, old["label"], set(train["tic_id"]) | exclude, seed)
        if old["label"] == "NOISE":
            pool = pool[pool["source"] == old["source"]]
        new = None
        for k in range(0, len(pool), 10):
            cand = pool.iloc[k:k + 10]
            why = problems(cand, gaia_vari)
            good = cand[[not why[int(t)] for t in cand["tic_id"]]]
            if len(good):
                new = good.iloc[0]
                break
        if new is None:
            print(f"WARNING: no usable replacement for {old['label']} TIC {old['tic_id']}")
            continue
        cols = new.index.intersection(train.columns)
        train.loc[idx, cols] = new[cols]
        swaps.append((old, new, bad[int(old["tic_id"])]))
    n_noise = int(((train["label"] == "NOISE") & (train["source"] == "random_2min")).sum())
    return train, swaps, n_noise


def fresh_eb_set(train: pd.DataFrame, exclude: set[int], sec: pd.DataFrame, n: int,
                 seed: int) -> pd.DataFrame:
    """EBs never used for training or design, for an honest check of the EB period rule.

    Same filters as the training EBs (Tmag 8-12, >= 2 SPOC sectors, period 0.5-15 d),
    drawn from the full TESS EB catalog, excluding training and demo stars, and
    screened the same way.
    """
    raw = CONFIG.paths.labels / "raw"
    ebs = pd.read_parquet(raw / "tess_ebs.parquet")
    ebs = ebs.assign(tic_id=ebs["TIC"].astype(np.int64)).drop_duplicates("tic_id")
    ebs = _with_sectors(ebs, sec)
    pool = ebs[ebs["Tmag"].between(8, 12) & (ebs["n_spoc"] >= 2) & ebs["Per"].between(0.5, 15)
               & ~ebs["tic_id"].isin(set(train["tic_id"]) | exclude)]
    pool = pool.sample(frac=1, random_state=seed + 7)
    rows = pd.DataFrame({
        "tic_id": pool["tic_id"], "label": "EB", "source": "ebcat_fresh",
        "catalog_period": pool["Per"].astype(float), "catalog_epoch": np.nan,
        "catalog_depth": np.nan, "catalog_periods": pool["Per"].map(lambda p: f"{p:.6f}"),
        "tmag": pool["Tmag"].astype(float),
        "notes": pool["Morph"].map(lambda m: f"morph={m:.2f}"),
        "gaia_source_id": None, "n_spoc": pool["n_spoc"], "train_sectors": pool["train_sectors"],
    })
    gaia_vari = _gaia_variables()
    picked = []
    for k in range(0, len(rows), 30):
        cand = rows.iloc[k:k + 30]
        why = problems(cand, gaia_vari)
        picked += [r for _, r in cand.iterrows() if not why[int(r["tic_id"])]]
        if len(picked) >= n:
            break
    return pd.DataFrame(picked[:n]).reset_index(drop=True)


def swap_in_pulsators(train: pd.DataFrame, exclude: set[int], n_rr: int = 2,
                      n_cep: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace rotators with large-amplitude pulsators (RR Lyrae, Cepheids).

    The 400-star label sample has none that pass the filters, so they come from the
    cached Gaia variable list, matched to 2-minute TESS targets by position (the
    same matching and exclusion rules as the label builder). The rotators dropped
    are the last ones in the random order, never one with a known Gaia period.
    Returns (new training set, added rows, dropped rows).
    """
    from src.triage.labels import crossmatch_gaia_to_tic

    raw = CONFIG.paths.labels / "raw"
    gaia = pd.read_parquet(raw / "gaia_vari_simple_g11.parquet")
    gaia = gaia[gaia["best_class_name"].isin(["RR", "CEP"])]
    targets = pd.read_parquet(raw / "tess_2min_targets.parquet")
    x = crossmatch_gaia_to_tic(gaia, targets)
    toi = set(pd.read_parquet(raw / "toi.parquet")["tid"].astype(np.int64))
    ebs = set(pd.read_parquet(raw / "tess_ebs.parquet")["TIC"].astype(np.int64))
    labelled = set(pd.read_parquet(CONFIG.paths.labels / "labels.parquet")["tic_id"])
    x = _with_sectors(x, _sector_table())
    x = x[(x["best_class_score"] >= 0.5) & x["tmag"].between(8, 12) & (x["n_spoc"] >= 2)
          & ~x["tic_id"].isin(toi | ebs | labelled | exclude)]
    x = x.sort_values("best_class_score", ascending=False)
    picks = pd.concat([x[x["best_class_name"] == "RR"].head(n_rr),
                       x[x["best_class_name"] == "CEP"].head(n_cep)])
    names = {"RR": "RR_LYRAE", "CEP": "CEPHEID"}
    added = pd.DataFrame({
        "tic_id": picks["tic_id"].astype(np.int64), "label": "VARIABLE",
        "source": "gaia_dr3_pulsator", "catalog_period": np.nan, "catalog_epoch": np.nan,
        "catalog_depth": np.nan, "catalog_periods": "", "tmag": picks["tmag"],
        "notes": [f"{c} ({names[c]}), score {s:.2f}"
                  for c, s in zip(picks["best_class_name"], picks["best_class_score"])],
        "gaia_source_id": picks["source_id"].astype(np.int64).astype(str),
        "n_spoc": picks["n_spoc"], "train_sectors": picks["train_sectors"],
        "gap_sectors": picks["gap_sectors"],
        "var_class": picks["best_class_name"].map(names), "name": "", "reused_build": False,
    })
    rot = train[(train["label"] == "VARIABLE") & (train["var_class"] == "ROTATION/SPOTTED")
                & train["catalog_period"].isna()]
    dropped = rot.tail(len(added))
    out = pd.concat([train.drop(dropped.index), added], ignore_index=True)
    return out, added, dropped


def check_dispositions(demo: pd.DataFrame) -> None:
    """Re-read the cached TOI table: demo TOIs must still have their expected status."""
    toi = pd.read_parquet(CONFIG.paths.labels / "raw" / "toi.parquet")
    for tic, expected in EXPECTED_TOI.items():
        if tic in set(demo["tic_id"]):
            disp = sorted(set(toi.loc[toi["tid"] == tic, "tfopwg_disp"].dropna()))
            status = "OK" if disp == [expected] else "CHANGED"
            print(f"TOI check TIC {tic}: expected {expected}, cached table says {disp} -> "
                  f"{status}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=CONFIG.limit_per_class)
    ap.add_argument("--seed", type=int, default=CONFIG.random_state)
    ap.add_argument("--fresh-ebs", type=int, default=25,
                    help="size of the separate EB set for checking the EB period rule")
    args = ap.parse_args()

    sec = _sector_table()
    names = _cached_names()
    lab = _with_sectors(pd.read_parquet(CONFIG.paths.labels / "labels.parquet"), sec)
    lab["var_class"] = lab["notes"].str.extract(r"\(([A-Z_/]+)\)", expand=False)
    lab["name"] = lab["tic_id"].map(names).fillna("")

    demo = demo_candidates(lab, sec, names)
    exclude = set(demo["tic_id"])
    train = training_set(lab, args.per_class, exclude, args.seed)
    scale = args.per_class / 25
    train, added, dropped = swap_in_pulsators(train, exclude, n_rr=round(N_RR * scale),
                                              n_cep=round(N_CEP * scale))
    train, swaps, n_noise_checked = enforce_quality(train, lab, exclude, args.seed)
    assert (train.groupby("label").size() == args.per_class).all(), train.groupby("label").size()
    final_demo = pd.concat([demo[demo["tic_id"] == tic].head(1).assign(slot=slot)
                            for slot, tic in DEMO_SET], ignore_index=True)
    fresh = fresh_eb_set(train, exclude, sec, args.fresh_ebs, args.seed)

    out = CONFIG.paths.labels
    train.to_parquet(out / "training_set.parquet", index=False)
    demo.to_csv(out / "demo_candidates.csv", index=False)
    final_demo.to_csv(out / "demo_set.csv", index=False)
    fresh.to_parquet(out / "fresh_eb_set.parquet", index=False)
    check_dispositions(final_demo)
    print("Pulsators added:\n" + added[["tic_id", "tmag", "n_spoc", "train_sectors", "notes"]]
          .to_string(index=False))
    print("Rotators dropped:\n" + dropped[["tic_id", "name", "tmag", "notes"]]
          .to_string(index=False))
    print(f"Quality check (TIC entry, then Gaia variables for {n_noise_checked} random noise "
          f"stars): {len(swaps)} star(s) replaced")
    for old_row, new_row, why in swaps:
        print(f"  {old_row['label']}: TIC {old_row['tic_id']} ({why}) -> TIC {new_row['tic_id']} "
              f"Tmag {new_row['tmag']:.2f}, {int(new_row['n_spoc'])} sectors, "
              f"{new_row['source']}, uses sectors {new_row['train_sectors']}")
    print(f"Fresh EB set: {len(fresh)} stars, periods {fresh['catalog_period'].min():.2f}-"
          f"{fresh['catalog_period'].max():.2f} d, Tmag {fresh['tmag'].min():.1f}-"
          f"{fresh['tmag'].max():.1f}")

    pd.set_option("display.width", 200, "display.max_colwidth", 70, "display.max_rows", 200)
    cols = ["label", "tic_id", "name", "tmag", "n_spoc", "train_sectors", "catalog_period",
            "source", "var_class", "reused_build"]
    print(train[cols].to_string(index=False))
    print(train.groupby("label").agg(n=("tic_id", "size"), reused=("reused_build", "sum"),
                                     tmag_median=("tmag", "median")))
    print()
    print(demo.to_string(index=False))


if __name__ == "__main__":
    main()
