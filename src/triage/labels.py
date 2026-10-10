"""Build training labels from public catalogs (no circular training).

Classes and sources
- PLANET:   TOIs with disposition CP (confirmed) or KP (known planet)
- EB:       the TESS eclipsing binary catalog (TOI false positives count only if
            the star is also in that catalog)
- VARIABLE: Gaia DR3 variability classes other than eclipsing binary and
            exoplanet transit, for stars not in the EB catalog or the TOI list
- NOISE:    TOIs marked FA (false alarm) plus random 2-minute targets that are in
            none of the lists above, sampled to match the brightness of the other classes
- held out: TOIs that are still PC/APC (unconfirmed candidates) are never trained
            on; they are kept for the demo list

Priority when a star appears in several lists:
CP/KP > PC/APC (held out) > EB catalog > Gaia variable > FA / random.
Every such conflict is written to ``conflicts.csv``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CONFIG, Config

log = logging.getLogger(__name__)

CLASSES = ["PLANET", "EB", "VARIABLE", "NOISE"]
LABEL_COLUMNS = ["tic_id", "label", "source", "catalog_period", "catalog_epoch",
                 "catalog_depth", "catalog_periods", "tmag", "notes", "gaia_source_id"]

# Gaia DR3 class names (as returned by the archive) grouped into coarse types.
GAIA_COARSE = {
    "DSCT|GDOR|SXPHE": "DELTA_SCUTI/GAMMA_DOR", "RR": "RR_LYRAE", "CEP": "CEPHEID",
    "SOLAR_LIKE": "ROTATION/SPOTTED", "RS": "ROTATION/SPOTTED", "BY": "ROTATION/SPOTTED",
    "ACV|CP|MCP|ROAM|ROAP|SXARI": "ROTATION/SPOTTED", "LPV": "LONG_PERIOD/IRREGULAR",
    "YSO": "LONG_PERIOD/IRREGULAR", "BE|GCAS|SDOR|WR": "LONG_PERIOD/IRREGULAR",
    "SPB": "OTHER", "BCEP": "OTHER", "ACYG": "OTHER", "RCB": "OTHER", "CV": "OTHER",
    "SYST": "OTHER", "AGN": "OTHER", "WD": "OTHER", "MICROLENSING": "OTHER",
}


@dataclass
class LabelSet:
    labels: pd.DataFrame
    holdout: pd.DataFrame
    conflicts: pd.DataFrame
    summary: dict


def crossmatch_gaia_to_tic(gaia: pd.DataFrame, targets: pd.DataFrame,
                           max_sep_arcsec: float = 2.0, max_dmag: float = 2.0) -> pd.DataFrame:
    """Match Gaia variables to 2-minute TIC targets by position.

    Gaia DR3 positions (epoch 2016.0) are moved back to epoch 2000 with the proper
    motion, because TIC positions are given at epoch 2000. A match also needs a
    consistent brightness (|G - Tmag| < max_dmag).
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    g = gaia.dropna(subset=["ra", "dec"]).copy()
    dt_yr = -16.0
    pmra = g["pmra"].fillna(0).to_numpy()
    pmdec = g["pmdec"].fillna(0).to_numpy()
    dec = g["dec"].to_numpy()
    ra2000 = g["ra"].to_numpy() + pmra * dt_yr / 3.6e6 / np.cos(np.radians(dec))
    dec2000 = dec + pmdec * dt_yr / 3.6e6
    cg = SkyCoord(ra2000 * u.deg, dec2000 * u.deg)
    ct = SkyCoord(targets["ra"].to_numpy() * u.deg, targets["dec"].to_numpy() * u.deg)
    idx, sep, _ = cg.match_to_catalog_sky(ct)
    g["tic_id"] = targets["tic_id"].to_numpy()[idx]
    g["tmag"] = targets["tmag"].to_numpy()[idx]
    g["sep_arcsec"] = sep.arcsec
    ok = (g["sep_arcsec"] < max_sep_arcsec) & ((g["phot_g_mean_mag"] - g["tmag"]).abs() < max_dmag)
    g = g[ok].sort_values("sep_arcsec").drop_duplicates("tic_id")
    g["coarse_class"] = g["best_class_name"].map(GAIA_COARSE).fillna("OTHER")
    return g


def _tmag_matched_sample(pool: pd.DataFrame, reference: pd.Series, n: int,
                         rng: np.random.Generator) -> pd.DataFrame:
    """Sample ``n`` rows from ``pool`` so their Tmag histogram follows ``reference``."""
    if n <= 0 or pool.empty:
        return pool.iloc[:0]
    lo, hi = np.floor(reference.min()), np.ceil(reference.max())
    edges = np.arange(lo, max(hi, lo + 1) + 0.25, 0.5)
    ref_h, _ = np.histogram(reference, edges, density=True)
    pool_h, _ = np.histogram(pool["tmag"], edges, density=True)
    b = np.clip(np.digitize(pool["tmag"], edges) - 1, 0, len(edges) - 2)
    inside = (pool["tmag"] >= edges[0]) & (pool["tmag"] <= edges[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(inside & (pool_h[b] > 0), ref_h[b] / pool_h[b], 0.0)
    if w.sum() == 0:
        return pool.sample(min(n, len(pool)), random_state=int(rng.integers(1 << 31)))
    take = rng.choice(len(pool), size=min(n, int((w > 0).sum())), replace=False, p=w / w.sum())
    return pool.iloc[take]


def build_labels(toi: pd.DataFrame, ebs: pd.DataFrame, gaia_x: pd.DataFrame,
                 targets: pd.DataFrame, cfg: Config = CONFIG,
                 limit_per_class: int | None = None, tmag_max: float = 12.5,
                 min_gaia_score: float = 0.5, seed: int | None = None) -> LabelSet:
    """Apply the class rules and priorities; sample up to ``limit_per_class`` per class."""
    limit = cfg.limit_per_class if limit_per_class is None else limit_per_class
    rng = np.random.default_rng(cfg.random_state if seed is None else seed)
    pmin, pmax = cfg.period_min_d, cfg.period_max_d

    tgt = targets.set_index("tic_id")
    two_min = set(tgt.index)
    bright = set(tgt.index[tgt["tmag"] <= tmag_max])

    disp = toi.groupby("tic_id")["tfopwg_disp"].agg(set)
    cpkp = {t for t, d in disp.items() if d & {"CP", "KP"}}
    pc_any = {t for t, d in disp.items() if d & {"PC", "APC"}}
    pc = pc_any - cpkp
    fp = {t for t, d in disp.items() if "FP" in d} - cpkp - pc
    fa = {t for t, d in disp.items() if d <= {"FA"}}
    eb_set = set(ebs["tic_id"])
    var_ok = gaia_x[gaia_x["best_class_score"] >= min_gaia_score]
    var_set = set(var_ok["tic_id"])
    var_any = set(gaia_x["tic_id"])
    toi_any = set(toi["tic_id"])

    rows, conflicts, holdout = [], [], []
    ambiguous: set[int] = set()
    excluded: dict[str, int] = {}

    def exclude(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    def note_conflict(tic: int, chosen: str) -> None:
        srcs = [name for name, s in (("CP/KP", cpkp), ("PC/APC", pc_any), ("TOI FP", fp),
                                     ("TOI FA", fa), ("EB catalog", eb_set),
                                     ("Gaia variable", var_set)) if tic in s]
        if len(srcs) > 1:
            conflicts.append({"tic_id": tic, "sources": "; ".join(srcs), "resolution": chosen})

    # PLANET
    planets = toi[toi["tic_id"].isin(cpkp) & toi["tfopwg_disp"].isin(["CP", "KP"])]
    for tic, grp in planets.groupby("tic_id"):
        note_conflict(tic, "PLANET")
        if tic not in two_min:
            exclude("PLANET: no 2-min data")
            continue
        inrange = grp[(grp["pl_orbper"] >= pmin) & (grp["pl_orbper"] <= pmax)]
        if inrange.empty:
            exclude("PLANET: period outside search range")
            continue
        best = inrange.sort_values("pl_trandep", ascending=False).iloc[0]
        rows.append({"tic_id": tic, "label": "PLANET", "source": "toi_" + best["tfopwg_disp"].lower(),
                     "catalog_period": best["pl_orbper"], "catalog_epoch": best["epoch_btjd"],
                     "catalog_depth": best["pl_trandep"],
                     "catalog_periods": ";".join(f"{p:.6f}" for p in grp["pl_orbper"].dropna()),
                     "notes": f"TOI {best['toi']}"})

    # Held-out candidates (never trained on)
    for tic, grp in toi[toi["tic_id"].isin(pc)].groupby("tic_id"):
        note_conflict(tic, "held out (PC/APC)")
        g = grp.sort_values("pl_trandep", ascending=False).iloc[0]
        holdout.append({"tic_id": tic, "toi": g["toi"], "tfopwg_disp": g["tfopwg_disp"],
                        "catalog_period": g["pl_orbper"], "catalog_epoch": g["epoch_btjd"],
                        "catalog_depth": g["pl_trandep"], "has_2min": tic in two_min,
                        "tmag": g["st_tmag"]})

    # EB
    used = {r["tic_id"] for r in rows} | pc
    for _, r in ebs.drop_duplicates("tic_id").iterrows():
        tic = int(r["tic_id"])
        if tic in used:
            continue
        note_conflict(tic, "EB")
        if tic not in two_min:
            exclude("EB: no 2-min data")
            continue
        if not (pmin <= (r["period"] if pd.notna(r["period"]) else -1) <= pmax):
            exclude("EB: period missing or outside search range")
            continue
        rows.append({"tic_id": tic, "label": "EB",
                     "source": "ebcat+toi_fp" if tic in fp else "ebcat",
                     "catalog_period": r["period"], "catalog_epoch": r["epoch_btjd"],
                     "catalog_depth": np.nan, "catalog_periods": f"{r['period']:.6f}",
                     "notes": f"morph={r['morph']:.2f}" if pd.notna(r["morph"]) else ""})
    used |= {r["tic_id"] for r in rows}
    for tic in fp - eb_set:
        exclude("TOI FP not in EB catalog")

    # VARIABLE
    for _, r in var_ok.iterrows():
        tic = int(r["tic_id"])
        if tic in used:
            continue
        if tic in toi_any or tic in eb_set:
            note_conflict(tic, "excluded (variable that is also a TOI / EB)")
            exclude("VARIABLE: also in TOI list")
            ambiguous.add(tic)
            continue
        p = r.get("gaia_period", np.nan)
        rows.append({"tic_id": tic, "label": "VARIABLE", "source": "gaia_dr3",
                     "catalog_period": p if pd.notna(p) else np.nan, "catalog_epoch": np.nan,
                     "catalog_depth": np.nan,
                     # kept as text: 19-digit ids do not survive a float column
                     "gaia_source_id": str(int(r["source_id"])) if pd.notna(r.get("source_id"))
                     else None,
                     "catalog_periods": f"{p:.6f}" if pd.notna(p) else "",
                     "notes": f"{r['best_class_name']} ({r['coarse_class']}), "
                              f"score {r['best_class_score']:.2f}"})
    used |= {r["tic_id"] for r in rows}

    # NOISE from FA
    for tic in sorted(fa - used - ambiguous):
        if tic not in two_min:
            exclude("NOISE (FA): no 2-min data")
            continue
        rows.append({"tic_id": tic, "label": "NOISE", "source": "toi_fa",
                     "catalog_period": np.nan, "catalog_epoch": np.nan, "catalog_depth": np.nan,
                     "catalog_periods": "", "notes": "TOI false alarm"})

    df = pd.DataFrame(rows)
    df["tmag"] = df["tic_id"].map(tgt["tmag"])
    df = df[df["tic_id"].isin(bright)]

    # Sample each class, then fill NOISE with brightness-matched random quiet stars.
    parts = []
    for cls in CLASSES:
        sub = df[df["label"] == cls]
        if len(sub) > limit:
            sub = sub.sample(limit, random_state=int(rng.integers(1 << 31)))
        parts.append(sub)
    lab = pd.concat(parts, ignore_index=True)

    n_noise = int((lab["label"] == "NOISE").sum())
    if n_noise < limit:
        listed = toi_any | eb_set | var_any
        pool = targets[targets["tic_id"].isin(bright) & ~targets["tic_id"].isin(listed)]
        ref = lab.loc[lab["label"] != "NOISE", "tmag"]
        rand = _tmag_matched_sample(pool, ref, limit - n_noise, rng)
        lab = pd.concat([lab, pd.DataFrame({
            "tic_id": rand["tic_id"].to_numpy(), "label": "NOISE", "source": "random_2min",
            "catalog_period": np.nan, "catalog_epoch": np.nan, "catalog_depth": np.nan,
            "catalog_periods": "", "tmag": rand["tmag"].to_numpy(),
            "notes": "random 2-min target in no catalog"})], ignore_index=True)

    lab = lab.reindex(columns=LABEL_COLUMNS).sort_values(["label", "tic_id"]).reset_index(drop=True)
    lab["tic_id"] = lab["tic_id"].astype("int64")
    hold = pd.DataFrame(holdout)
    conf = pd.DataFrame(conflicts, columns=["tic_id", "sources", "resolution"])
    summary = {
        "class_counts": lab["label"].value_counts().to_dict(),
        "source_counts": lab.groupby(["label", "source"]).size().rename("n").reset_index()
                            .to_dict("records"),
        "eligible_before_sampling": df["label"].value_counts().to_dict(),
        "excluded": excluded,
        "n_conflicts": len(conf),
        "n_holdout": len(hold),
        "n_holdout_2min": int(hold["has_2min"].sum()) if len(hold) else 0,
        "tmag_median_by_class": lab.groupby("label")["tmag"].median().round(2).to_dict(),
    }
    return LabelSet(lab, hold, conf, summary)


def add_variable_periods(labels: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    """Fill catalog periods for VARIABLE rows from a (source_id, period) table."""
    lab = labels.copy()
    pmap = {str(int(k)): v for k, v in zip(periods["source_id"], periods["period"])}
    is_var = (lab["label"] == "VARIABLE") & lab["gaia_source_id"].notna()
    per = lab.loc[is_var, "gaia_source_id"].astype(str).map(pmap)
    lab.loc[is_var, "catalog_period"] = per
    lab.loc[is_var, "catalog_periods"] = per.map(lambda p: f"{p:.6f}" if pd.notna(p) else "")
    return lab


def label_hash(labels: pd.DataFrame) -> str:
    """Short fingerprint of a label set, stored with trained models."""
    import hashlib

    key = labels[["tic_id", "label"]].sort_values("tic_id").to_csv(index=False)
    return hashlib.sha256(key.encode()).hexdigest()[:12]
