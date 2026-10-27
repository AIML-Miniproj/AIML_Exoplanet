"""Resolve a star name or TIC ID to a TIC entry with stellar parameters."""

# ── In plain words ───────────────────────────────────────────────────────────
# DATASET: facts about each star.
# What: looks a star up in the TESS Input Catalog (TIC) by name or TIC ID and keeps
#   its temperature, radius, mass, surface gravity, brightness, distance and
#   contamination from neighbours. These become 5 of the 56 features and are used for
#   the planet size, orbit and expected dip length.
# Where: one small file per star, data/cache/resolver/tic_<TIC ID>.json
# Format: JSON (plain text "key": value pairs, readable in any editor).
#   Re-used for 30 days before it is looked up again.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from config import CONFIG, Config
from src.errors import DataDownloadError, TargetNotFoundError
from src.ingest import cache

log = logging.getLogger(__name__)

_TIC_RE = re.compile(r"^\s*(?:TIC)?[\s_-]*(\d{1,10})\s*$", re.IGNORECASE)
_STELLAR_FIELDS = ("teff", "rad", "mass", "logg", "rho", "contratio", "tmag")
_BAD_DISPOSITIONS = {"ARTIFACT", "DUPLICATE", "SPLIT"}


@dataclass
class TargetInfo:
    """One TIC star. Missing values are NaN and listed in ``flags``."""

    tic_id: int
    common_name: str | None = None
    ra: float = math.nan
    dec: float = math.nan
    tmag: float = math.nan
    teff: float = math.nan
    teff_err: float = math.nan
    rad: float = math.nan
    rad_err: float = math.nan
    mass: float = math.nan
    mass_err: float = math.nan
    logg: float = math.nan
    rho: float = math.nan
    contratio: float = math.nan
    distance_pc: float = math.nan
    lumclass: str | None = None
    gaia_id: str | None = None          # TIC lists the Gaia DR2 source id
    hip: str | None = None
    ruwe: float = math.nan
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TargetInfo:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ("ra", "dec", "tmag", "teff", "teff_err", "rad", "rad_err", "mass",
                  "mass_err", "logg", "rho", "contratio", "distance_pc", "ruwe"):
            if known.get(k) is None:
                known[k] = math.nan
        return cls(**known)

    @property
    def label(self) -> str:
        return f"TIC {self.tic_id}" + (f" ({self.common_name})" if self.common_name else "")


def parse_tic(text: str) -> int | None:
    """Return the TIC number if the text is 'TIC 123', 'tic-123' or just digits."""
    m = _TIC_RE.match(str(text))
    return int(m.group(1)) if m else None


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def _str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return None if s in ("", "--", "nan", "None") else s


# ── Network (replaced in tests) ──────────────────────────────

def _set_query_timeout(seconds: int) -> None:
    from astroquery.mast import conf

    conf.timeout = seconds


def _query_tic_by_id(tic_id: int | list[int]) -> list[dict[str, Any]]:
    from astroquery.mast import Catalogs

    _set_query_timeout(CONFIG.query_timeout_s)
    rows = Catalogs.query_criteria(catalog="Tic", ID=tic_id)
    return [dict(zip(rows.colnames, r)) for r in rows]


def _query_tic_by_name(name: str, radius_arcsec: float = 10.0) -> list[dict[str, Any]]:
    from astropy import units as u
    from astroquery.mast import Catalogs

    _set_query_timeout(CONFIG.query_timeout_s)
    rows = Catalogs.query_object(name, catalog="TIC", radius=radius_arcsec * u.arcsec)
    return [dict(zip(rows.colnames, r)) for r in rows]


def _query_gaia_ruwe(gaia_dr2_id: str) -> float:
    from astroquery.gaia import Gaia

    job = Gaia.launch_job(
        "select g.ruwe from gaiadr3.dr2_neighbourhood n "
        "join gaiadr3.gaia_source g on g.source_id = n.dr3_source_id "
        f"where n.dr2_source_id = {int(gaia_dr2_id)} order by n.angular_distance")
    res = job.get_results()
    return _num(res["ruwe"][0]) if len(res) else math.nan


# ── Logic ────────────────────────────────────────────────────

def _row_to_info(row: dict[str, Any], common_name: str | None) -> TargetInfo:
    info = TargetInfo(
        tic_id=int(row["ID"]), common_name=common_name,
        ra=_num(row.get("ra")), dec=_num(row.get("dec")), tmag=_num(row.get("Tmag")),
        teff=_num(row.get("Teff")), teff_err=_num(row.get("e_Teff")),
        rad=_num(row.get("rad")), rad_err=_num(row.get("e_rad")),
        mass=_num(row.get("mass")), mass_err=_num(row.get("e_mass")),
        logg=_num(row.get("logg")), rho=_num(row.get("rho")),
        contratio=_num(row.get("contratio")), distance_pc=_num(row.get("d")),
        lumclass=_str(row.get("lumclass")), gaia_id=_str(row.get("GAIA")),
        hip=_str(row.get("HIP")),
    )
    if info.common_name is None and info.hip:
        info.common_name = f"HIP {info.hip}"
    _fill_derived(info)
    return info


def _fill_derived(info: TargetInfo) -> None:
    """Derive missing mass/density where possible, then flag what is still missing."""
    if math.isnan(info.mass) and not math.isnan(info.logg) and not math.isnan(info.rad):
        info.mass = 10 ** (info.logg - 4.438) * info.rad**2
        info.flags.append("mass_from_logg")
    if math.isnan(info.rho) and not math.isnan(info.mass) and not math.isnan(info.rad):
        info.rho = info.mass / info.rad**3
        info.flags.append("rho_derived")
    for name in _STELLAR_FIELDS:
        if math.isnan(getattr(info, name)):
            info.flags.append(f"missing_{name}")


def _pick_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    good = [r for r in rows if _str(r.get("disposition")) not in _BAD_DISPOSITIONS]
    if not good:
        return None
    if "dstArcSec" in good[0]:
        good.sort(key=lambda r: _num(r.get("dstArcSec")))
    return good[0]


def _cache_key(text: str) -> str:
    tic = parse_tic(text)
    return f"tic_{tic}" if tic is not None else "name_" + re.sub(r"[^a-z0-9+-]", "", text.lower())


def resolve_target(name_or_tic: str | int, cfg: Config = CONFIG,
                   use_cache: bool = True) -> TargetInfo:
    """Look up a star in the TESS Input Catalog.

    Accepts 'TIC 420814525', '420814525' or a name such as 'HD 209458'.
    Raises TargetNotFoundError if nothing matches.
    """
    text = str(name_or_tic).strip()
    if not text:
        raise TargetNotFoundError("empty target name")
    path = cfg.paths.cache / "resolver" / f"{_cache_key(text)}.json"
    if use_cache:
        cached = cache.read_json(path, cfg.resolver_cache_days)
        if cached:
            return TargetInfo.from_dict(cached)

    tic = parse_tic(text)
    try:
        rows = _query_tic_by_id(tic) if tic is not None else _query_tic_by_name(text)
    except Exception as e:  # noqa: BLE001 - name resolver and HTTP errors vary
        msg = str(e).lower()
        if "resolve" in msg or "could not" in msg or "not found" in msg:
            raise TargetNotFoundError(f"{text!r}: {e}") from e
        raise DataDownloadError(f"TIC lookup for {text!r} failed: {e}") from e

    row = _pick_row(rows)
    if row is None:
        raise TargetNotFoundError(f"no TIC entry for {text!r}")
    info = _row_to_info(row, common_name=None if tic is not None else text)

    if cfg.use_gaia_ruwe and info.gaia_id:
        try:
            info.ruwe = _query_gaia_ruwe(info.gaia_id)
        except Exception as e:  # noqa: BLE001
            log.warning("Gaia RUWE lookup failed for %s: %s", info.label, e)
            info.flags.append("ruwe_failed")

    if info.flags:
        log.info("Resolved %s with flags %s", info.label, info.flags)
    else:
        log.info("Resolved %s", info.label)
    cache.write_json(path, info.to_dict())
    if tic is None:   # also cache under the TIC key
        cache.write_json(cfg.paths.cache / "resolver" / f"tic_{info.tic_id}.json", info.to_dict())
    return info


def prefetch_tic(tic_ids, cfg: Config = CONFIG, chunk: int = 500) -> int:
    """Look up many TIC entries in a few batched queries and fill the resolver cache.

    Returns the number of new cache entries. Used before large batch runs.
    """
    todo = [int(t) for t in tic_ids
            if cache.read_json(cfg.paths.cache / "resolver" / f"tic_{int(t)}.json",
                               cfg.resolver_cache_days) is None]
    n = 0
    for i in range(0, len(todo), chunk):
        rows = _query_tic_by_id(todo[i:i + chunk])
        for row in rows:
            if _str(row.get("disposition")) in _BAD_DISPOSITIONS:
                continue
            info = _row_to_info(row, None)
            cache.write_json(cfg.paths.cache / "resolver" / f"tic_{info.tic_id}.json",
                             info.to_dict())
            n += 1
        log.info("TIC prefetch: %d/%d", min(i + chunk, len(todo)), len(todo))
    return n
