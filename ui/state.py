"""Session state, URL query parameters, recent history and cached data loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_FILE = ROOT / "data" / "examples" / "examples.json"
TOI_FILE = ROOT / "data" / "labels" / "raw" / "toi.parquet"
MAX_RECENT = 8


def init() -> None:
    ss = st.session_state
    ss.setdefault("expert", False)
    ss.setdefault("recent", [])
    ss.setdefault("report", None)          # dict of the report being shown
    ss.setdefault("report_is_demo", False)
    ss.setdefault("target", None)          # {"tic": int, "name": str, "source": str}
    ss.setdefault("running", False)
    ss.setdefault("preset", "standard")


def expert() -> bool:
    return bool(st.session_state.get("expert", False))


# ── URL ──────────────────────────────────────────────────────

def read_query() -> tuple[int | None, str | None]:
    qp = st.query_params
    tic = qp.get("tic")
    try:
        tic = int(str(tic).replace("TIC", "").strip()) if tic else None
    except ValueError:
        tic = None
    preset = qp.get("preset")
    return tic, preset if preset in ("quick", "standard", "thorough") else None


def write_query(tic: int | None, preset: str | None) -> None:
    if tic:
        st.query_params["tic"] = str(tic)
    if preset:
        st.query_params["preset"] = preset


def share_suffix(tic: int | None, preset: str | None) -> str:
    return f"?tic={tic}&preset={preset or 'standard'}" if tic else ""


# ── Recent history ───────────────────────────────────────────

def remember(report: dict[str, Any], name: str) -> None:
    recent = [r for r in st.session_state["recent"] if r["tic"] != report.get("tic_id")]
    recent.insert(0, {"tic": report.get("tic_id"), "name": name,
                      "verdict": (report.get("routing") or {}).get("primary_class"),
                      "report": report})
    st.session_state["recent"] = recent[:MAX_RECENT]


# ── Cached loaders ───────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_examples() -> dict[str, Any]:
    if not EXAMPLES_FILE.exists():
        return {"groups": [], "examples": []}
    return json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_demo(tic: int) -> dict[str, Any] | None:
    for ex in load_examples()["examples"]:
        if int(ex["tic_id"]) == int(tic):
            path = EXAMPLES_FILE.parent / ex["demo_file"]
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
    return None


def example_for(tic: int) -> dict[str, Any] | None:
    return next((e for e in load_examples()["examples"] if int(e["tic_id"]) == int(tic)), None)


@st.cache_data(show_spinner=False)
def load_toi_list() -> list[dict[str, Any]]:
    """The cached TESS candidate list (TOI number, TIC, period, status)."""
    if not TOI_FILE.exists():
        return []
    import pandas as pd

    toi = pd.read_parquet(TOI_FILE, columns=["toi", "tid", "tfopwg_disp", "pl_orbper"])
    toi = toi.sort_values("toi")
    return [{"toi": float(r.toi), "tic": int(r.tid), "status": r.tfopwg_disp,
             "period": float(r.pl_orbper) if r.pl_orbper == r.pl_orbper else None}
            for r in toi.itertuples()]


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def resolve(text: str) -> dict[str, Any]:
    """Look a star up in the TIC (cached). Raises the pipeline's friendly errors."""
    from src.ingest.resolver import resolve_target

    return resolve_target(text).to_dict()


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def available_sectors(tic: int) -> list[dict[str, Any]]:
    """Which TESS sectors observed this star (a cheap search, no download)."""
    from src.ingest.fetcher import list_available_sectors

    return [s.to_dict() for s in list_available_sectors(int(tic))]
