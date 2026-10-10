"""Analyze a Star: choose -> settings -> results (the main flow)."""

import json
import random
from dataclasses import replace

import pandas as pd
import streamlit as st

from ui import components as C
from ui import plots
from ui import state
from ui import text as T

state.init()
ss = st.session_state
PRESET_SECTORS = {"quick": 3, "standard": 10, "thorough": None}


# ── helpers ──────────────────────────────────────────────────

def set_target(tic: int, name: str, source: str) -> None:
    """Choose a star. Examples load their precomputed report straight away."""
    if ss.get("target") and ss["target"]["tic"] == tic and ss.get("report"):
        return
    demo = state.load_demo(tic) if source == "example" else None
    ss.update(target={"tic": tic, "name": name, "source": source}, report=demo,
              report_is_demo=demo is not None)
    if demo:
        state.remember(demo, name)


def pick_random() -> None:
    examples = state.load_examples()["examples"]
    if examples:
        ex = random.choice(examples)
        ss["example_sel"] = int(ex["tic_id"])
        ss["pick_mode"] = T.PICK_EXAMPLE


def reset() -> None:
    for k in ("target", "report", "example_sel", "toi_sel"):
        ss[k] = None
    ss["report_is_demo"] = False
    st.query_params.clear()


def estimate(n_available: int, preset: str) -> tuple[str, float]:
    cap = PRESET_SECTORS[preset]
    n = min(n_available, cap) if cap else n_available
    secs = 20 + 12 * n + (90 if preset == "thorough" else 0)
    words = "under a minute" if secs < 60 else f"about {round(secs / 60)} minute" + \
        ("s" if secs >= 90 else "")
    return words, 2.2 * n


def run_analysis(tic: int, name: str, preset: str, overrides: dict) -> None:
    from config import CONFIG, PRESETS, Preset
    from src.pipeline import analyze_target

    cfg = replace(CONFIG, **{k: v for k, v in overrides.items() if k in
                             ("period_min_d", "period_max_d", "detrend_window_d", "sectors")})
    p = PRESETS[preset]
    run_preset = preset
    if overrides.get("use_tls") is not None or overrides.get("max_signals"):
        run_preset = Preset(preset, p.max_sectors, bool(overrides.get("use_tls", p.use_tls)),
                            int(overrides.get("max_signals") or p.max_signals))
    with st.status(T.STAGES["Looking up the star"], expanded=True) as status:
        bar = st.progress(0.0)

        def progress(stage: str, frac: float) -> None:
            if stage in T.STAGES:
                msg = T.STAGES[stage]
            elif stage.lower().startswith("download"):
                msg = T.STAGE_DOWNLOAD.format(detail=stage.split(" ", 1)[-1])
            else:
                msg = stage
            status.update(label=msg)
            bar.progress(min(max(frac, 0.0), 1.0))

        try:
            rep = analyze_target(f"TIC {tic}", preset=run_preset, progress_cb=progress, cfg=cfg)
        except Exception as e:  # noqa: BLE001 - every failure gets a friendly message
            status.update(label=T.ERRORS[C.error_kind(e)][0], state="error")
            ss["last_error"] = e
            return
        status.update(label=T.DONE, state="complete", expanded=False)
    d = rep.to_dict()
    d["preset"] = preset
    ss.update(report=d, report_is_demo=False, last_error=None)
    state.remember(d, name)
    state.write_query(tic, preset)


# ── page ─────────────────────────────────────────────────────

st.title("Analyze a Star")
qp_tic, qp_preset = state.read_query()
if qp_tic and not ss.get("target"):
    ex = state.example_for(qp_tic)
    set_target(qp_tic, ex["display_name"] if ex else f"TIC {qp_tic}",
               "example" if ex else "url")
    if qp_preset:
        ss["preset"] = qp_preset

step = 3 if ss.get("report") else 2 if ss.get("target") else 1
C.step_indicator(step)

# Step 1: choose a star
st.subheader(f"1. {T.STEP_NAMES[0]}")
mode = st.radio(T.PICK_HOW, [T.PICK_EXAMPLE, T.PICK_TOI], horizontal=True, key="pick_mode")
if mode == T.PICK_EXAMPLE:
    data = state.load_examples()
    order = {g: i for i, g in enumerate(data["groups"])}
    exs = sorted(data["examples"], key=lambda e: order.get(e["group"], 99))
    labels = {int(e["tic_id"]): f"{e['group']} · {e['display_name']}: {e['hint']}" for e in exs}
    c1, c2 = st.columns([4, 1])
    sel = c1.selectbox(T.EXAMPLE_LABEL, [None] + list(labels), key="example_sel",
                       format_func=lambda t: "Choose an example..." if t is None else labels[t],
                       help=T.EXAMPLE_HELP)
    c2.button(T.RANDOM_EXAMPLE, on_click=pick_random)
    if sel:
        set_target(int(sel), state.example_for(int(sel))["display_name"], "example")
else:
    tois = state.load_toi_list()
    opts = {i: f"TOI {t['toi']:.2f} · TIC {t['tic']} · {t['status']}" for i, t in enumerate(tois)}
    sel = st.selectbox(T.TOI_LABEL, [None] + list(opts), key="toi_sel",
                       format_func=lambda i: "Type a TOI or TIC number..." if i is None else opts[i],
                       help=T.TOI_HELP)
    if sel is not None:
        t = tois[sel]
        ex = state.example_for(t["tic"])
        set_target(t["tic"], ex["display_name"] if ex else f"TOI {t['toi']:.2f}",
                   "example" if ex else "toi")

with st.expander(T.PICK_MANUAL):
    with st.form("manual"):
        text = st.text_input(T.MANUAL_LABEL, help=T.MANUAL_HELP)
        found = st.form_submit_button(T.MANUAL_FIND)
    if found and text.strip():
        try:
            info = state.resolve(text.strip())
        except Exception as e:  # noqa: BLE001
            C.friendly_error(C.error_kind(e), e)
        else:
            name = info.get("common_name") or f"TIC {info['tic_id']}"
            st.success(T.FOUND.format(what=f"TIC {info['tic_id']} ({name})"))
            ex = state.example_for(info["tic_id"])
            set_target(int(info["tic_id"]), name, "example" if ex else "manual")

target = ss.get("target")
report = ss.get("report")
available = None
if target:
    if report:
        C.star_card(report.get("star") or {"tic_id": target["tic"]}, report.get("sectors") or [])
    else:
        try:
            info = state.resolve(f"TIC {target['tic']}")
            available = state.available_sectors(target["tic"])
            C.star_card(info, [], available)
        except Exception as e:  # noqa: BLE001
            C.friendly_error(C.error_kind(e), e)
            target = None

# Step 2: settings
if target:
    st.subheader(f"2. {T.STEP_NAMES[1]}")
    preset = st.segmented_control(T.PRESET_LABEL, list(T.PRESETS), key="preset",
                                  format_func=lambda p: T.PRESETS[p][0],
                                  help=" ".join(f"{v[0]}: {v[1]}" for v in T.PRESETS.values()))
    preset = preset or "standard"
    n_avail = len(available) if available is not None else len((report or {}).get("sectors") or [])
    words, mb = estimate(max(n_avail, 1), preset)
    st.caption(f"{T.PRESETS[preset][1]} {T.ESTIMATE.format(minutes=words, mb=mb)}")
    overrides: dict = {}
    with st.expander(T.ADVANCED):
        lo, hi = st.slider(T.ADV_PERIOD, 0.5, 30.0, (0.5, 30.0), 0.5, help=T.ADV_PERIOD_HELP)
        overrides.update(period_min_d=lo, period_max_d=hi)
        if available:
            chosen = st.multiselect(T.ADV_SECTORS, [s["sector"] for s in available],
                                    help=T.ADV_SECTORS_HELP)
            if chosen:
                overrides["sectors"] = tuple(int(s) for s in chosen)
        overrides["detrend_window_d"] = st.slider(T.ADV_WINDOW, 0.5, 3.0, 0.75, 0.25,
                                                  help=T.ADV_WINDOW_HELP)
        tls = st.checkbox(T.ADV_TLS, value=preset == "thorough", help=T.ADV_TLS_HELP)
        sig = st.select_slider(T.ADV_SIGNALS, [1, 2, 3],
                               value={"quick": 1, "standard": 2, "thorough": 3}[preset],
                               help=T.ADV_SIGNALS_HELP)
        if tls != (preset == "thorough") or sig != {"quick": 1, "standard": 2,
                                                     "thorough": 3}[preset]:
            overrides.update(use_tls=tls, max_signals=sig)
    if ss.get("report_is_demo"):
        st.info(T.DEMO_BADGE)
        go = st.button(T.RERUN)
    else:
        go = st.button(T.ANALYZE, type="primary", disabled=bool(report))
    if go:
        run_analysis(target["tic"], target["name"], preset, overrides)
        report = ss.get("report")
    if ss.get("last_error") is not None:
        err = ss["last_error"]
        C.friendly_error(C.error_kind(err), err)

# Step 3: results
if report:
    st.subheader(f"3. {T.STEP_NAMES[2]}")
    if (report.get("triage") or {}).get("status") == "no_model":
        C.friendly_error("model_missing")
    C.verdict_card(report)
    left, right = st.columns([3, 2])
    with left:
        C.why_block(report)
    with right:
        C.catalog_block(report)
    C.key_facts(report.get("key_facts") or [])

    st.subheader(T.CHARTS_TITLE)
    kind, mod_fig = plots.module_chart(report)
    charts = [("lightcurve", plots.light_curve(report)), ("fold", plots.fold(report)),
              (kind, mod_fig)]
    for key, fig in charts:
        title, explain = T.CHARTS[key]
        st.markdown(f"**{title}**")
        if fig is None:
            st.caption("Not enough data for this chart.")
            continue
        st.plotly_chart(fig, config={"displayModeBar": False}, key=f"chart_{key}")
        with st.expander(T.WHAT_AM_I):
            st.write(explain)

    if state.expert():
        tabs = st.tabs(T.EXPERT_TABS)
        sig = (report.get("signals") or [{}])[0]
        feats = report.get("features") or {}
        with tabs[0]:
            fig = plots.periodogram_bls(report)
            if fig:
                st.plotly_chart(fig, key="x_bls")
            st.write(f"Best period {sig.get('period', 0):.5f} d, SDE {sig.get('sde', 0):.1f}, "
                     f"depth {sig.get('depth_ppm', 0):.0f} ppm, detected: {sig.get('detected')}")
            st.dataframe(pd.DataFrame(sig.get("top_peaks") or []))
            if (report.get("extra") or {}).get("prewhitening"):
                st.write("Prewhitening:", report["extra"]["prewhitening"])
        with tabs[1]:
            keys = ["odd_even_depth_sigma", "odd_even_depth_ratio", "secondary_snr",
                    "secondary_to_primary_ratio", "secondary_best_phase", "ingress_to_total_ratio",
                    "t14_hr", "duration_ratio_expected", "sector_depth_cv", "implied_rp_rearth"]
            st.dataframe(pd.DataFrame({"feature": keys, "value": [feats.get(k) for k in keys]}))
            fig = plots.per_sector_depths(report)
            if fig:
                st.plotly_chart(fig, key="x_sectors")
            eb = ((report.get("modules") or {}).get("eb") or {}).get("details") or {}
            alt = (eb.get("alternate_eclipse_check") or {}).get("alternate_eclipses")
            if alt:
                st.caption("Alternate-eclipse comparison (information only, not used for the "
                           f"period): {json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in alt.items()})}")
        with tabs[2]:
            fig = plots.periodogram_ls(report)
            if fig:
                st.plotly_chart(fig, key="x_ls")
            st.write(f"Main rhythm {feats.get('var_period_d')} d, ACF period "
                     f"{feats.get('acf_period')} d")
            var = ((report.get("modules") or {}).get("variable") or {}).get("details")
            if var:
                st.json(var)
        with tabs[3]:
            tri = report.get("triage") or {}
            st.markdown(f"**{T.RAW_PROBS}**")
            st.dataframe(pd.DataFrame({"calibrated": tri.get("probabilities") or {},
                                       "raw": tri.get("raw_probabilities") or {}}))
            st.dataframe(pd.DataFrame(tri.get("top_contributions") or []))
            st.write("Routing:", report.get("routing"))
            st.dataframe(pd.DataFrame({"feature": list(feats), "value": list(feats.values())}))
        with tabs[4]:
            st.dataframe(pd.DataFrame(report.get("sectors") or []))
            st.write("Cleaning:", report.get("preprocessing"))
            st.write("Timings (s):", report.get("timings"))

    st.subheader(T.DOWNLOADS)
    name = target["name"] if target else str(report.get("tic_id"))
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(T.DL_HTML, C.html_report(report, [(T.CHARTS[k][0], f) for k, f in charts]),
                       file_name=f"{name}_report.html", mime="text/html")
    d2.download_button(T.DL_JSON, json.dumps(report, default=str), file_name=f"{name}.json",
                       mime="application/json")
    row = {"target": name, "tic_id": report.get("tic_id"),
           "verdict": (report.get("routing") or {}).get("primary_class"),
           "confidence": (report.get("routing") or {}).get("confidence_label"),
           "summary": report.get("plain_summary"),
           "period_d": (report.get("signals") or [{}])[0].get("period")}
    d3.download_button(T.DL_CSV, pd.DataFrame([row]).to_csv(index=False),
                       file_name=f"{name}.csv", mime="text/csv")
    with d4.popover(T.SHARE):
        st.write(T.SHARE_HELP)
        st.code(state.share_suffix(report.get("tic_id"), report.get("preset")
                                   if isinstance(report.get("preset"), str) else "standard"))
    st.button(T.ANALYZE_ANOTHER, on_click=reset)
