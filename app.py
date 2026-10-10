# app.py
"""Exoplanet Detection & False Positive Recycler: a single-page Streamlit app.

Pick one of six example stars in the sidebar. By default its precomputed report
(data/examples/demo/) opens instantly and offline; "Re-run live analysis" runs the
full pipeline on fresh TESS data instead.
"""

import json
import math
import re
import traceback

import streamlit as st

from config import CONFIG
from ui import explain, plots
from ui import theme as th
from ui.views import compute_views, eb_period, final_period, variable_period

st.set_page_config(page_title="Exoplanet Detection & False Positive Recycler", page_icon="🪐",
                   layout="wide")
st.markdown(th.CSS, unsafe_allow_html=True)

TITLE = "🪐 Exoplanet Detection & False Positive Recycler"
SUBTITLE = ("Finds planets in TESS data, and identifies what the signal really is when it "
            "isn't a planet.")
EXAMPLES_FILE = CONFIG.paths.examples / "examples.json"
EVAL_FILE = CONFIG.paths.models / "model_evaluation.json"
TYPE_ICONS = {"Confirmed planets": "🪐", "Eclipsing binaries": "🌗", "Variable stars": "✨",
              "Unconfirmed candidates": "❓", "Tricky cases": "⚠️"}
MODULE_TITLES = {"planet": "🪐 If it is a planet", "eb": "🌗 If it is a binary",
                 "variable": "✨ If it is a variable star", "noise": "〰️ If it is noise"}
STAGES = {
    "Looking up the star": "🔭 Looking up the star...",
    "Cleaning the data": "🧹 Preprocessing...",
    "Searching for repeating dips": "🔍 Running BLS: searching for transits...",
    "Re-cleaning around the dips": "🧹 Re-cleaning around the dips...",
    "Checking whether it's really a planet": "📐 Measuring the dip...",
    "Classifying the signal": "🤖 Classifying...",
    "Working out what it is": "♻️ Working out what the signal really is...",
    "Writing your report": "📝 Writing the report...",
}


# ── Data ─────────────────────────────────────────────────────
@st.cache_data
def load_examples() -> list[dict]:
    return json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))["examples"]


@st.cache_data
def load_demo(tic: int) -> dict:
    return json.loads((CONFIG.paths.examples / "demo" / f"{tic}.json").read_text(encoding="utf-8"))


@st.cache_data
def load_evaluation(mtime: float) -> dict:
    return json.loads(EVAL_FILE.read_text(encoding="utf-8"))


def run_live(tic: int) -> dict:
    """Run the full pipeline with spinner + progress messages; returns the report dict."""
    from src.pipeline import analyze_target

    bar = st.progress(0.0, text="🛰️ Contacting NASA's archive...")

    def progress(stage: str, frac: float) -> None:
        m = re.match(r"Downloading (\d+) sectors of TESS data \((\d+)/(\d+)\)", stage)
        if m:
            text = f"📡 Fetching TESS data ({m.group(1)} sectors)... {m.group(2)}/{m.group(3)}"
        else:
            text = STAGES.get(stage, f"🛰️ {stage}...")
        bar.progress(min(max(frac, 0.0), 1.0), text=text)

    with st.spinner("🛰️ Downloading TESS sectors and searching for transits..."):
        rep = analyze_target(f"TIC {tic}", preset="standard", progress_cb=progress)
        d = rep.to_dict()
        pre = getattr(rep, "_objects", {}).get("pre")
        if pre is not None:
            d["plot_data"]["views"] = compute_views(pre.lc_flat.time.value,
                                                    pre.lc_flat.flux.value, d)
    bar.empty()
    return d


def show_error(e: Exception) -> None:
    from src.errors import PipelineError

    if isinstance(e, PipelineError):
        st.error(f"**{e.user_message}** {e.suggestion}")
    else:
        st.error("**Something went wrong while analysing this star.** Try again, or untick "
                 "\"Re-run live analysis\" to open the precomputed result.")
    with st.expander("Technical details", expanded=False):
        st.code("".join(traceback.format_exception(e)), language="text")


def chart(fig, key: str, bar: bool = True) -> None:
    """A Plotly chart in the app's own dark template (not Streamlit's theme)."""
    st.plotly_chart(fig, theme=None, key=key,
                    config=plots.CONFIG if bar else {"displayModeBar": False})


def pm(d: dict) -> str:
    """'0.76 ± 0.10' from {"mean": ..., "std": ...}."""
    return f"{d['mean']:.2f} ± {d['std']:.2f}"


def num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def main() -> None:
    # ── Header ───────────────────────────────────────────────────
    st.markdown(th.hero_html(TITLE.split(" ", 1)[1].upper(), SUBTITLE,
                             ["TESS", "Multi-sector", "100 training stars", "4-class triage"]),
                unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────
    examples = load_examples()
    by_tic = {e["tic_id"]: e for e in examples}
    with st.sidebar:
        st.markdown('<div class="side-title">Select a star</div>', unsafe_allow_html=True)
        tic = st.selectbox(
            "Select a star", list(by_tic), key="star", label_visibility="collapsed",
            format_func=lambda t: f"{TYPE_ICONS.get(by_tic[t]['group'], '•')} "
                                  f"{by_tic[t]['display_name']}: {by_tic[t]['type']}")
        st.markdown(f'<div class="star-desc">{by_tic[tic]["hint"]}</div>', unsafe_allow_html=True)
        live = st.checkbox("Re-run live analysis (~1–2 min, downloads TESS data)", value=False,
                           key="live")
        run_btn = st.button("🚀 Analyze", key="analyze", width="stretch")
        st.markdown("---")
        st.caption("Data: NASA TESS via MAST")
        st.caption("Model: XGBoost triage + physics rules")
        st.caption("Multi-sector mode")

    if run_btn:
        try:
            report = run_live(tic) if live else load_demo(tic)
            st.session_state.update(report=report, report_tic=tic, report_live=live)
        except Exception as e:  # noqa: BLE001 - shown to the user, details collapsed
            st.session_state.pop("report", None)
            show_error(e)

    report = st.session_state.get("report")
    if not report or st.session_state.get("report_tic") != tic:
        if not run_btn:
            st.info("👈 Pick a star in the sidebar and click **Analyze**.")
        return

    ex = by_tic[tic]
    routing = report.get("routing") or {}
    final = routing.get("primary_class") or (report.get("triage") or {}).get("predicted_class", "")
    cls_col = th.color(final)
    features = report.get("features") or {}
    signal = (report.get("signals") or [{}])[0]
    pdata = report.get("plot_data") or {}
    views = pdata.get("views") or {}
    modules = report.get("modules") or {}
    probs = routing.get("probabilities") or (report.get("triage") or {}).get("probabilities") or {}
    st.markdown(th.class_color_css(final), unsafe_allow_html=True)
    if (report.get("triage") or {}).get("status") == "no_model":
        st.warning("No trained classifier found in models/, so there is no verdict. Build it with "
                   "scripts/build_features.py and scripts/train_triage.py.")
    st.caption(("🔴 Live analysis" if st.session_state.get("report_live") else
                "⚡ Precomputed result (tick \"Re-run live analysis\" to analyse fresh data)")
               + f" · {ex['display_name']} · TIC {tic}")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Light Curves", "🔍 BLS Detection",
                                      "🌀 Phase-Folded", "🤖 Results"])

    # ── TAB 1: Light curves ──────────────────────────────────────
    with tab1:
        st.subheader("Raw vs cleaned light curve")
        spans = plots.sector_spans(pdata.get("lc_norm") or {})
        col1, col2 = st.columns(2)
        with col1:
            chart(plots.light_curve(pdata.get("lc_norm") or {}, spans,
                  "Before preprocessing (normalized, stitched)"),
    key="lc_raw")
        with col2:
            chart(plots.light_curve(pdata.get("lc_flat") or {}, spans,
                  "After cleaning + flattening",
                  col=th.rgba(th.CYAN, 0.45)),
    key="lc_flat")
        sectors = [s.get("sector") for s in report.get("sectors") or []]
        n_cad = (report.get("preprocessing") or {}).get("n_cadences")
        st.info(f"📦 {len(sectors)} sectors used: {sectors} | Cadences after cleaning: "
                f"{n_cad:,}" if n_cad else f"📦 {len(sectors)} sectors used: {sectors}")
        st.caption("Shaded bands are TESS sectors (about 27 days each). Drag to zoom, double-click "
                   "to reset. The plots show a thinned copy; the analysis used every point.")

    # ── TAB 2: BLS detection ─────────────────────────────────────
    with tab2:
        st.subheader("Box Least Squares periodogram")
        best, sde = num(signal.get("period")), num(signal.get("sde", features.get("sde")))
        chart(plots.bls_periodogram(pdata.get("periodogram") or {}, best, sde, th.VIOLET),
              key="bls")
        st.caption("BLS slides a box-shaped dip across every trial period; the highest peak is the "
                   "period where repeating dips fit best. SDE says how far that peak stands out "
                   "from the rest.")
        peaks = views.get("top_peaks") or [{"period": p.get("period"), "power": p.get("power")}
                                           for p in (signal.get("top_peaks") or [])[:3]]
        if peaks:
            st.markdown("**Top 3 candidate periods**")
            st.table([{"Rank": i, "Period (days)": f"{num(p['period']):.5f}",
                       "BLS power": f"{num(p['power']):,.1f}"} for i, p in enumerate(peaks, 1)])
        ranked = sorted(probs, key=lambda c: -num(probs[c]))[:2]
        if final == "VARIABLE" or "VARIABLE" in ranked:
            st.subheader("Lomb-Scargle periodogram")
            chart(plots.ls_periodogram(pdata.get("ls") or {}, variable_period(report)),
                  key="ls")
            st.caption("Variable stars change brightness smoothly instead of dipping, so they are "
                       "found with this second periodogram, which looks for smooth, regular "
                       "cycles. The marked period is the one the variable-star module uses (it may "
                       "be twice the tallest peak when alternate cycles differ).")

    # ── TAB 3: Phase-folded ──────────────────────────────────────
    with tab3:
        period, kind = final_period(report)
        fv = views.get("variable") if kind == "variable" else views.get("final")
        if not fv:
            st.warning("This report has no folded light curve.")
        elif kind == "variable":
            st.subheader(f"Folded on the star's own cycle | P = {fv['period']:.5f} days")
            chart(plots.fold(fv, cls_col, [], "Phase-folded light curve"),
                  key="fold")
            st.caption("A variable star changes smoothly over its whole cycle instead of staying "
                       "flat with one short dip. The faintest point is placed at phase 0.")
        else:
            half = fv.get("half_window") or 0.02
            if kind == "eb":
                sec = (views.get("eb") or {}).get("secondary_phase", 0.5)
                st.subheader(f"Folded on the binary's true period | P = {fv['period']:.5f} days")
                chart(plots.fold(fv, cls_col, [(0, half), (sec, half)],
                      "Phase-folded light curve", x_range=(-0.25, 0.75)),
           key="fold")
                st.caption(f"Two stars: the deep eclipse at phase 0 and a second, shallower "
                           f"one near phase {sec:.2f}, when the fainter star is hidden. A planet "
                           f"makes one U-shaped dip and almost nothing at phase 0.5.")
            else:
                st.subheader(f"Phase-folded transit | P = {fv['period']:.5f} days")
                chart(plots.fold(fv, cls_col, [(0, half)], "Phase-folded light curve",
                      x_range=(-0.5, 0.5)), key="fold")
                st.caption("A planet makes one U-shaped dip; a binary usually shows a second, "
                           "smaller dip at phase 0.5.")
        ebv = views.get("eb")
        if ebv and (kind == "eb" or "eb" in modules):
            c1, c2 = st.columns(2)
            with c1:
                chart(plots.overlay(ebv["primary"], ebv["secondary"],
                      ("Primary (phase 0)",
                       f"Secondary (phase {ebv['secondary_phase']:.2f})"),
                      (th.color("EB"), th.CYAN), "Primary vs secondary eclipse"),
        key="eb_ps")
                st.caption("Both eclipses centred on top of each other. Different depths mean two "
                           "stars of different temperature; a planet gives almost no second dip.")
            with c2:
                chart(plots.overlay(ebv["odd"], ebv["even"], ("Odd eclipses",
                      "Even eclipses"),
    (th.color("EB"), th.color("VARIABLE")),
    "Odd vs even eclipses"),
    key="eb_oe")
                st.caption("Odd and even eclipses should match if the period is right; different "
                           "depths would mean the true period is twice as long.")
        if kind != "variable" and views.get("variable") and "variable" in modules:
            vv = views["variable"]
            chart(plots.fold(vv, th.color("VARIABLE"), [],
                  f"Folded on the star's own cycle (P = {vv['period']:.4f} d)"),
       key="fold_var")
            st.caption("The variable-star module's view: if the star itself varies smoothly "
                       "on this cycle, the curve rises and falls over the whole phase range.")

    # ── TAB 4: Results ───────────────────────────────────────────
    with tab4:
        # Section 1: classification
        with st.container(key="glass-cls-verdict"):
            st.subheader("Classification")
            left, right = st.columns([1.25, 1])
            with left:
                top_p = num(routing.get("top_probability")) if routing else None
                st.markdown(th.verdict_html(final, routing.get("confidence_label", ""), top_p,
                                            report.get("plain_summary", "")),
                            unsafe_allow_html=True)
            with right:
                chart(plots.probabilities(probs), bar=False,
                      key="probs")
            st.markdown("**Why?**")
            st.markdown("\n".join(f"- {r}" for r in explain.reasons(report)) or
                        "- No strong reasons to report.")
            kind_c, what = explain.catalog_statement(report)
            mark = {"agree": "✅ agrees", "disagree": "⚠️ disagrees",
                    "unsure": "🔍 we're not sure yet, kept for review", "none": "—"}[kind_c]
            st.markdown(f'<div class="catalog">NASA catalog says: <b>{what}</b> → {mark}</div>',
                        unsafe_allow_html=True)
            cp = explain.catalog_period(report, final_period(report)[0])
            if cp:
                verdict_p = {"match": "✅ match (within 1%)",
                             "half": "⚠️ we found half the catalog period",
                             "double": "⚠️ we found double the catalog period",
                             "differs": "⚠️ the periods differ"}[cp["kind"]]
                st.markdown(f'<div class="catalog">Catalog period: <b>{cp["catalog"]:.4f} d</b>'
                            f' | Our period: <b>{cp["ours"]:.4f} d</b> → {verdict_p}</div>',
                            unsafe_allow_html=True)
                if final == "EB" and cp["kind"] == "half":
                    st.warning(f"Our pipeline measures {cp['ours']:.2f} d, NASA's catalog says "
                               f"{cp['catalog']:.2f} d. When both eclipses look alike, "
                               f"brightness data alone can't tell these apart. Known "
                               f"limitation.")
            else:
                st.markdown('<div class="catalog">Catalog period: none known for this star, '
                            'so there is nothing to compare.</div>', unsafe_allow_html=True)

        # Section 2: findings
        with st.container(key="glass-cls-findings"):
            st.subheader("Findings")
            hint = explain.review_hint(report)
            if final == "NEEDS_REVIEW" and hint:
                st.markdown(f'<div class="hint">🔍 {hint}</div>', unsafe_allow_html=True)
            left, right = st.columns([1.4, 1])
            with left:
                shown = routing.get("modules") or list(modules)
                for name in shown:
                    facts = (modules.get(name) or {}).get("key_facts") or []
                    if not facts:
                        continue
                    if len(shown) > 1:
                        st.markdown(f"**{MODULE_TITLES.get(name, name)}**")
                    rows = "\n".join(
                        f"| {f[0]} | `{str(f[1]).replace('|', '/')}` {f[2] if len(f) > 2 else ''} |"
                        for f in facts)
                    st.markdown(f"| Parameter | Value |\n|---|---|\n{rows}")
            with right:
                m1, m2 = st.columns(2)
                m3, m4 = st.columns(2)
                m1.metric("Period", f"{period:.4f} d" if math.isfinite(period) else "—")
                if kind == "variable":
                    amp = num(features.get("amp_p2p_ppm")) / 1e4
                    m2.metric("Amplitude", f"{amp:.2f} %" if math.isfinite(amp) else "—")
                else:
                    ecl = ((modules.get("eb") or {}).get("details") or {}).get("eclipses") or {}
                    depth = num(ecl.get("primary_depth")) * 100 if kind == "eb" else \
                        num(signal.get("depth_ppm", features.get("depth_ppm"))) / 1e4
                    m2.metric("Depth", f"{depth:.3f} %" if math.isfinite(depth) else "—")
                snr = num(features.get("depth_snr", signal.get("depth_snr")))
                m3.metric("SNR", f"{snr:.1f}" if math.isfinite(snr) else "—")
                m4.metric("Sectors used", f"{len(report.get('sectors') or [])}")
                if kind == "eb" and math.isfinite(eb_period(report)) and \
                        abs(eb_period(report) / num(signal.get("period")) - 1) > 0.01:
                    st.caption(f"The search found {num(signal.get('period')):.4f} d; the binary "
                               f"module doubled it because the alternate eclipses differ.")

        # Section 3: model evaluation
        with st.container(key="glass-eval"):
            st.subheader("Model evaluation")
            if not EVAL_FILE.exists():
                st.warning("Saved evaluation results not found. Create them (no retraining) with "
                           "`python scripts/save_evaluation.py`.")
            else:
                ev = load_evaluation(EVAL_FILE.stat().st_mtime)
                classes = ev["classes"]
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown("**Confusion matrix** (5-fold grouped CV, normalized by "
                                "true class)")
                    chart(plots.confusion(ev["confusion_normalized"], ev["confusion"],
                          classes), bar=False,
          key="confusion")
                with c2:
                    st.markdown("**Per-class scores** (mean ± std over the 5 folds)")
                    pc = ev["per_class"]
                    st.table([{"Class": f"{th.icon(c)} {th.label(c)}",
                               **{k: pm(pc[c][m])
                                  for k, m in (("Precision", "precision"), ("Recall", "recall"),
                                               ("F1", "f1"), ("PR-AUC", "pr_auc"))}}
                              for c in classes])
                    mf = ev["macro_f1"]
                    st.metric("Macro-F1", pm(mf))
                    st.markdown("**Model comparison** (same folds)")
                    st.table([{"Model": r["model"],
                               "Macro-F1": pm(r["macro_f1"]), "Accuracy": pm(r["accuracy"])}
                              for r in ev["comparison"]])
                c3, c4 = st.columns([1, 1])
                with c3:
                    st.markdown("**What the classifier relies on** (top 10 features)")
                    chart(plots.importance(ev["importance"], explain.READABLE),
                          bar=False, key="importance")
                with c4:
                    st.markdown("**Signal recovery** (did the search find the catalog period?)")
                    rec = ev.get("recovery") or {}
                    how = {"PLANET": "BLS period matches the catalog",
                           "EB": "BLS period matches the catalog or half of it",
                           "VARIABLE": "Significant periodogram peak (false-alarm chance < 1%)",
                           "NOISE": "No catalog period to check"}
                    rows = []
                    for c in classes:
                        r = rec.get(c) or {}
                        n_chk, rate = r.get("n_recovery_checked") or 0, r.get("recovery_rate")
                        found = (f"{round(rate * n_chk)}/{n_chk}" if rate is not None and n_chk
                                 else "—")
                        ratios = r.get("recovered_at_ratio") or {}
                        if c == "EB" and ratios:
                            found += (f" ({ratios.get('1.0', 0)} at the true period, "
                                      f"{ratios.get('0.5', 0)} at half)")
                        rows.append({"Class": f"{th.icon(c)} {th.label(c)}", "Recovered": found,
                                     "How it is counted": how[c]})
                    st.table(rows)
                    ebp = ev.get("eb_true_period") or {}
                    if ebp:
                        st.markdown("**Binary true period** (after the binary module's P vs 2P "
                                    "decision)")
                        st.table([{"Binaries": {"fresh": "Fresh (never used in design)",
                                                "training": "Training"}.get(k, k),
                                   "Correct period": f"{v['correct']}/{v['n']}"}
                                  for k, v in sorted(ebp.items())])
            st.caption("Trained on 100 TESS stars (25 per class). Small academic dataset, so "
                       "expect ±0.1 variation.")


# Streamlit runs this file as __main__. A multiprocessing worker that re-imports it
# (as __mp_main__, e.g. after an in-process AppTest) must not draw the page.
if __name__ != "__mp_main__":
    main()
