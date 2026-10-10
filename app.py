"""Streamlit app (interim technical UI; the guided multipage UI comes in Phase 6).

The app never trains anything. It calls ``src.pipeline.analyze_target`` and
renders the returned TargetReport.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import CONFIG, PRESETS
from src.errors import PipelineError
from src.log import setup_logging
from src.pipeline import analyze_target

setup_logging()

st.set_page_config(page_title="Exoplanet Transit Detector", page_icon="🪐", layout="wide")
st.title("Exoplanet Transit Detection Pipeline")
st.markdown("Searches all available TESS sectors of a star for repeating dips, "
            "measures them, and sorts the signal into planet / eclipsing binary / "
            "variable star / noise.")

with st.sidebar:
    st.header("Configuration")
    star_input = st.text_input("Star name or TIC ID", value=CONFIG.star_name,
                               help="For example 'HD 209458' or 'TIC 420814525'.")
    preset = st.selectbox("Preset", list(PRESETS), index=1,
                          help="Quick: up to 3 sectors. Standard: up to 10. "
                               "Thorough: all sectors and up to 3 signals.")
    run_btn = st.button("Analyze", width="stretch", type="primary")
    st.caption("Data: NASA TESS via the MAST archive")


def line_fig(x, y, xlabel, ylabel, title, mode="markers", color="#4c78a8", size=2):
    fig = go.Figure(go.Scattergl(x=x, y=y, mode=mode,
                                 marker=dict(size=size, color=color), line=dict(color=color)))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, height=320,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def render(report):
    pdx = report.plot_data
    sig = report.signals[0]
    tab1, tab2, tab3, tab4 = st.tabs(["Light curves", "Detection", "Folded dip", "Results"])

    with tab1:
        st.subheader("Brightness over time")
        ln = pdx["lc_norm"]
        st.plotly_chart(line_fig(ln["time"], ln["flux"], "Time (BTJD)", "Relative brightness",
                                 "Cleaned light curve (stellar variability kept)",
                                 color="gray"), width="stretch")
        lf = pdx["lc_flat"]
        st.plotly_chart(line_fig(lf["time"], lf["flux"], "Time (BTJD)", "Relative brightness",
                                 "Flattened light curve (used for the dip search)"),
                        width="stretch")
        st.info(f"Sectors used: {[s['sector'] for s in report.sectors]} | "
                f"Cadences after cleaning: {report.preprocessing.get('n_cadences')} | "
                f"Outliers removed: {report.preprocessing.get('n_clipped')}")

    with tab2:
        st.subheader("Box Least Squares periodogram")
        pg = pdx["periodogram"]
        fig = line_fig(pg["period"], pg["power"], "Period (days)", "BLS power", "Periodogram",
                       mode="lines", color="navy")
        fig.add_vline(x=sig["period"], line_dash="dash", line_color="red")
        fig.update_xaxes(type="log")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame([{"Rank": i + 1, "Period (days)": round(p["period"], 5),
                                    "Power": round(p["power"], 1)}
                                   for i, p in enumerate(sig["top_peaks"])]),
                     width="stretch", hide_index=True)
        if len(report.signals) > 1:
            st.markdown("**Additional signals**")
            st.dataframe(pd.DataFrame([{"Period (days)": round(s["period"], 5),
                                        "Depth (ppm)": round(s["depth_ppm"]),
                                        "SDE": round(s["sde"], 1)}
                                       for s in report.signals[1:]]), hide_index=True)

    with tab3:
        st.subheader(f"Folded at P = {sig['period']:.5f} days")
        fd = pdx["fold"]
        fig = line_fig(fd["phase"], fd["flux"], "Phase", "Relative brightness",
                       "All dips stacked", color="lightgray")
        fig.add_trace(go.Scatter(x=fd["bin_phase"], y=fd["bin_flux"], mode="lines",
                                 line=dict(color="red", width=2), name="Binned"))
        st.plotly_chart(fig, width="stretch")
        tz = pdx.get("trapezoid") or {}
        fz = pdx["fold_zoom"]
        fig = line_fig(fz["hours"], fz["flux"], "Hours from mid-dip", "Relative brightness",
                       "Close-up of the dip", color="lightgray")
        if tz.get("x"):
            fig.add_trace(go.Scatter(x=np.array(tz["x"]) * 24, y=tz["model"], mode="lines",
                                     line=dict(color="red", width=2), name="Trapezoid fit"))
        st.plotly_chart(fig, width="stretch")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Signal")
            ph = report.physical

            def fmt(key, pattern):
                v = ph.get(key)
                return pattern.format(v) if v is not None and np.isfinite(v) else "n/a"

            rows = [("Period", fmt("period_d", "{:.5f} d")),
                    ("Dip depth", fmt("depth_ppm", "{:.0f} ppm")),
                    ("Detection strength (SDE)", f"{sig['sde']:.1f}"),
                    ("Size if it is a planet (Earth radii)", fmt("rp_rearth", "{:.2f}")),
                    ("Size if it is a planet (Jupiter radii)", fmt("rp_rjup", "{:.2f}")),
                    ("Orbit size", fmt("a_au", "{:.4f} AU")),
                    ("Temperature", fmt("t_eq_k", "{:.0f} K"))]
            st.table(pd.DataFrame(rows, columns=["Quantity", "Value"]).set_index("Quantity"))
        with col2:
            st.markdown("### Classification")
            tri = report.triage
            if tri.get("status") == "ok":
                st.metric("Most likely", tri["predicted_class"].replace("_", " ").title(),
                          f"{tri['confidence']} confidence", delta_color="off")
                st.bar_chart(pd.Series(tri["probabilities"]), horizontal=True)
            else:
                st.warning("No trained classifier found. Build one with:")
                st.code("python scripts/build_labels.py\npython scripts/build_features.py\n"
                        "python scripts/train_triage.py", language="bash")
        with st.expander("All vetting features"):
            st.dataframe(pd.Series(report.features, name="value"), width="stretch")
        star = report.star
        st.caption(f"TIC {report.tic_id} | Tmag {star.get('tmag')} | Teff {star.get('teff')} K | "
                   f"R {star.get('rad')} Rsun | flags: {', '.join(star.get('flags', [])) or 'none'}"
                   f" | runtime {report.timings.get('total_s')} s")


if run_btn:
    progress = st.progress(0.0, text="Starting...")
    try:
        report = analyze_target(star_input, preset=preset,
                                progress_cb=lambda stage, frac: progress.progress(frac, text=stage))
        st.session_state["report"] = report
        progress.empty()
    except PipelineError as e:
        progress.empty()
        st.error(f"{e.user_message} {e.suggestion}")
        with st.expander("Show technical details"):
            st.code(str(e))
    except Exception as e:  # noqa: BLE001
        progress.empty()
        st.error("Something unexpected went wrong. Try again or pick another star.")
        with st.expander("Show technical details"):
            st.exception(e)

if "report" in st.session_state:
    render(st.session_state["report"])
elif not run_btn:
    st.info("Enter a star name in the sidebar and click **Analyze** to begin.")
