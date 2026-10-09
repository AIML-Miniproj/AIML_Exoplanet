# app.py

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os

from config import STAR_NAME, MODEL_PATH, OUTPUT_DIR
from src.fetcher import fetch_lightcurve
from src.preprocessor import preprocess
from src.detector import run_bls
from src.analyzer import analyze_transit
from src.ml_model import build_dataset, train_model, predict

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Exoplanet Transit Detector",
    page_icon  = "🪐",
    layout     = "wide"
)

st.title("🪐 Exoplanet Transit Detection Pipeline")
st.markdown("Detects planetary transits from TESS photometric data using **BLS + XGBoost**")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    star_input = st.text_input("Star Name", value=STAR_NAME)
    run_btn    = st.button("🚀 Analyze", width="stretch")
    st.markdown("---")
    st.caption("Data: NASA TESS via MAST Archive")
    st.caption("ML: XGBoost Classifier")
    st.caption("Single sector mode")

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Light Curves",
    "🔍 BLS Detection",
    "🌀 Phase-Folded Transit",
    "🤖 Results"
])

if run_btn:
    try:
        # FETCH
        with st.spinner("📡 Fetching TESS data..."):
            lc_raw = fetch_lightcurve(star_input)

        # PREPROCESS
        with st.spinner("🧹 Preprocessing..."):
            lc_normalized, lc_flat, trend = preprocess(lc_raw)

        # DETECT
        with st.spinner("🔍 Running BLS periodogram (this takes ~30s)..."):
            bls_output = run_bls(lc_flat)

        # ANALYZE
        with st.spinner("📐 Analyzing transit..."):
            folded, features, physical = analyze_transit(lc_flat, bls_output)

        # ML
        with st.spinner("🤖 Training XGBoost..."):
            X, y       = build_dataset(features, lc_flat, bls_output)
            model      = train_model(X, y)
            ml_label, ml_confidence = predict(features)

        st.success("✅ Analysis complete!")

        best_period = bls_output["best_period"]
        phase       = folded.phase.value
        flux_f      = folded.flux.value

        # Binned curve for phase fold plot
        bins        = np.linspace(-0.5, 0.5, 100)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_means   = [
            np.median(flux_f[(phase >= bins[i]) & (phase < bins[i+1])])
            if np.any((phase >= bins[i]) & (phase < bins[i+1])) else np.nan
            for i in range(len(bins) - 1)
        ]

        # ── TAB 1 — Light Curves ─────────────────────────────
        with tab1:
            st.subheader("Raw vs Cleaned Light Curve")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Before Preprocessing**")
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(lc_normalized.time.value,
                        lc_normalized.flux.value,
                        color="gray", lw=0.5, alpha=0.8)
                ax.set_xlabel("Time (BTJD)")
                ax.set_ylabel("Normalized Flux")
                ax.set_title("Raw Light Curve")
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("**After Sigma Clipping + Flattening**")
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(lc_flat.time.value,
                        lc_flat.flux.value,
                        color="steelblue", lw=0.5, alpha=0.8)
                ax.set_xlabel("Time (BTJD)")
                ax.set_ylabel("Normalized Flux")
                ax.set_title("Cleaned Light Curve")
                st.pyplot(fig)
                plt.close()

            st.info(f"📦 Single sector | Cadences after cleaning: {len(lc_flat)}")

        # ── TAB 2 — BLS Detection ────────────────────────────
        with tab2:
            st.subheader("Box Least Squares Periodogram")
            bls_result = bls_output["bls_result"]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(bls_result.period, bls_result.power,
                    color="navy", lw=0.8, alpha=0.9)
            ax.axvline(best_period, color="red", linestyle="--", lw=1.5,
                       label=f"Best period: {best_period:.4f} d")
            ax.set_xlabel("Period (days)")
            ax.set_ylabel("BLS Power")
            ax.set_title("BLS Periodogram")
            ax.legend()
            st.pyplot(fig)
            plt.close()

            st.markdown("**Top 3 Candidate Periods**")
            st.dataframe(pd.DataFrame({
                "Rank"         : [1, 2, 3],
                "Period (days)": [f"{p:.5f}" for p in bls_output["top3_periods"]],
                "BLS Power"    : [f"{pw:.4f}" for pw in bls_output["top3_powers"]]
            }), width="stretch")

        # ── TAB 3 — Phase-Folded Transit ─────────────────────
        with tab3:
            st.subheader(f"Phase-Folded Transit  |  P = {best_period:.5f} days")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.scatter(phase, flux_f, s=2, color="gray",
                       alpha=0.3, label="Data")
            ax.plot(bin_centers, bin_means, color="red",
                    lw=2, label="Binned median")
            ax.axvspan(-0.02, 0.02, alpha=0.12,
                       color="cyan", label="Transit window")
            ax.set_xlabel("Phase")
            ax.set_ylabel("Normalized Flux")
            ax.set_title("Phase-Folded Light Curve")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        # ── TAB 4 — Results ──────────────────────────────────
        with tab4:
            st.subheader("Analysis Results")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔭 Physical Parameters")
                st.markdown(f"""
| Parameter | Value |
|---|---|
| Period | `{physical['period']:.5f}` days |
| Transit Depth | `{physical['depth']*100:.4f}%` |
| Rp/Rs (measured) | `{physical['rp_over_rs']:.4f}` |
| Rp/Rs (NASA) | `{physical['rp_over_rs_literature']:.4f}` |
| Error | `{physical['error_percent']:.2f}%` |
| Semi-major axis | `{physical['semi_major_axis_AU']:.4f}` AU |
| Eq. Temperature | `{physical['eq_temperature_K']:.1f}` K |
""")

            with col2:
                st.markdown("### 🤖 ML Classification")
                verdict = "✅ Transit Detected" if ml_label == 1 else "❌ Noise"
                if ml_label == 1:
                    st.success(f"**{verdict}**")
                else:
                    st.error(f"**{verdict}**")
                st.metric("Confidence",    f"{ml_confidence}%")
                st.metric("SNR",           f"{features['snr']:.2f}")
                st.metric("Symmetry Score",f"{features['symmetry']:.4f}")

            st.markdown("---")
            st.markdown("### 📊 Model Evaluation")
            col3, col4 = st.columns(2)

            with col3:
                cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix")

            with col4:
                fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
                if os.path.exists(fi_path):
                    st.image(fi_path, caption="Feature Importance (Gain)")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    st.info("👈 Enter a star name in the sidebar and click **Analyze** to begin.")