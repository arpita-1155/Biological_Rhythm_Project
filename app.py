import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Biological Rhythm Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM UI
# ============================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.22);
        background: linear-gradient(135deg, rgba(90,120,255,0.12), rgba(100,200,180,0.08));
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        margin-bottom: 0.2rem;
    }

    .hero p {
        margin: 0;
        color: #6b7280;
        font-size: 1.02rem;
    }

    .section-note {
        padding: 0.8rem 1rem;
        border-left: 4px solid #7c83fd;
        background: rgba(124,131,253,0.08);
        border-radius: 8px;
        margin: 0.7rem 0 1rem 0;
    }

    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .mini-card {
        padding: 0.75rem 0.9rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.20);
        margin-bottom: 0.55rem;
    }

    .mini-card b {
        display: block;
        margin-bottom: 0.2rem;
    }

    .footer {
        text-align: center;
        color: #777;
        font-size: 0.85rem;
        padding-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA PATHS
# ============================================================
DATA_DIR = Path("results/activity_5min")
COSINOR_FILE = Path("results/cosinor_features.csv")
WAVELET_FILE = Path("results/wavelet_features.csv")
SAMPLE_ENTROPY_FILE = Path("results/sample_entropy_features.csv")
PERMUTATION_ENTROPY_FILE = Path("results/permutation_entropy_features.csv")
DFA_FILE = Path("results/dfa_features.csv")
FEATURE_TABLE_FILE = Path("results/feature_table.csv")

MODEL_DIR = Path("models")
RF_MODEL_FILE = MODEL_DIR / "random_forest_model.joblib"
XGB_MODEL_FILE = MODEL_DIR / "xgboost_model.joblib"
FEATURE_ORDER_FILE = MODEL_DIR / "feature_order.txt"

# ============================================================
# LOAD SUBJECTS
# ============================================================
subject_files = sorted(DATA_DIR.glob("*.csv"))

if not subject_files:
    st.error("No 5-minute activity files found in results/activity_5min.")
    st.stop()

subject_names = [file.stem for file in subject_files]

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧬 Project Dashboard</div>', unsafe_allow_html=True)
    st.caption("Automated Biological Rhythm Analysis")

    st.divider()

    selected_subject = st.selectbox(
        "Select subject",
        subject_names,
        index=0,
    )

    st.divider()

    st.markdown("**Analysis Pipeline**")
    st.markdown("1. 🧹 Preprocessing")
    st.markdown("2. 🕐 Cosinor Analysis")
    st.markdown("3. 🌊 Wavelet Transform")
    st.markdown("4. 🧠 Entropy + DFA")
    st.markdown("5. 🤖 ML Classification")

    st.divider()

    st.markdown("**Data resolution**")
    st.caption("5-minute resampled activity")

    st.markdown("**Models**")
    st.caption("Random Forest • XGBoost")

    st.divider()
    st.caption("Final Year B.Tech Project • Team of 4")

selected_file = DATA_DIR / f"{selected_subject}.csv"
subject_df = pd.read_csv(selected_file)
label = subject_df["label"].iloc[0]

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🧬 Automated Biological Rhythm Analysis</h1>
    <p>Wearable Activity &amp; Health State Classification</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
This interactive dashboard combines **24-hour rhythm analysis, time-frequency analysis,
signal complexity measures, and machine-learning classification** from wearable activity data.
""")

# ============================================================
# TOP STATUS CARDS
# ============================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Selected Subject", selected_subject)

with c2:
    st.metric("Dataset Label", label)

with c3:
    st.metric("5-min Records", f"{len(subject_df):,}")

with c4:
    duration_days = (
        pd.to_datetime(subject_df["timestamp"]).max()
        - pd.to_datetime(subject_df["timestamp"]).min()
    ).total_seconds() / 86400
    st.metric("Recording Duration", f"{duration_days:.1f} days")

st.caption("Dataset labels are inherited from the source dataset; they are not independent clinical diagnoses.")

# ============================================================
# ACTIVITY SIGNAL
# ============================================================
st.divider()
st.header("📊 Activity Signal")

chart_df = subject_df[["timestamp", "activity"]].copy()
chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
chart_df = chart_df.set_index("timestamp")

st.line_chart(chart_df["activity"], height=320)

with st.expander("About this signal"):
    st.write(
        "The displayed signal is the subject's activity data after 5-minute resampling. "
        "The original cleaned activity data is retained separately in the project pipeline."
    )

# ============================================================
# COSINOR ANALYSIS
# ============================================================
st.divider()
st.header("🕐 24-Hour Cosinor Analysis")

cosinor_df = pd.read_csv(COSINOR_FILE)
cosinor_row = cosinor_df[cosinor_df["subject"] == selected_subject]

if cosinor_row.empty:
    st.warning("Cosinor results are not available for this subject.")
else:
    cosinor_row = cosinor_row.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Mesor", f"{cosinor_row['mesor']:.2f}")
    with c2:
        st.metric("Amplitude", f"{cosinor_row['amplitude']:.2f}")
    with c3:
        st.metric("Acrophase", f"{cosinor_row['acrophase_hours']:.2f} h")
    with c4:
        st.metric("R²", f"{cosinor_row['r_squared']:.3f}")

    st.markdown(
        '<div class="section-note"><b>Interpretation:</b> Cosinor estimates the broad '
        "24-hour rhythm's level, strength, timing, and explained variance.</div>",
        unsafe_allow_html=True,
    )

    activity_df = subject_df[["timestamp", "activity"]].copy()
    activity_df["timestamp"] = pd.to_datetime(activity_df["timestamp"])
    activity_df["hour"] = (
        activity_df["timestamp"].dt.hour
        + activity_df["timestamp"].dt.minute / 60
        + activity_df["timestamp"].dt.second / 3600
    )

    hourly_profile = (
        activity_df.groupby("hour")["activity"]
        .mean()
        .reset_index()
    )

    mesor = cosinor_row["mesor"]
    amplitude = cosinor_row["amplitude"]
    acrophase = cosinor_row["acrophase_hours"]

    phase = 2 * np.pi * acrophase / 24
    omega = 2 * np.pi / 24
    beta = amplitude * np.cos(phase)
    gamma = amplitude * np.sin(phase)

    time_hours = np.linspace(0, 24, 240)
    fitted_curve = (
        mesor
        + beta * np.cos(omega * time_hours)
        + gamma * np.sin(omega * time_hours)
    )

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(
        hourly_profile["hour"],
        hourly_profile["activity"],
        marker="o",
        markersize=3,
        linewidth=1,
        label="Observed Activity",
    )
    ax.plot(
        time_hours,
        fitted_curve,
        linewidth=3,
        label="24-Hour Cosinor Fit",
    )
    ax.set_xlabel("Time of Day (hours)")
    ax.set_ylabel("Mean Activity")
    ax.set_title("Observed Activity vs 24-Hour Cosinor Model")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

# ============================================================
# WAVELET ANALYSIS
# ============================================================
st.divider()
st.header("🌊 Wavelet Analysis")

wavelet_df = pd.read_csv(WAVELET_FILE)
wavelet_row = wavelet_df[wavelet_df["subject"] == selected_subject]

if wavelet_row.empty:
    st.warning("Wavelet results are not available for this subject.")
else:
    wavelet_row = wavelet_row.iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Circadian Power Ratio",
            f"{wavelet_row['circadian_power_ratio']:.3f}",
        )
    with c2:
        st.metric(
            "Dominant Period",
            f"{wavelet_row['dominant_period_hours']:.2f} h",
        )
    with c3:
        st.metric(
            "Dominant Power",
            f"{wavelet_row['dominant_power']:.2e}",
        )

    st.markdown(
        '<div class="section-note"><b>Interpretation:</b> Wavelet analysis examines '
        "how activity power changes across time scales. The dashboard highlights the "
        "2–48 hour period range used in the feature extraction.</div>",
        unsafe_allow_html=True,
    )

    signal = subject_df["activity"].to_numpy(dtype=float)
    signal = signal[np.isfinite(signal)]

    if len(signal) > 10:
        max_points = 3000

        if len(signal) > max_points:
            indices = np.linspace(0, len(signal) - 1, max_points).astype(int)
            signal_plot = signal[indices]
        else:
            signal_plot = signal

        signal_centered = signal_plot - np.mean(signal_plot)

        sampling_period = 5 / 60
        periods_hours = np.logspace(np.log10(2), np.log10(48), 48)
        frequencies = 1 / periods_hours

        scales = pywt.central_frequency("morl") / (
            frequencies * sampling_period
        )

        coefficients, _ = pywt.cwt(
            signal_centered,
            scales,
            "morl",
            sampling_period=sampling_period,
        )

        power = np.abs(coefficients)

        time_axis = np.linspace(
            0,
            len(signal_plot) * sampling_period,
            len(signal_plot),
        )

        fig_wavelet, ax_wavelet = plt.subplots(figsize=(12, 5.8))
        image = ax_wavelet.imshow(
            power,
            aspect="auto",
            extent=[
                time_axis[0],
                time_axis[-1],
                periods_hours[-1],
                periods_hours[0],
            ],
            origin="upper",
        )

        ax_wavelet.set_xlabel("Time (hours)")
        ax_wavelet.set_ylabel("Period (hours)")
        ax_wavelet.set_title("Continuous Wavelet Transform")
        ax_wavelet.set_yscale("log")
        fig_wavelet.colorbar(
            image,
            ax=ax_wavelet,
            label="Wavelet Power",
        )

        st.pyplot(fig_wavelet, clear_figure=True)
    else:
        st.warning("Not enough activity data for Wavelet analysis.")

# ============================================================
# COMPLEXITY ANALYSIS
# ============================================================
st.divider()
st.header("🧠 Complexity & Irregularity Analysis")

sample_entropy_df = pd.read_csv(SAMPLE_ENTROPY_FILE)
permutation_entropy_df = pd.read_csv(PERMUTATION_ENTROPY_FILE)
dfa_df = pd.read_csv(DFA_FILE)

sample_row = sample_entropy_df[
    sample_entropy_df["subject"] == selected_subject
]
permutation_row = permutation_entropy_df[
    permutation_entropy_df["subject"] == selected_subject
]
dfa_row = dfa_df[dfa_df["subject"] == selected_subject]

if sample_row.empty or permutation_row.empty or dfa_row.empty:
    st.warning("One or more complexity features are not available for this subject.")
else:
    sample_entropy_value = sample_row.iloc[0]["sample_entropy"]
    permutation_entropy_value = permutation_row.iloc[0]["permutation_entropy"]
    dfa_alpha_value = dfa_row.iloc[0]["dfa_alpha"]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Sample Entropy", f"{sample_entropy_value:.4f}")

    with c2:
        st.metric("Permutation Entropy", f"{permutation_entropy_value:.4f}")

    with c3:
        st.metric("DFA α", f"{dfa_alpha_value:.4f}")

    st.markdown(
        '<div class="section-note"><b>Interpretation:</b> Entropy measures describe '
        "signal complexity/irregularity, while DFA characterizes long-range scaling behavior.</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# MACHINE LEARNING CLASSIFICATION
# ============================================================
st.divider()
st.header("🤖 Health State Classification")

if not (
    RF_MODEL_FILE.exists()
    and XGB_MODEL_FILE.exists()
    and FEATURE_ORDER_FILE.exists()
):
    st.error(
        "Saved ML models were not found. Run src/train_final_models.py first."
    )
else:
    rf_model = joblib.load(RF_MODEL_FILE)
    xgb_model = joblib.load(XGB_MODEL_FILE)

    with open(FEATURE_ORDER_FILE, "r") as file:
        feature_order = [
            line.strip()
            for line in file
            if line.strip()
        ]

    feature_table = pd.read_csv(FEATURE_TABLE_FILE)

    selected_features = feature_table[
        feature_table["subject"] == selected_subject
    ]

    if selected_features.empty:
        st.error("Feature data is not available for the selected subject.")
    else:
        selected_features = selected_features.iloc[0]

        X_subject = pd.DataFrame(
            [[selected_features[feature] for feature in feature_order]],
            columns=feature_order,
        )

        rf_prediction = rf_model.predict(X_subject)[0]
        rf_probability = rf_model.predict_proba(X_subject)[0, 1]

        xgb_prediction = xgb_model.predict(X_subject)[0]
        xgb_probability = xgb_model.predict_proba(X_subject)[0, 1]

        rf_label = "Condition" if rf_prediction == 1 else "Control"
        xgb_label = "Condition" if xgb_prediction == 1 else "Control"

        st.markdown(
            '<div class="section-note"><b>Important:</b> The values below are '
            "model outputs from models trained on the complete feature dataset. "
            "They should not be interpreted as a clinical diagnosis. "
            "Reported model performance comes from separate subject-level cross-validation.</div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 🌲 Random Forest")
            st.metric("Predicted State", rf_label)
            st.write(
                f"Model-estimated condition probability: "
                f"**{rf_probability * 100:.2f}%**"
            )
            st.progress(float(rf_probability))

        with c2:
            st.markdown("### ⚡ XGBoost")
            st.metric("Predicted State", xgb_label)
            st.write(
                f"Model-estimated condition probability: "
                f"**{xgb_probability * 100:.2f}%**"
            )
            st.progress(float(xgb_probability))

        with st.expander("🔎 View extracted ML feature vector"):
            st.dataframe(
                X_subject.T.rename(columns={0: "Value"}),
                use_container_width=True,
            )

# ============================================================
# FOOTER
# ============================================================
st.divider()

st.markdown(
    '<div class="footer">'
    "Automated Biological Rhythm Analysis & Health State Classification • "
    "Final Year B.Tech Project"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "For research and educational demonstration purposes. "
    "This dashboard is not a clinical diagnostic system."
)
