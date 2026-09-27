import pandas as pd
import numpy as np
import pywt
from pathlib import Path


# =====================================================
# SETTINGS
# =====================================================

INPUT_DIR = Path("results/activity_5min")
OUTPUT_FILE = Path("results/wavelet_features.csv")

WAVELET = "morl"

# Sampling interval = 5 minutes
SAMPLING_SECONDS = 5 * 60


# =====================================================
# PERIOD RANGES
# =====================================================

# We analyze periods from 2 hours to 48 hours.
#
# 2–6 hours   -> short-term fluctuations
# 6–18 hours  -> intermediate activity patterns
# 18–30 hours -> circadian range
# 30–48 hours -> longer-period variation

periods_hours = np.logspace(
    np.log10(2),
    np.log10(48),
    48
)


# =====================================================
# CONVERT PERIODS TO WAVELET SCALES
# =====================================================

# For CWT:
#
# frequency = central_frequency / (scale * sampling_period)
#
# Therefore:
#
# scale = central_frequency * desired_period / sampling_period

central_frequency = pywt.central_frequency(WAVELET)

scales = (
    central_frequency
    * periods_hours * 3600
    / SAMPLING_SECONDS
)


# =====================================================
# PROCESS ONE SUBJECT
# =====================================================

def extract_features(file_path):

    subject = file_path.stem

    print(f"\nProcessing: {subject}")

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.dropna(
        subset=["timestamp", "activity"]
    )

    signal = df["activity"].to_numpy(
        dtype=float
    )

    # -------------------------------------------------
    # Need enough observations
    # -------------------------------------------------

    if len(signal) < 100:

        print("Not enough data. Skipping.")

        return None

    # -------------------------------------------------
    # Remove mean
    # -------------------------------------------------

    signal = signal - np.mean(signal)

    # -------------------------------------------------
    # Continuous Wavelet Transform
    # -------------------------------------------------

    coefficients, frequencies = pywt.cwt(
        signal,
        scales,
        WAVELET,
        sampling_period=SAMPLING_SECONDS
    )

    # -------------------------------------------------
    # Wavelet power
    # -------------------------------------------------

    power = np.abs(coefficients) ** 2

    # -------------------------------------------------
    # Average power at each period
    # -------------------------------------------------

    mean_power = np.mean(
        power,
        axis=1
    )

    # -------------------------------------------------
    # Total power
    # -------------------------------------------------

    total_power = np.sum(
        mean_power
    )

    # -------------------------------------------------
    # Period masks
    # -------------------------------------------------

    short_mask = (
        (periods_hours >= 2)
        &
        (periods_hours < 6)
    )

    intermediate_mask = (
        (periods_hours >= 6)
        &
        (periods_hours < 18)
    )

    circadian_mask = (
        (periods_hours >= 18)
        &
        (periods_hours <= 30)
    )

    long_mask = (
        (periods_hours > 30)
        &
        (periods_hours <= 48)
    )

    # -------------------------------------------------
    # Band powers
    # -------------------------------------------------

    short_power = np.sum(
        mean_power[short_mask]
    )

    intermediate_power = np.sum(
        mean_power[intermediate_mask]
    )

    circadian_power = np.sum(
        mean_power[circadian_mask]
    )

    long_power = np.sum(
        mean_power[long_mask]
    )

    # -------------------------------------------------
    # Power ratios
    # -------------------------------------------------

    if total_power > 0:

        short_ratio = (
            short_power / total_power
        )

        intermediate_ratio = (
            intermediate_power / total_power
        )

        circadian_ratio = (
            circadian_power / total_power
        )

        long_ratio = (
            long_power / total_power
        )

    else:

        short_ratio = np.nan
        intermediate_ratio = np.nan
        circadian_ratio = np.nan
        long_ratio = np.nan

    # -------------------------------------------------
    # Dominant period
    # -------------------------------------------------

    dominant_index = np.argmax(
        mean_power
    )

    dominant_period_hours = (
        periods_hours[dominant_index]
    )

    dominant_power = (
        mean_power[dominant_index]
    )

    # -------------------------------------------------
    # Create feature dictionary
    # -------------------------------------------------

    features = {

        "subject": df["subject"].iloc[0],

        "label": df["label"].iloc[0],

        "wavelet_total_power": total_power,

        "short_period_power": short_power,

        "intermediate_period_power":
            intermediate_power,

        "circadian_power":
            circadian_power,

        "long_period_power":
            long_power,

        "short_power_ratio":
            short_ratio,

        "intermediate_power_ratio":
            intermediate_ratio,

        "circadian_power_ratio":
            circadian_ratio,

        "long_power_ratio":
            long_ratio,

        "dominant_period_hours":
            dominant_period_hours,

        "dominant_power":
            dominant_power
    }

    return features


# =====================================================
# PROCESS ALL SUBJECTS
# =====================================================

print("\n========================================")
print("WAVELET FEATURE EXTRACTION")
print("========================================")

files = sorted(
    INPUT_DIR.glob("*.csv")
)

print(
    f"Subjects found: {len(files)}"
)


results = []


for file_path in files:

    features = extract_features(
        file_path
    )

    if features is not None:

        results.append(features)


# =====================================================
# CREATE FEATURE TABLE
# =====================================================

features_df = pd.DataFrame(
    results
)


features_df = features_df.sort_values(
    "subject"
).reset_index(drop=True)


# =====================================================
# SAVE
# =====================================================

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =====================================================
# SUMMARY
# =====================================================

print("\n========================================")
print("WAVELET ANALYSIS COMPLETE")
print("========================================")

print(
    f"Subjects processed: {len(features_df)}"
)

print("\nFeature table:")

print(
    features_df.to_string(
        index=False
    )
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)