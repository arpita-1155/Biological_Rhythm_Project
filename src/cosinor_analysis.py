import pandas as pd
import numpy as np
from pathlib import Path


# =====================================================
# SETTINGS
# =====================================================

INPUT_DIR = Path("results/activity_5min")
OUTPUT_FILE = Path("results/cosinor_features.csv")

PERIOD_HOURS = 24.0


# =====================================================
# COSINOR FUNCTION
# =====================================================

def calculate_cosinor(df):

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Remove invalid values
    df = df.dropna(subset=["timestamp", "activity"])

    # -------------------------------------------------
    # Time of day in decimal hours
    # -------------------------------------------------
    #
    # Example:
    # 06:30 -> 6.5
    # 18:45 -> 18.75
    #
    # We use clock time for the 24-hour rhythm.
    # -------------------------------------------------

    time_hours = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute / 60.0
        + df["timestamp"].dt.second / 3600.0
    )

    activity = df["activity"].to_numpy(dtype=float)

    # -------------------------------------------------
    # Cosinor design matrix
    # -------------------------------------------------

    omega = 2 * np.pi / PERIOD_HOURS

    cos_term = np.cos(omega * time_hours)
    sin_term = np.sin(omega * time_hours)

    X = np.column_stack([
        np.ones(len(df)),
        cos_term,
        sin_term
    ])

    # -------------------------------------------------
    # Fit linear regression using least squares
    # -------------------------------------------------

    coefficients, _, _, _ = np.linalg.lstsq(
        X,
        activity,
        rcond=None
    )

    mesor = coefficients[0]
    beta = coefficients[1]
    gamma = coefficients[2]

    # -------------------------------------------------
    # Amplitude
    # -------------------------------------------------

    amplitude = np.sqrt(
        beta**2 + gamma**2
    )

    # -------------------------------------------------
    # Acrophase
    # -------------------------------------------------
    #
    # Convert phase to clock hours.
    # -------------------------------------------------

    phase = np.arctan2(
        gamma,
        beta
    )

    acrophase_hours = (
        phase % (2 * np.pi)
    ) * PERIOD_HOURS / (2 * np.pi)

    # -------------------------------------------------
    # Predicted values
    # -------------------------------------------------

    predicted = X @ coefficients

    # -------------------------------------------------
    # R-squared
    # -------------------------------------------------

    ss_res = np.sum(
        (activity - predicted) ** 2
    )

    ss_tot = np.sum(
        (activity - np.mean(activity)) ** 2
    )

    if ss_tot == 0:
        r_squared = np.nan
    else:
        r_squared = 1 - (ss_res / ss_tot)

    return {
        "mesor": mesor,
        "amplitude": amplitude,
        "acrophase_hours": acrophase_hours,
        "r_squared": r_squared
    }


# =====================================================
# PROCESS ALL SUBJECTS
# =====================================================

print("\n========================================")
print("24-HOUR COSINOR ANALYSIS")
print("========================================")

files = sorted(INPUT_DIR.glob("*.csv"))

print(f"Subjects found: {len(files)}")

results = []


for file_path in files:

    print(f"\nProcessing: {file_path.stem}")

    df = pd.read_csv(file_path)

    features = calculate_cosinor(df)

    features["subject"] = df["subject"].iloc[0]
    features["label"] = df["label"].iloc[0]

    results.append(features)


# =====================================================
# CREATE FEATURE TABLE
# =====================================================

features_df = pd.DataFrame(results)

features_df = features_df[
    [
        "subject",
        "label",
        "mesor",
        "amplitude",
        "acrophase_hours",
        "r_squared"
    ]
]

# Sort by subject
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


print("\n========================================")
print("COSINOR ANALYSIS COMPLETE")
print("========================================")

print(f"Subjects processed: {len(features_df)}")

print("\nFeature table:")
print(features_df.to_string(index=False))

print(f"\nSaved to: {OUTPUT_FILE}")