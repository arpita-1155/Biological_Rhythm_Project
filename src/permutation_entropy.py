import pandas as pd
import numpy as np
from pathlib import Path
from math import factorial


# =====================================================
# SETTINGS
# =====================================================

INPUT_DIR = Path("results/activity_5min")
OUTPUT_FILE = Path("results/permutation_entropy_features.csv")

# Embedding dimension
M = 3

# Time delay
TAU = 1


# =====================================================
# PERMUTATION ENTROPY FUNCTION
# =====================================================

def permutation_entropy(signal, m=3, tau=1):

    signal = np.asarray(signal, dtype=float)

    # Remove invalid values
    signal = signal[np.isfinite(signal)]

    # Number of required points
    required_points = (m - 1) * tau + 1

    if len(signal) < required_points:
        return np.nan

    # -------------------------------------------------
    # Generate ordinal patterns
    # -------------------------------------------------

    patterns = []

    for i in range(
        len(signal) - (m - 1) * tau
    ):

        window = signal[
            i:i + m * tau:tau
        ]

        # Stable sorting handles equal values
        pattern = tuple(
            np.argsort(
                window,
                kind="stable"
            )
        )

        patterns.append(pattern)

    # -------------------------------------------------
    # Count pattern frequencies
    # -------------------------------------------------

    unique_patterns, counts = np.unique(
        patterns,
        axis=0,
        return_counts=True
    )

    probabilities = (
        counts / np.sum(counts)
    )

    # -------------------------------------------------
    # Shannon entropy
    # -------------------------------------------------

    entropy = -np.sum(
        probabilities
        * np.log(probabilities)
    )

    # -------------------------------------------------
    # Normalize
    #
    # Maximum entropy = log(m!)
    # -------------------------------------------------

    maximum_entropy = np.log(
        factorial(m)
    )

    normalized_entropy = (
        entropy / maximum_entropy
    )

    return normalized_entropy


# =====================================================
# PROCESS ONE SUBJECT
# =====================================================

def process_subject(file_path):

    subject = file_path.stem

    print(f"\nProcessing: {subject}")

    df = pd.read_csv(file_path)

    signal = df["activity"].to_numpy(
        dtype=float
    )

    entropy = permutation_entropy(
        signal,
        m=M,
        tau=TAU
    )

    return {
        "subject": df["subject"].iloc[0],
        "label": df["label"].iloc[0],
        "permutation_entropy": entropy
    }


# =====================================================
# PROCESS ALL SUBJECTS
# =====================================================

print("\n========================================")
print("PERMUTATION ENTROPY ANALYSIS")
print("========================================")

files = sorted(
    INPUT_DIR.glob("*.csv")
)

print(
    f"Subjects found: {len(files)}"
)

results = []


for file_path in files:

    result = process_subject(
        file_path
    )

    results.append(result)


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
print("PERMUTATION ENTROPY COMPLETE")
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