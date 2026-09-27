import pandas as pd
import numpy as np
from pathlib import Path


# =====================================================
# SETTINGS
# =====================================================

INPUT_DIR = Path("results/activity_5min")
OUTPUT_FILE = Path("results/sample_entropy_features.csv")

# Sample Entropy parameters
M = 2
R_RATIO = 0.2


# =====================================================
# SAMPLE ENTROPY FUNCTION
# =====================================================

def sample_entropy(signal, m=2, r_ratio=0.2):

    signal = np.asarray(signal, dtype=float)

    # Remove invalid values
    signal = signal[np.isfinite(signal)]

    if len(signal) < 100:
        return np.nan

    # Standard deviation
    std = np.std(signal)

    if std == 0:
        return 0.0

    # Tolerance
    r = r_ratio * std

    # -------------------------------------------------
    # Create embedded vectors
    # -------------------------------------------------

    def create_vectors(m):

        return np.array([
            signal[i:i + m]
            for i in range(len(signal) - m + 1)
        ])

    # -------------------------------------------------
    # Count matching patterns
    # -------------------------------------------------

    def count_matches(m):

        vectors = create_vectors(m)

        count = 0

        for i in range(len(vectors)):

            # Compare only with later vectors.
            # This avoids counting a vector against itself.

            distances = np.max(
                np.abs(
                    vectors[i + 1:] - vectors[i]
                ),
                axis=1
            )

            count += np.sum(
                distances <= r
            )

        return count

    # -------------------------------------------------
    # Counts for m and m+1
    # -------------------------------------------------

    B = count_matches(m)
    A = count_matches(m + 1)

    # -------------------------------------------------
    # Normalize by number of possible comparisons
    # -------------------------------------------------

    n_m = len(signal) - m + 1
    n_m1 = len(signal) - (m + 1) + 1

    total_B = n_m * (n_m - 1) / 2
    total_A = n_m1 * (n_m1 - 1) / 2

    if B == 0 or A == 0:
        return np.nan

    B_probability = B / total_B
    A_probability = A / total_A

    # -------------------------------------------------
    # Sample Entropy
    # -------------------------------------------------

    sampen = -np.log(
        A_probability / B_probability
    )

    return sampen


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

    entropy = sample_entropy(
        signal,
        m=M,
        r_ratio=R_RATIO
    )

    return {
        "subject": df["subject"].iloc[0],
        "label": df["label"].iloc[0],
        "sample_entropy": entropy
    }


# =====================================================
# PROCESS ALL SUBJECTS
# =====================================================

print("\n========================================")
print("SAMPLE ENTROPY ANALYSIS")
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
print("SAMPLE ENTROPY COMPLETE")
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