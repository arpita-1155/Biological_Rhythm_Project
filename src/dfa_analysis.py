import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("results/activity_5min")
OUTPUT_FILE = Path("results/dfa_features.csv")

# Minimum and maximum window sizes in samples
MIN_WINDOW = 10
MAX_WINDOW = None


def dfa(signal, min_window=10, max_window=None):
    """
    Detrended Fluctuation Analysis (DFA).

    Returns the DFA scaling exponent alpha.
    """

    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]

    if len(signal) < 100:
        return np.nan

    # Remove the mean and create the cumulative profile
    profile = np.cumsum(signal - np.mean(signal))

    if max_window is None:
        max_window = len(profile) // 4

    # Use logarithmically spaced window sizes
    window_sizes = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num=20
        ).astype(int)
    )

    fluctuation_sizes = []

    for window in window_sizes:

        if window < 2:
            continue

        n_segments = len(profile) // window

        if n_segments < 2:
            continue

        segment_fluctuations = []

        for i in range(n_segments):

            start = i * window
            end = start + window

            segment = profile[start:end]
            x = np.arange(window)

            # Linear detrending
            coefficients = np.polyfit(x, segment, 1)
            trend = np.polyval(coefficients, x)

            residuals = segment - trend

            fluctuation = np.sqrt(
                np.mean(residuals ** 2)
            )

            segment_fluctuations.append(fluctuation)

        # Average fluctuation across all segments
        fluctuation_sizes.append(
            np.sqrt(np.mean(np.array(segment_fluctuations) ** 2))
        )

    window_sizes = np.array(window_sizes[:len(fluctuation_sizes)])
    fluctuation_sizes = np.array(fluctuation_sizes)

    # Remove invalid values
    valid = (
        np.isfinite(window_sizes)
        & np.isfinite(fluctuation_sizes)
        & (window_sizes > 0)
        & (fluctuation_sizes > 0)
    )

    window_sizes = window_sizes[valid]
    fluctuation_sizes = fluctuation_sizes[valid]

    if len(window_sizes) < 5:
        return np.nan

    # Linear fit in log-log space
    log_windows = np.log10(window_sizes)
    log_fluctuations = np.log10(fluctuation_sizes)

    alpha, intercept = np.polyfit(
        log_windows,
        log_fluctuations,
        1
    )

    return alpha


def process_subject(file_path):

    subject = file_path.stem

    print(f"\nProcessing: {subject}")

    df = pd.read_csv(file_path)

    signal = df["activity"].to_numpy(dtype=float)

    alpha = dfa(
        signal,
        min_window=MIN_WINDOW,
        max_window=MAX_WINDOW
    )

    return {
        "subject": df["subject"].iloc[0],
        "label": df["label"].iloc[0],
        "dfa_alpha": alpha
    }


print("\n========================================")
print("DFA ANALYSIS")
print("========================================")

files = sorted(INPUT_DIR.glob("*.csv"))

print(f"Subjects found: {len(files)}")

results = []

for file_path in files:

    result = process_subject(file_path)

    results.append(result)


features_df = pd.DataFrame(results)

features_df = features_df.sort_values(
    "subject"
).reset_index(drop=True)


features_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n========================================")
print("DFA ANALYSIS COMPLETE")
print("========================================")

print(f"Subjects processed: {len(features_df)}")

print("\nFeature table:")
print(features_df.to_string(index=False))

print(f"\nSaved to: {OUTPUT_FILE}")