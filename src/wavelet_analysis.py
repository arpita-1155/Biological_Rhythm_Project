import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt


# =====================================================
# SETTINGS
# =====================================================

FILE_PATH = "results/activity_5min/condition_1.csv"

# Morlet wavelet
WAVELET = "morl"


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(FILE_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.dropna(subset=["timestamp", "activity"])

signal = df["activity"].to_numpy(dtype=float)


# =====================================================
# REMOVE MEAN
# =====================================================

# Removing the mean helps us focus on fluctuations
# around the subject's average activity.

signal = signal - np.mean(signal)


# =====================================================
# DEFINE SCALES
# =====================================================

# These scales allow us to examine activity patterns
# from short-term fluctuations to longer rhythms.

scales = np.arange(1, 289)


# =====================================================
# CONTINUOUS WAVELET TRANSFORM
# =====================================================

coefficients, frequencies = pywt.cwt(
    signal,
    scales,
    WAVELET,
    sampling_period=5 * 60
)


# =====================================================
# WAVELET POWER
# =====================================================

power = np.abs(coefficients) ** 2


# =====================================================
# CONVERT FREQUENCY TO PERIOD
# =====================================================

period_hours = 1 / frequencies / 3600


# =====================================================
# PLOT WAVELET POWER
# =====================================================

plt.figure(figsize=(15, 7))

plt.imshow(
    power,
    extent=[
        0,
        len(signal) * 5 / 60,
        period_hours[-1],
        period_hours[0]
    ],
    aspect="auto",
    origin="lower"
)

plt.yscale("log")

plt.colorbar(label="Wavelet Power")

plt.xlabel("Time (hours)")
plt.ylabel("Period (hours)")

plt.title(
    "Continuous Wavelet Transform - Condition 1"
)

plt.tight_layout()

plt.show()