import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# SETTINGS
# ==========================================

FILE_PATH = "results/cleaned_depresjon/condition_1.csv"


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(FILE_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])


# ==========================================
# PLOT RAW ACTIVITY
# ==========================================

plt.figure(figsize=(15, 5))

plt.plot(
    df["timestamp"],
    df["activity"],
    linewidth=0.6
)

plt.title("Raw Activity Signal - Condition 1")
plt.xlabel("Time")
plt.ylabel("Activity")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()