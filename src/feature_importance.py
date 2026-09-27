import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

INPUT_FILE = Path("results/feature_table.csv")
OUTPUT_CSV = Path("results/random_forest_feature_importance.csv")
OUTPUT_PLOT = Path("results/random_forest_feature_importance.png")

RANDOM_STATE = 42


print("\n========================================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("========================================")


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["subject", "label"])

y = df["label"].map({
    "control": 0,
    "condition": 1
})


print(f"\nSubjects: {len(df)}")
print(f"Features: {X.shape[1]}")


# --------------------------------------------------
# 2. Train Random Forest on complete dataset
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    max_features="sqrt"
)

model.fit(X, y)


# --------------------------------------------------
# 3. Extract feature importance
# --------------------------------------------------

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})


importance_df = importance_df.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


# --------------------------------------------------
# 4. Display results
# --------------------------------------------------

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

print(
    importance_df.to_string(index=False)
)


# --------------------------------------------------
# 5. Save feature importance
# --------------------------------------------------

importance_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    f"\nSaved to: {OUTPUT_CSV}"
)


# --------------------------------------------------
# 6. Plot feature importance
# --------------------------------------------------

plt.figure(figsize=(10, 8))

plt.barh(
    importance_df["feature"],
    importance_df["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

# Highest importance at top
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=300
)

plt.show()

print(
    f"Saved plot to: {OUTPUT_PLOT}"
)


print("\n========================================")
print("FEATURE IMPORTANCE COMPLETE")
print("========================================")