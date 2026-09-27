import pandas as pd
from pathlib import Path

INPUT_FILE = Path("results/feature_table.csv")

print("\n========================================")
print("PREPARING ML DATASET")
print("========================================")

# Load combined feature table
df = pd.read_csv(INPUT_FILE)

print(f"\nOriginal shape: {df.shape}")


# Keep subject IDs separately for group-based validation
groups = df["subject"].copy()


# Separate labels
y = df["label"].copy()


# Keep only numerical feature columns
X = df.drop(
    columns=["subject", "label"]
).copy()


# Convert labels to numerical values
label_mapping = {
    "control": 0,
    "condition": 1
}

y = y.map(label_mapping)


print("\n========================================")
print("DATASET SUMMARY")
print("========================================")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"groups shape: {groups.shape}")

print("\nLabel mapping:")
print("control   -> 0")
print("condition -> 1")

print("\nClass distribution:")
print(y.value_counts().sort_index())

print("\nFeature columns:")
for column in X.columns:
    print(f"- {column}")


# Verify that everything is valid
print("\n========================================")
print("VALIDATION")
print("========================================")

print(f"Missing values in X: {X.isna().sum().sum()}")
print(f"Missing values in y: {y.isna().sum()}")
print(f"Duplicate subjects: {groups.duplicated().sum()}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of subjects: {len(groups)}")


# Final preview
print("\nFirst 5 feature rows:")
print(X.head().to_string(index=False))

print("\nFirst 5 labels:")
print(y.head().to_string(index=False))

print("\nFirst 5 subject IDs:")
print(groups.head().to_string(index=False))


print("\n========================================")
print("ML DATASET PREPARATION COMPLETE")
print("========================================")