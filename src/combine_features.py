import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")

OUTPUT_FILE = RESULTS_DIR / "feature_table.csv"


# Feature files to combine
FEATURE_FILES = [
    "cosinor_features.csv",
    "wavelet_features.csv",
    "sample_entropy_features.csv",
    "permutation_entropy_features.csv",
    "dfa_features.csv"
]


print("\n========================================")
print("COMBINING FEATURE TABLES")
print("========================================")


# Start with Cosinor features
first_file = RESULTS_DIR / FEATURE_FILES[0]

combined_df = pd.read_csv(first_file)

print(f"\nStarting with: {FEATURE_FILES[0]}")
print(f"Rows: {len(combined_df)}")
print(f"Columns: {len(combined_df.columns)}")


# Merge remaining feature files
for file_name in FEATURE_FILES[1:]:

    file_path = RESULTS_DIR / file_name

    print(f"\nMerging: {file_name}")

    df = pd.read_csv(file_path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Keep label only from the first file
    df = df.drop(columns=["label"], errors="ignore")

    combined_df = combined_df.merge(
        df,
        on="subject",
        how="inner"
    )


# Sort by subject
combined_df = combined_df.sort_values(
    "subject"
).reset_index(drop=True)


print("\n========================================")
print("VALIDATION")
print("========================================")


# 1. Number of subjects
print(f"\nTotal subjects: {len(combined_df)}")


# 2. Duplicate subjects
duplicate_count = combined_df["subject"].duplicated().sum()

print(f"Duplicate subjects: {duplicate_count}")


# 3. Labels
print("\nClass distribution:")
print(combined_df["label"].value_counts())


# 4. Missing values
print("\nMissing values:")
print(combined_df.isna().sum())


# 5. Number of features
feature_columns = [
    column
    for column in combined_df.columns
    if column not in ["subject", "label"]
]

print(f"\nTotal feature columns: {len(feature_columns)}")

print("\nFeature columns:")
for column in feature_columns:
    print(f"- {column}")


# Save combined table
combined_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n========================================")
print("FEATURE COMBINATION COMPLETE")
print("========================================")

print(f"Final shape: {combined_df.shape}")

print(f"\nSaved to: {OUTPUT_FILE}")

print("\nFirst 5 rows:")
print(combined_df.head().to_string(index=False))