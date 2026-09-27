import pandas as pd
from pathlib import Path


# =====================================================
# SETTINGS
# =====================================================

INPUT_DIR = Path("results/cleaned_depresjon")
OUTPUT_DIR = Path("results/activity_5min")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# PROCESS ONE SUBJECT
# =====================================================

def prepare_subject(file_path):

    subject = file_path.stem

    print(f"\nProcessing: {subject}")

    # -----------------------------------------------
    # Read cleaned data
    # -----------------------------------------------

    df = pd.read_csv(file_path)

    # -----------------------------------------------
    # Convert timestamp
    # -----------------------------------------------

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # -----------------------------------------------
    # Make timestamp the index
    # -----------------------------------------------

    df = df.set_index("timestamp")

    # -----------------------------------------------
    # 5-minute aggregation
    # -----------------------------------------------
    #
    # Mean activity within every 5-minute interval.
    #
    # We use mean rather than sum because it keeps
    # the activity value on a comparable scale.
    # -----------------------------------------------

    activity_5min = df["activity"].resample("5min").mean()

    # -----------------------------------------------
    # Create output dataframe
    # -----------------------------------------------

    result = activity_5min.reset_index()

    result["subject"] = df["subject"].iloc[0]
    result["label"] = df["label"].iloc[0]

    # -----------------------------------------------
    # Reorder columns
    # -----------------------------------------------

    result = result[
        [
            "subject",
            "label",
            "timestamp",
            "activity"
        ]
    ]

    # -----------------------------------------------
    # Save
    # -----------------------------------------------

    output_file = OUTPUT_DIR / f"{subject}.csv"

    result.to_csv(
        output_file,
        index=False
    )

    print(f"Saved: {output_file}")
    print(f"Original rows: {len(df)}")
    print(f"5-minute rows: {len(result)}")


# =====================================================
# MAIN PROGRAM
# =====================================================

print("\n========================================")
print("5-MINUTE ACTIVITY PREPARATION")
print("========================================")


files = sorted(INPUT_DIR.glob("*.csv"))

print(f"Subjects found: {len(files)}")


for file_path in files:

    prepare_subject(file_path)


print("\n========================================")
print("5-MINUTE PREPARATION COMPLETE")
print("========================================")

print(f"Output directory: {OUTPUT_DIR}")