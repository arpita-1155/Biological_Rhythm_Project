import zipfile
import pandas as pd
from pathlib import Path


ZIP_PATH = "data/Depresjon.zip"
OUTPUT_DIR = Path("results/cleaned_depresjon")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_subject(z, file_path, label):

    subject = Path(file_path).stem

    print(f"\nProcessing: {subject} | Label: {label}")

    # Read CSV from ZIP
    with z.open(file_path) as f:
        df = pd.read_csv(f)

    # --------------------------------------------------
    # 1. Convert timestamp to datetime
    # --------------------------------------------------
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 2. Remove rows with invalid timestamps
    # --------------------------------------------------
    invalid_timestamps = df["timestamp"].isna().sum()

    if invalid_timestamps > 0:
        print(f"Removing {invalid_timestamps} invalid timestamps")

        df = df.dropna(subset=["timestamp"])

    # --------------------------------------------------
    # 3. Convert activity to numeric
    # --------------------------------------------------
    df["activity"] = pd.to_numeric(
        df["activity"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 4. Remove rows with missing activity
    # --------------------------------------------------
    missing_activity = df["activity"].isna().sum()

    if missing_activity > 0:
        print(f"Removing {missing_activity} rows with missing activity")

        df = df.dropna(subset=["activity"])

    # --------------------------------------------------
    # 5. Remove negative activity values
    # --------------------------------------------------
    negative_activity = (df["activity"] < 0).sum()

    if negative_activity > 0:
        print(f"Removing {negative_activity} negative activity values")

        df = df[df["activity"] >= 0]

    # --------------------------------------------------
    # 6. Sort chronologically
    # --------------------------------------------------
    df = df.sort_values("timestamp")

    # --------------------------------------------------
    # 7. Remove duplicate timestamps
    # --------------------------------------------------
    duplicate_timestamps = df["timestamp"].duplicated().sum()

    if duplicate_timestamps > 0:
        print(f"Removing {duplicate_timestamps} duplicate timestamps")

        df = df.drop_duplicates(
            subset="timestamp",
            keep="first"
        )

    # --------------------------------------------------
    # 8. Add subject and label
    # --------------------------------------------------
    df["subject"] = subject
    df["label"] = label

    # --------------------------------------------------
    # 9. Reorder columns
    # --------------------------------------------------
    df = df[
        [
            "subject",
            "label",
            "timestamp",
            "date",
            "activity"
        ]
    ]

    # --------------------------------------------------
    # 10. Save cleaned subject
    # --------------------------------------------------
    output_file = OUTPUT_DIR / f"{subject}.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print(f"Saved: {output_file}")
    print(f"Rows after preprocessing: {len(df)}")


# ======================================================
# MAIN PROGRAM
# ======================================================

print("\n========================================")
print("DEPRESJON PREPROCESSING")
print("========================================")


with zipfile.ZipFile(ZIP_PATH, "r") as z:

    files = z.namelist()

    processed = 0

    for file_path in files:

        # -----------------------------
        # Condition subjects
        # -----------------------------
        if (
            file_path.startswith("data/condition/")
            and file_path.endswith(".csv")
        ):

            preprocess_subject(
                z,
                file_path,
                "condition"
            )

            processed += 1

        # -----------------------------
        # Control subjects
        # -----------------------------
        elif (
            file_path.startswith("data/control/")
            and file_path.endswith(".csv")
        ):

            preprocess_subject(
                z,
                file_path,
                "control"
            )

            processed += 1


print("\n========================================")
print("PREPROCESSING COMPLETE")
print("========================================")

print(f"Total subjects processed: {processed}")
print(f"Output directory: {OUTPUT_DIR}")