import zipfile
import pandas as pd


ZIP_PATH = "data/Depresjon.zip"

results = []

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    files = z.namelist()

    # Process condition subjects
    for file_path in files:

        if file_path.startswith("data/condition/") and file_path.endswith(".csv"):

            label = "condition"
            subject = file_path.split("/")[-1].replace(".csv", "")

        elif file_path.startswith("data/control/") and file_path.endswith(".csv"):

            label = "control"
            subject = file_path.split("/")[-1].replace(".csv", "")

        else:
            continue

        with z.open(file_path) as f:
            df = pd.read_csv(f)

        # Convert timestamp
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")

        # Calculate time differences
        differences = timestamps.sort_values().diff().dropna()

        # Most common sampling interval
        if len(differences) > 0:
            sampling_interval = differences.mode().iloc[0]
        else:
            sampling_interval = None

        # Number of large gaps (> 1 minute)
        large_gaps = (differences > pd.Timedelta(minutes=1)).sum()

        results.append({
            "Subject": subject,
            "Label": label,
            "Rows": len(df),
            "Start": timestamps.min(),
            "End": timestamps.max(),
            "Days": (timestamps.max() - timestamps.min()).total_seconds() / 86400,
            "Missing_Timestamp": df["timestamp"].isna().sum(),
            "Missing_Activity": df["activity"].isna().sum(),
            "Sampling_Interval": sampling_interval,
            "Large_Gaps": large_gaps
        })


# Create final audit table
audit_df = pd.DataFrame(results)

# Sort by label and subject
audit_df = audit_df.sort_values(
    by=["Label", "Subject"]
).reset_index(drop=True)


print("\n===== DEPRESJON DATASET AUDIT =====")
print(audit_df.to_string(index=False))


print("\n===== SUBJECT COUNT =====")
print(audit_df["Label"].value_counts())


print("\n===== TOTAL SUBJECTS =====")
print(len(audit_df))


print("\n===== SAMPLING INTERVAL COUNTS =====")
print(audit_df["Sampling_Interval"].value_counts())


# Save audit result
output_path = "results/depresjon_audit.csv"
audit_df.to_csv(output_path, index=False)

print(f"\nAudit saved to: {output_path}")