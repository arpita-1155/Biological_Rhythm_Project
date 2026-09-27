import zipfile
import pandas as pd

ZIP_PATH = "data/Depresjon.zip"

# Inspect one condition subject
# FILE_TO_INSPECT = "data/condition/condition_1.csv"

# Inspect one control subject
FILE_TO_INSPECT = "data/control/control_1.csv"

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    print("\n===== INSPECTING FILE =====")
    print(FILE_TO_INSPECT)

    with z.open(FILE_TO_INSPECT) as f:
        df = pd.read_csv(f)

    print("\n===== SHAPE =====")
    print(df.shape)

    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== FIRST 10 ROWS =====")
    print(df.head(10).to_string())

    print("\n===== LAST 10 ROWS =====")
    print(df.tail(10).to_string())

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== BASIC STATISTICS =====")
    print(df.describe())

    # If timestamp exists, inspect it
    if "timestamp" in df.columns:

        print("\n===== TIMESTAMP INFORMATION =====")

        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")

        print("First timestamp:", timestamps.min())
        print("Last timestamp:", timestamps.max())

        print("\n===== TIME DIFFERENCES =====")

        differences = timestamps.diff().dropna()

        print(differences.value_counts().head(10))