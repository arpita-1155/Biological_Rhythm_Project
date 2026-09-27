import zipfile
import pandas as pd


ZIP_PATH = "data/Depresjon.zip"

subjects_to_check = [
    "control_1",
    "control_6",
    "control_7",
    "control_31",
    "control_32"
]

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    for subject in subjects_to_check:

        file_path = f"data/control/{subject}.csv"

        with z.open(file_path) as f:
            df = pd.read_csv(f)

        timestamps = pd.to_datetime(df["timestamp"])

        differences = timestamps.diff()

        gaps = differences[differences > pd.Timedelta(minutes=1)]

        print("\n====================================")
        print(subject)
        print("====================================")

        for index in gaps.index:

            previous_time = timestamps.loc[index - 1]
            current_time = timestamps.loc[index]

            print("Previous:", previous_time)
            print("Current :", current_time)
            print("Gap     :", current_time - previous_time)