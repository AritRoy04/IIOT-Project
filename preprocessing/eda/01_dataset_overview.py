"""
=========================================================
Industry 5.0 IIoT Predictive Maintenance Project

Phase 3 : Exploratory Data Analysis

File:
01_dataset_overview.py

Purpose:
Generate a complete overview of the NASA C-MAPSS dataset.

Author:
Arit Roy
=========================================================
"""

import pandas as pd
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw"
TRAIN_FILE = DATA_PATH / "train_FD001.txt"

REPORT_PATH = PROJECT_ROOT / "reports"
REPORT_PATH.mkdir(exist_ok=True)



columns = [
    "engine_id",
    "cycle",
    "setting1",
    "setting2",
    "setting3",
]

for i in range(1, 22):
    columns.append(f"sensor_{i}")



df = pd.read_csv(
    TRAIN_FILE,
    sep=r"\s+",
    header=None,
    names=columns
)



total_rows = df.shape[0]
total_columns = df.shape[1]

total_engines = df["engine_id"].nunique()

sensor_columns = len([col for col in df.columns if "sensor" in col])

setting_columns = len([col for col in df.columns if "setting" in col])

missing_values = df.isnull().sum().sum()

duplicate_rows = df.duplicated().sum()

memory_usage = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

engine_lifetime = df.groupby("engine_id")["cycle"].max()

max_cycle = engine_lifetime.max()

min_cycle = engine_lifetime.min()

average_cycle = round(engine_lifetime.mean(), 2)



print("=" * 60)
print("NASA C-MAPSS DATASET OVERVIEW")
print("=" * 60)

print(f"Dataset Shape      : {df.shape}")
print(f"Total Records      : {total_rows}")
print(f"Total Features     : {total_columns}")
print()

print(f"Total Engines      : {total_engines}")
print(f"Sensor Columns     : {sensor_columns}")
print(f"Setting Columns    : {setting_columns}")
print()

print(f"Maximum Lifetime   : {max_cycle}")
print(f"Minimum Lifetime   : {min_cycle}")
print(f"Average Lifetime   : {average_cycle}")
print()

print(f"Missing Values     : {missing_values}")
print(f"Duplicate Rows     : {duplicate_rows}")
print(f"Memory Usage (MB)  : {memory_usage}")

print("=" * 60)
print("DATASET STATUS : READY FOR PREPROCESSING")
print("=" * 60)



report_file = REPORT_PATH / "dataset_overview.txt"

with open(report_file, "w") as f:

    f.write("=" * 60 + "\n")
    f.write("NASA C-MAPSS DATASET OVERVIEW\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Dataset Shape      : {df.shape}\n")
    f.write(f"Total Records      : {total_rows}\n")
    f.write(f"Total Features     : {total_columns}\n\n")

    f.write(f"Total Engines      : {total_engines}\n")
    f.write(f"Sensor Columns     : {sensor_columns}\n")
    f.write(f"Setting Columns    : {setting_columns}\n\n")

    f.write(f"Maximum Lifetime   : {max_cycle}\n")
    f.write(f"Minimum Lifetime   : {min_cycle}\n")
    f.write(f"Average Lifetime   : {average_cycle}\n\n")

    f.write(f"Missing Values     : {missing_values}\n")
    f.write(f"Duplicate Rows     : {duplicate_rows}\n")
    f.write(f"Memory Usage (MB)  : {memory_usage}\n\n")

    f.write("DATASET STATUS : READY FOR PREPROCESSING\n")

print("\nReport Saved Successfully")
print(report_file)