#LOAD DATASET

import pandas as pd
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent



DATA_PATH = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = DATA_PATH / "train_FD001.txt"



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


print("=" * 60)
print("NASA C-MAPSS Dataset Loaded Successfully")
print("=" * 60)

print("\nFirst Five Rows\n")
print(df.head())



print("\n")
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()



print("\n")
print("=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())



print("\n")
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())



print("\n")
print("=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"Total Duplicate Rows : {duplicates}")



print("\n")
print("=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)



print("\n")
print("=" * 60)
print("ENGINE ANALYSIS")
print("=" * 60)

print(f"Total Engines : {df['engine_id'].nunique()}")

engine_cycles = df.groupby("engine_id")["cycle"].max()

print(f"Maximum Engine Life : {engine_cycles.max()} cycles")

print(f"Minimum Engine Life : {engine_cycles.min()} cycles")

print(f"Average Engine Life : {engine_cycles.mean():.2f} cycles")


print("\n")
print("=" * 60)
print("SENSOR VARIANCE")
print("=" * 60)

sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

sensor_variance = df[sensor_columns].var()

print(sensor_variance.sort_values(ascending=False))