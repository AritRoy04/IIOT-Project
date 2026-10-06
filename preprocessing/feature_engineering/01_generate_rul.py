import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw"
TRAIN_FILE = DATA_PATH / "train_FD001.txt"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

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

max_cycle = (
    df.groupby("engine_id")["cycle"]
    .max()
    .reset_index()
    .rename(columns={"cycle": "max_cycle"})
)

df = df.merge(max_cycle, on="engine_id")

df["RUL"] = df["max_cycle"] - df["cycle"]

df = df.drop(columns=["max_cycle"])

output_file = OUTPUT_PATH / "train_with_rul.csv"

df.to_csv(output_file, index=False)

print("=" * 60)
print("RUL Generation Completed")
print("=" * 60)

print(f"\nDataset Shape : {df.shape}")
print(f"Maximum RUL   : {df['RUL'].max()}")
print(f"Minimum RUL   : {df['RUL'].min()}")

print("\nFirst 10 Rows\n")
print(df[["engine_id", "cycle", "RUL"]].head(10))

print(f"\nSaved to:\n{output_file}")