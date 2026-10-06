import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "train_with_rul.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

feature_columns = [
    "cycle",
    "setting1",
    "setting2",
    "setting3"
]

feature_columns.extend(
    [f"sensor_{i}" for i in range(1, 22)]
)

selected_columns = feature_columns + ["RUL"]

selected_df = df[selected_columns]

output_file = OUTPUT_PATH / "selected_features.csv"

selected_df.to_csv(output_file, index=False)

print("=" * 60)
print("Feature Selection Completed")
print("=" * 60)

print(f"\nSelected Features : {len(feature_columns)}")

print("\nFeatures Used\n")

for feature in feature_columns:
    print(feature)

print(f"\nDataset Shape : {selected_df.shape}")

print("\nFirst 5 Rows\n")
print(selected_df.head())

print(f"\nSaved to:\n{output_file}")