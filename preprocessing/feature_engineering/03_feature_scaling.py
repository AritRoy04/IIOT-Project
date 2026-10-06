import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "selected_features.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["RUL"])
y = df["RUL"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

scaled_df["RUL"] = y

output_file = OUTPUT_PATH / "scaled_features.csv"

scaled_df.to_csv(output_file, index=False)

print("=" * 60)
print("Feature Scaling Completed")
print("=" * 60)

print(f"\nDataset Shape : {scaled_df.shape}")

print("\nFeature Means\n")
print(scaled_df.drop(columns=["RUL"]).mean().round(4))

print("\nFeature Standard Deviations\n")
print(scaled_df.drop(columns=["RUL"]).std().round(4))

print("\nFirst 5 Rows\n")
print(scaled_df.head())

print(f"\nSaved to:\n{output_file}")