import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "scaled_features.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["RUL"])

dbscan = DBSCAN(
    eps=2.5,
    min_samples=20
)

clusters = dbscan.fit_predict(X)

df["DBSCAN_Cluster"] = clusters

output_file = OUTPUT_PATH / "dbscan_clusters.csv"

df.to_csv(output_file, index=False)

num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
num_noise = list(clusters).count(-1)

print("=" * 60)
print("DBSCAN Clustering Completed")
print("=" * 60)

print(f"\nDataset Shape : {df.shape}")

print(f"\nNumber of Clusters : {num_clusters}")
print(f"Noise Points : {num_noise}")

print("\nCluster Distribution\n")
print(df["DBSCAN_Cluster"].value_counts().sort_index())

print(f"\nSaved to:\n{output_file}")