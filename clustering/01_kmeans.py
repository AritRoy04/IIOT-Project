import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "scaled_features.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["RUL"])

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

df["Cluster"] = clusters

cluster_names = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2"
}

df["Cluster_Name"] = df["Cluster"].map(cluster_names)

output_file = OUTPUT_PATH / "kmeans_clusters.csv"

df.to_csv(output_file, index=False)

print("=" * 60)
print("K-Means Clustering Completed")
print("=" * 60)

print(f"\nDataset Shape : {df.shape}")

print("\nCluster Distribution\n")
print(df["Cluster_Name"].value_counts())

print("\nCluster Centers Shape")
print(kmeans.cluster_centers_.shape)

print(f"\nSaved to:\n{output_file}")