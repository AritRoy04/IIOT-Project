import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KMEANS_FILE = PROJECT_ROOT / "data" / "processed" / "kmeans_clusters.csv"
DBSCAN_FILE = PROJECT_ROOT / "data" / "processed" / "dbscan_clusters.csv"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = REPORT_DIR / "clustering_summary.txt"

kmeans_df = pd.read_csv(KMEANS_FILE)
dbscan_df = pd.read_csv(DBSCAN_FILE)

total_samples = len(kmeans_df)

cluster_counts = kmeans_df["Cluster"].value_counts().sort_index()

dbscan_counts = dbscan_df["DBSCAN_Cluster"].value_counts().sort_index()

num_clusters = len(set(dbscan_df["DBSCAN_Cluster"])) - (
    1 if -1 in dbscan_df["DBSCAN_Cluster"].values else 0
)

noise_points = (dbscan_df["DBSCAN_Cluster"] == -1).sum()

report = []

report.append("=" * 60)
report.append("CLUSTERING SUMMARY")
report.append("=" * 60)

report.append(f"\nTotal Samples : {total_samples}")

report.append("\nK-MEANS CLUSTERS")

for cluster, count in cluster_counts.items():
    percent = count / total_samples * 100
    report.append(f"Cluster {cluster} : {count} samples ({percent:.2f}%)")

report.append("\nDBSCAN")

report.append(f"Clusters Detected : {num_clusters}")
report.append(f"Noise Points (-1) : {noise_points}")

report.append("\nDBSCAN Distribution")

for cluster, count in dbscan_counts.items():
    report.append(f"{cluster} : {count}")

report.append("\nObservations")

report.append("- K-Means grouped the machine data into three operating clusters.")
report.append("- DBSCAN detected dense regions and possible anomalies.")
report.append("- PCA visualization showed clear cluster separation.")
report.append("- Clustering helps understand machine behaviour before prediction.")

with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(report))

print("\n".join(report))

print(f"\nSummary saved to:\n{OUTPUT_FILE}")