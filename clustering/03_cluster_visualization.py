import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "kmeans_clusters.csv"

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["RUL", "Cluster", "Cluster_Name"])

pca = PCA(n_components=2)

components = pca.fit_transform(X)

plt.figure(figsize=(10,8))

scatter = plt.scatter(
    components[:,0],
    components[:,1],
    c=df["Cluster"],
    cmap="viridis",
    s=8,
    alpha=0.7
)

plt.title("K-Means Cluster Visualization using PCA", fontsize=18, fontweight="bold")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")

plt.colorbar(scatter, label="Cluster")

plt.grid(True)

plt.tight_layout()

plt.show()

print("="*60)
print("Cluster Visualization Completed")
print("="*60)

print("\nExplained Variance")

print(f"PC1 : {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"PC2 : {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Total : {sum(pca.explained_variance_ratio_)*100:.2f}%")