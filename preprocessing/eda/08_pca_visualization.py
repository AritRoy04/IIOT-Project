import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw"
TRAIN_FILE = DATA_PATH / "train_FD001.txt"

FIGURE_PATH = PROJECT_ROOT / "results" / "figures"
FIGURE_PATH.mkdir(parents=True, exist_ok=True)

columns = [
    "engine_id",
    "cycle",
    "setting1",
    "setting2",
    "setting3",
]

for i in range(1,22):
    columns.append(f"sensor_{i}")

df = pd.read_csv(
    TRAIN_FILE,
    sep=r"\s+",
    header=None,
    names=columns
)

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

variance = df[sensor_cols].var().sort_values(ascending=False)

top10 = variance.head(10).index.tolist()

engine_life = df.groupby("engine_id")["cycle"].max().rename("max_cycle")

df = df.merge(engine_life,on="engine_id")

df["life_percent"] = (df["cycle"]/df["max_cycle"])*100

bins=[0,20,40,60,80,100]

labels=[
    "Healthy",
    "Early Wear",
    "Moderate Wear",
    "Severe Wear",
    "Near Failure"
]

df["stage"]=pd.cut(
    df["life_percent"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

X=df[top10]

scaler=StandardScaler()

X_scaled=scaler.fit_transform(X)

pca=PCA(n_components=2)

X_pca=pca.fit_transform(X_scaled)

pca_df=pd.DataFrame(
    X_pca,
    columns=["PC1","PC2"]
)

pca_df["Stage"]=df["stage"].values

colors={
    "Healthy":"green",
    "Early Wear":"gold",
    "Moderate Wear":"orange",
    "Severe Wear":"red",
    "Near Failure":"purple"
}

plt.figure(figsize=(10,7))

for stage,color in colors.items():

    subset=pca_df[pca_df["Stage"]==stage]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        s=12,
        alpha=0.5,
        color=color,
        label=stage
    )

plt.title(
    "PCA Visualization of Engine Health Stages",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel(
    f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)"
)

plt.ylabel(
    f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)"
)

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_PATH/"pca_visualization.png",
    dpi=300
)

plt.show()