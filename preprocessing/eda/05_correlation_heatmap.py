import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

for i in range(1, 22):
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

corr = df[top10].corr()

plt.figure(figsize=(10,8))

img = plt.imshow(
    corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(img, label="Correlation Coefficient")

plt.xticks(
    range(len(top10)),
    [s.replace("_"," ") for s in top10],
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(top10)),
    [s.replace("_"," ") for s in top10]
)

for i in range(len(top10)):
    for j in range(len(top10)):
        plt.text(
            j,
            i,
            f"{corr.iloc[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="black"
        )

plt.title(
    "Correlation Heatmap of Top 10 Informative Sensors",
    fontsize=15,
    weight="bold"
)

plt.tight_layout()

plt.savefig(
    FIGURE_PATH/"correlation_heatmap.png",
    dpi=300
)

plt.show()