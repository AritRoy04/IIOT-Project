import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

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

top5 = variance.head(5).index.tolist()

scaler = MinMaxScaler()

normalized = pd.DataFrame(
    scaler.fit_transform(df[top5]),
    columns=top5
)

data = [normalized[sensor] for sensor in top5]

fig, ax = plt.subplots(figsize=(10,6))

box = ax.boxplot(
    data,
    tick_labels=[s.replace("_", " ").title() for s in top5],
    patch_artist=True,
    showfliers=True,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black"),
    flierprops=dict(
        marker="o",
        markerfacecolor="red",
        markeredgecolor="red",
        markersize=4,
        alpha=0.5
    )
)

colors = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2"
]

for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

ax.set_title(
    "Normalized Distribution of Top 5 Informative Sensors",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Sensors",
    fontsize=12
)

ax.set_ylabel(
    "Normalized Sensor Value",
    fontsize=12
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "sensor_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()