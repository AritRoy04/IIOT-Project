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

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

axes = axes.flatten()

colors = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2"
]

for i, sensor in enumerate(top5):

    axes[i].hist(
        normalized[sensor],
        bins=30,
        color=colors[i],
        edgecolor="black",
        alpha=0.8
    )

    axes[i].set_title(sensor.replace("_", " ").title())
    axes[i].set_xlabel("Normalized Value")
    axes[i].set_ylabel("Frequency")
    axes[i].grid(alpha=0.3)

axes[-1].axis("off")

plt.suptitle(
    "Distribution of Top 5 Informative Sensors",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "sensor_histograms.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()