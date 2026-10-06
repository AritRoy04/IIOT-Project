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
top3 = variance.head(3).index.tolist()

engine_life = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
df = df.merge(engine_life, on="engine_id")

df["life_percent"] = (df["cycle"] / df["max_cycle"]) * 100

scaler = MinMaxScaler()
df[top3] = scaler.fit_transform(df[top3])

bins = list(range(0, 105, 5))

labels = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

df["life_bin"] = pd.cut(
    df["life_percent"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

trend = df.groupby("life_bin")[top3].mean()

trend = trend.rolling(
    window=3,
    center=True,
    min_periods=1
).mean()

plt.figure(figsize=(12,6))

plt.axvspan(0,20,color="green",alpha=0.08,label="Healthy Zone")
plt.axvspan(20,80,color="yellow",alpha=0.08,label="Degradation Zone")
plt.axvspan(80,100,color="red",alpha=0.08,label="Critical Zone")

for sensor in top3:

    plt.plot(
        trend.index.astype(float),
        trend[sensor],
        linewidth=3,
        marker="o",
        markersize=4,
        label=sensor.replace("_"," ").title()
    )

plt.title(
    "Average Degradation Trend of Top Informative Sensors",
    fontsize=15,
    weight="bold"
)

plt.xlabel(
    "Engine Life (%)",
    fontsize=12
)

plt.ylabel(
    "Normalized Sensor Value",
    fontsize=12
)

plt.xlim(0,100)

plt.xticks(range(0,101,10))

plt.grid(True,linestyle="--",alpha=0.4)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_PATH/"engine_health_degradation.png",
    dpi=300
)

plt.show()