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

sensor_cols = [col for col in df.columns if "sensor" in col]

variance = df[sensor_cols].var().sort_values(ascending=False)

print("=" * 60)
print("SENSOR VARIANCE ANALYSIS")
print("=" * 60)

for sensor, value in variance.items():
    print(f"{sensor:<10} : {value:.4f}")

print("=" * 60)
print("TOP INFORMATIVE SENSORS IDENTIFIED")
print("=" * 60)

plt.figure(figsize=(12,6))
variance.plot(kind="bar")
plt.title("Sensor Variance Analysis")
plt.xlabel("Sensors")
plt.ylabel("Variance")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(FIGURE_PATH / "sensor_variance.png", dpi=300)
plt.show()