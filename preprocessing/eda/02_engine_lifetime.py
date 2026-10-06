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

engine_lifetime = df.groupby("engine_id")["cycle"].max()

print("=" * 60)
print("ENGINE LIFETIME ANALYSIS")
print("=" * 60)
print(f"Total Engines      : {len(engine_lifetime)}")
print(f"Maximum Lifetime   : {engine_lifetime.max()} cycles")
print(f"Minimum Lifetime   : {engine_lifetime.min()} cycles")
print(f"Average Lifetime   : {engine_lifetime.mean():.2f} cycles")
print(f"Median Lifetime    : {engine_lifetime.median():.2f} cycles")
print(f"Standard Deviation : {engine_lifetime.std():.2f}")
print("=" * 60)

plt.figure(figsize=(10,6))
plt.hist(engine_lifetime, bins=15)
plt.title("Distribution of Engine Lifetimes")
plt.xlabel("Engine Lifetime (Cycles)")
plt.ylabel("Number of Engines")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_PATH / "engine_lifetime_distribution.png", dpi=300)
plt.show()

top10 = engine_lifetime.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.bar(top10.index.astype(str), top10.values)
plt.title("Top 10 Longest-Living Engines")
plt.xlabel("Engine ID")
plt.ylabel("Lifetime (Cycles)")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(FIGURE_PATH / "top10_longest_engines.png", dpi=300)
plt.show()

bottom10 = engine_lifetime.sort_values().head(10)

plt.figure(figsize=(10,6))
plt.bar(bottom10.index.astype(str), bottom10.values)
plt.title("Top 10 Shortest-Living Engines")
plt.xlabel("Engine ID")
plt.ylabel("Lifetime (Cycles)")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(FIGURE_PATH / "top10_shortest_engines.png", dpi=300)
plt.show()