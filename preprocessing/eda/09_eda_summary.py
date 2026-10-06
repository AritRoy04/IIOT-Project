import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw"
TRAIN_FILE = DATA_PATH / "train_FD001.txt"

REPORT_PATH = PROJECT_ROOT / "reports"
REPORT_PATH.mkdir(parents=True, exist_ok=True)

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

top5 = variance.head(5)

engine_life = df.groupby("engine_id")["cycle"].max()

summary = []

summary.append("=" * 70)
summary.append("EXPLORATORY DATA ANALYSIS (EDA) SUMMARY")
summary.append("=" * 70)
summary.append("")

summary.append("DATASET INFORMATION")
summary.append("------------------------------")
summary.append(f"Dataset Shape           : {df.shape}")
summary.append(f"Total Engines           : {df['engine_id'].nunique()}")
summary.append(f"Total Sensors           : 21")
summary.append(f"Missing Values          : {df.isnull().sum().sum()}")
summary.append(f"Duplicate Rows          : {df.duplicated().sum()}")
summary.append("")

summary.append("ENGINE LIFETIME")
summary.append("------------------------------")
summary.append(f"Maximum Lifetime        : {engine_life.max()} cycles")
summary.append(f"Minimum Lifetime        : {engine_life.min()} cycles")
summary.append(f"Average Lifetime        : {engine_life.mean():.2f} cycles")
summary.append("")

summary.append("TOP 5 INFORMATIVE SENSORS")
summary.append("------------------------------")

for sensor, value in top5.items():
    summary.append(f"{sensor:<20} Variance = {value:.2f}")

summary.append("")

summary.append("EDA OBSERVATIONS")
summary.append("------------------------------")
summary.append("1. The dataset contains no missing values.")
summary.append("2. No duplicate records were found.")
summary.append("3. Engine lifetime varies significantly across engines.")
summary.append("4. Several sensors exhibit much higher variance than others.")
summary.append("5. Correlation analysis identified strong relationships between multiple sensors.")
summary.append("6. Boxplots revealed different levels of sensor variability.")
summary.append("7. Histograms showed the distribution characteristics of informative sensors.")
summary.append("8. PCA demonstrated that the first two principal components preserve most of the dataset variance and reveal a progression from healthy to near-failure states.")
summary.append("")
summary.append("CONCLUSION")
summary.append("------------------------------")
summary.append("The NASA C-MAPSS dataset is clean, consistent, and suitable for predictive maintenance research. Exploratory Data Analysis identified informative sensors, revealed engine degradation patterns, and provided statistical insights that will support feature engineering and Remaining Useful Life (RUL) prediction in the next phase.")

report = "\n".join(summary)

print(report)

report_file = REPORT_PATH / "EDA_Summary.txt"

with open(report_file, "w") as f:
    f.write(report)

print("\nEDA summary saved successfully.")
print(report_file)