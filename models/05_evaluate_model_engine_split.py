import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PREDICTION_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "mamba_predictions_engine_split.npz"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

FIGURE_DIR = os.path.join(
    RESULTS_DIR,
    "figures"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "mamba_evaluation_metrics_engine_split.txt"
)

os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# Header
# ============================================================
print("=" * 60)
print("MAMBA RUL MODEL EVALUATION - ENGINE-WISE SPLIT")
print("=" * 60)


# ============================================================
# Load Predictions
# ============================================================
print("\nLoading engine-wise prediction results...")

data = np.load(PREDICTION_PATH)

actual_rul = data["actual_rul"]
predicted_rul = data["predicted_rul"]

print(f"Actual RUL shape    : {actual_rul.shape}")
print(f"Predicted RUL shape : {predicted_rul.shape}")


# ============================================================
# Calculate Metrics
# ============================================================
mae = mean_absolute_error(
    actual_rul,
    predicted_rul
)

mse = mean_squared_error(
    actual_rul,
    predicted_rul
)

rmse = np.sqrt(mse)

r2 = r2_score(
    actual_rul,
    predicted_rul
)


# ============================================================
# Print Results
# ============================================================
print("\n" + "=" * 60)
print("ENGINE-WISE EVALUATION RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


# ============================================================
# Save Metrics
# ============================================================
with open(METRICS_PATH, "w") as file:

    file.write(
        "MAMBA RUL MODEL - ENGINE-WISE SPLIT\n"
    )

    file.write("=" * 50 + "\n\n")

    file.write(
        "Evaluation Protocol:\n"
        "80 training engines\n"
        "20 completely unseen testing engines\n"
        "Engine overlap: 0\n\n"
    )

    file.write(f"MAE  : {mae:.4f}\n")
    file.write(f"MSE  : {mse:.4f}\n")
    file.write(f"RMSE : {rmse:.4f}\n")
    file.write(f"R2   : {r2:.4f}\n")


# ============================================================
# Figure 1: Actual vs Predicted
# ============================================================
plt.figure(figsize=(8, 6))

plt.scatter(
    actual_rul,
    predicted_rul,
    alpha=0.4
)

minimum = min(
    actual_rul.min(),
    predicted_rul.min()
)

maximum = max(
    actual_rul.max(),
    predicted_rul.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")

plt.title(
    "Mamba RUL Prediction - Engine-Wise Split"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "mamba_actual_vs_predicted_engine_split.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# Figure 2: Error Distribution
# ============================================================
errors = predicted_rul - actual_rul

plt.figure(figsize=(8, 6))

plt.hist(
    errors,
    bins=50,
    edgecolor="black"
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.title(
    "Mamba Prediction Error Distribution "
    "- Engine-Wise Split"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "mamba_error_distribution_engine_split.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# Figure 3: Prediction Comparison
# ============================================================
sample_count = min(
    200,
    len(actual_rul)
)

plt.figure(figsize=(12, 6))

plt.plot(
    actual_rul[:sample_count],
    label="Actual RUL"
)

plt.plot(
    predicted_rul[:sample_count],
    label="Predicted RUL"
)

plt.xlabel("Test Sequence")
plt.ylabel("RUL")

plt.title(
    "Actual vs Predicted RUL "
    "- Engine-Wise Split"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "mamba_prediction_comparison_engine_split.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("ENGINE-WISE EVALUATION COMPLETED")
print("=" * 60)

print(f"\nMetrics saved to:\n{METRICS_PATH}")

print(
    f"\nFigures saved to:\n"
    f"{FIGURE_DIR}"
)