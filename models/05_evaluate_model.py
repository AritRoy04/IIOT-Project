import os
import pandas as pd
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
    "mamba_predictions.npz"
)

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures"
)

METRICS_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "mamba_evaluation_metrics.txt"
)

os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# Load Predictions
# ============================================================
print("=" * 60)
print("MAMBA RUL MODEL EVALUATION")
print("=" * 60)

print("\nLoading prediction results...")

data = np.load(PREDICTION_PATH)

y_true = data["y_true"]
y_pred = data["y_pred"]

print(f"Actual RUL shape    : {y_true.shape}")
print(f"Predicted RUL shape : {y_pred.shape}")


# ============================================================
# Calculate Evaluation Metrics
# ============================================================
mae = mean_absolute_error(
    y_true,
    y_pred
)

mse = mean_squared_error(
    y_true,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_true,
    y_pred
)


# ============================================================
# Display Metrics
# ============================================================
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


# ============================================================
# Save Metrics
# ============================================================
with open(
    METRICS_PATH,
    "w"
) as file:

    file.write(
        "MAMBA RUL MODEL EVALUATION\n"
    )

    file.write(
        "=" * 40 + "\n"
    )

    file.write(
        f"Total Test Samples : {len(y_true)}\n"
    )

    file.write(
        f"MAE                : {mae:.4f}\n"
    )

    file.write(
        f"MSE                : {mse:.4f}\n"
    )

    file.write(
        f"RMSE               : {rmse:.4f}\n"
    )

    file.write(
        f"R2 Score           : {r2:.4f}\n"
    )


# ============================================================
# Figure 1: Actual vs Predicted RUL Scatter Plot
# ============================================================
plt.figure(
    figsize=(10, 8)
)

plt.scatter(
    y_true,
    y_pred,
    alpha=0.35,
    s=20
)

minimum_value = min(
    y_true.min(),
    y_pred.min()
)

maximum_value = max(
    y_true.max(),
    y_pred.max()
)

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

plt.xlabel(
    "Actual RUL (Cycles)",
    fontsize=12
)

plt.ylabel(
    "Predicted RUL (Cycles)",
    fontsize=12
)

plt.title(
    "Mamba Model: Actual vs Predicted RUL",
    fontsize=16,
    fontweight="bold"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

ACTUAL_VS_PREDICTED_PATH = os.path.join(
    FIGURE_DIR,
    "mamba_actual_vs_predicted.png"
)

plt.savefig(
    ACTUAL_VS_PREDICTED_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Figure 2: Prediction Error Distribution
# ============================================================
errors = y_pred - y_true

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    errors,
    bins=50,
    edgecolor="black",
    alpha=0.8
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Zero Error"
)

plt.xlabel(
    "Prediction Error (Predicted RUL - Actual RUL)",
    fontsize=12
)

plt.ylabel(
    "Frequency",
    fontsize=12
)

plt.title(
    "Distribution of Mamba RUL Prediction Errors",
    fontsize=16,
    fontweight="bold"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

ERROR_DISTRIBUTION_PATH = os.path.join(
    FIGURE_DIR,
    "mamba_error_distribution.png"
)

plt.savefig(
    ERROR_DISTRIBUTION_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Figure 3: Smoothed Actual vs Predicted RUL
# ============================================================

from scipy.signal import savgol_filter

# Sort predictions according to actual RUL
sorted_idx = np.argsort(y_true)

y_true_sorted = y_true[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

# ------------------------------------------------------------
# Savitzky-Golay Smoothing
# ------------------------------------------------------------
window_length = 41     # Must be odd
poly_order = 3

y_true_smooth = savgol_filter(
    y_true_sorted,
    window_length,
    poly_order
)

y_pred_smooth = savgol_filter(
    y_pred_sorted,
    window_length,
    poly_order
)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(16, 7))

plt.plot(
    y_true_smooth,
    color="#0057B8",
    linewidth=2.6,
    label="Actual RUL"
)

plt.plot(
    y_pred_smooth,
    color="#FF6F00",
    linestyle="--",
    linewidth=2.6,
    label="Predicted RUL"
)

plt.xlabel(
    "Sorted Test Samples",
    fontsize=15
)

plt.ylabel(
    "Remaining Useful Life (Cycles)",
    fontsize=15
)

plt.title(
    "Actual vs Predicted Remaining Useful Life",
    fontsize=18,
    fontweight="bold"
)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(
    fontsize=13,
    loc="upper left",
    frameon=True
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.18
)

# Remove extra white margins
plt.margins(x=0)

plt.tight_layout()

PREDICTION_COMPARISON_PATH = os.path.join(
    FIGURE_DIR,
    "mamba_prediction_comparison.png"
)

plt.savefig(
    PREDICTION_COMPARISON_PATH,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "mamba_prediction_comparison.pdf"
    ),
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print(f"Metrics saved to:")
print(METRICS_PATH)

print("\nFigures saved to:")
print(ACTUAL_VS_PREDICTED_PATH)
print(ERROR_DISTRIBUTION_PATH)
print(PREDICTION_COMPARISON_PATH)