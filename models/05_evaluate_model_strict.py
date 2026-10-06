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
    "mamba_predictions_strict.npz"
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
    "mamba_evaluation_metrics_strict.txt"
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)


# ============================================================
# Header
# ============================================================
print("=" * 60)
print("MAMBA RUL MODEL EVALUATION - STRICT ENGINE-WISE SPLIT")
print("=" * 60)


# ============================================================
# Load Predictions
# ============================================================
print("\nLoading strict prediction results...")

data = np.load(
    PREDICTION_PATH
)

actual_rul = data[
    "actual_rul"
]

predicted_rul = data[
    "predicted_rul"
]

print(
    f"Actual RUL shape    : "
    f"{actual_rul.shape}"
)

print(
    f"Predicted RUL shape : "
    f"{predicted_rul.shape}"
)


# ============================================================
# Sanity Checks
# ============================================================
if len(actual_rul) != len(predicted_rul):
    raise ValueError(
        "Actual and predicted RUL lengths do not match."
    )

if not np.all(
    np.isfinite(actual_rul)
):
    raise ValueError(
        "Actual RUL contains NaN or infinite values."
    )

if not np.all(
    np.isfinite(predicted_rul)
):
    raise ValueError(
        "Predicted RUL contains NaN or infinite values."
    )


# ============================================================
# Calculate Evaluation Metrics
# ============================================================
mae = mean_absolute_error(
    actual_rul,
    predicted_rul
)

mse = mean_squared_error(
    actual_rul,
    predicted_rul
)

rmse = np.sqrt(
    mse
)

r2 = r2_score(
    actual_rul,
    predicted_rul
)


# ============================================================
# Additional Error Statistics
# ============================================================
errors = (
    predicted_rul
    - actual_rul
)

mean_error = np.mean(
    errors
)

error_std = np.std(
    errors
)


# ============================================================
# Print Results
# ============================================================
print("\n" + "=" * 60)
print("FINAL STRICT EVALUATION RESULTS")
print("=" * 60)

print(
    f"MAE        : "
    f"{mae:.4f}"
)

print(
    f"MSE        : "
    f"{mse:.4f}"
)

print(
    f"RMSE       : "
    f"{rmse:.4f}"
)

print(
    f"R2         : "
    f"{r2:.4f}"
)

print(
    f"Mean Error : "
    f"{mean_error:.4f}"
)

print(
    f"Error STD  : "
    f"{error_std:.4f}"
)


# ============================================================
# Save Metrics
# ============================================================
with open(
    METRICS_PATH,
    "w"
) as file:

    file.write(
        "MAMBA RUL MODEL - FINAL STRICT ENGINE-WISE EVALUATION\n"
    )

    file.write(
        "=" * 60
        + "\n\n"
    )

    file.write(
        "Evaluation Protocol\n"
    )

    file.write(
        "-------------------\n"
    )

    file.write(
        "Training engines   : 64\n"
    )

    file.write(
        "Validation engines : 16\n"
    )

    file.write(
        "Testing engines    : 20\n"
    )

    file.write(
        "Train/Validation overlap : 0\n"
    )

    file.write(
        "Train/Test overlap       : 0\n"
    )

    file.write(
        "Validation/Test overlap  : 0\n\n"
    )

    file.write(
        "Final Evaluation Metrics\n"
    )

    file.write(
        "------------------------\n"
    )

    file.write(
        f"MAE        : {mae:.4f}\n"
    )

    file.write(
        f"MSE        : {mse:.4f}\n"
    )

    file.write(
        f"RMSE       : {rmse:.4f}\n"
    )

    file.write(
        f"R2         : {r2:.4f}\n"
    )

    file.write(
        f"Mean Error : {mean_error:.4f}\n"
    )

    file.write(
        f"Error STD  : {error_std:.4f}\n"
    )


# ============================================================
# Figure 1: Actual vs Predicted RUL
# ============================================================
plt.figure(
    figsize=(8, 6)
)

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

plt.xlabel(
    "Actual RUL"
)

plt.ylabel(
    "Predicted RUL"
)

plt.title(
    "Mamba RUL Prediction "
    "- Strict Engine-Wise Split"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "mamba_actual_vs_predicted_strict.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# Figure 2: Prediction Error Distribution
# ============================================================
plt.figure(
    figsize=(8, 6)
)

plt.hist(
    errors,
    bins=50,
    edgecolor="black"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Prediction Error "
    "(Predicted RUL - Actual RUL)"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Mamba Prediction Error Distribution "
    "- Strict Split"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "mamba_error_distribution_strict.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# Figure 3: Actual vs Predicted Sequence Comparison
# ============================================================
sample_count = min(
    200,
    len(actual_rul)
)

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    actual_rul[:sample_count],
    label="Actual RUL"
)

plt.plot(
    predicted_rul[:sample_count],
    label="Predicted RUL"
)

plt.xlabel(
    "Test Sequence"
)

plt.ylabel(
    "RUL"
)

plt.title(
    "Actual vs Predicted RUL "
    "- Strict Engine-Wise Split"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "mamba_prediction_comparison_strict.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# Figure 4: Residual Plot
# ============================================================
plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    predicted_rul,
    errors,
    alpha=0.4
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted RUL"
)

plt.ylabel(
    "Residual Error"
)

plt.title(
    "Mamba Residual Analysis "
    "- Strict Engine-Wise Split"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "mamba_residual_plot_strict.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 60)
print("FINAL STRICT EVALUATION COMPLETED")
print("=" * 60)

print(
    f"\nMetrics saved to:\n"
    f"{METRICS_PATH}"
)

print(
    f"\nFigures saved to:\n"
    f"{FIGURE_DIR}"
)