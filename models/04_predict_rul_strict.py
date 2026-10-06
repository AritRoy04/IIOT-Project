import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Project Path
# ============================================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from models.mamba_model import MambaRULModel


# ============================================================
# Configuration
# ============================================================
BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Paths
# ============================================================
X_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "X_test_strict.npy"
)

Y_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "y_test_strict.npy"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "models",
    "best_mamba_rul_model_strict.pth"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "mamba_predictions_strict.npz"
)


# ============================================================
# Header
# ============================================================
print("=" * 60)
print("MAMBA RUL PREDICTION - STRICT ENGINE-WISE SPLIT")
print("=" * 60)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# Load Strict Test Data
# ============================================================
print("\nLoading strict test data...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")

X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)


# ============================================================
# Create Test Dataset and DataLoader
# ============================================================
X_tensor = torch.from_numpy(X_test)
y_tensor = torch.from_numpy(y_test)

test_dataset = TensorDataset(
    X_tensor,
    y_tensor
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Load Best Strict Model
# ============================================================
print("\nLoading best strict Mamba model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model = MambaRULModel().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Best model loaded from epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation loss: "
    f"{checkpoint['best_val_loss']:.4f}"
)

print(
    f"Split type: "
    f"{checkpoint.get('split_type', 'unknown')}"
)


# ============================================================
# Generate Predictions
# ============================================================
print("\nGenerating predictions on unseen test engines...")

all_predictions = []
all_actual = []

with torch.no_grad():

    for batch_X, batch_y in test_loader:

        batch_X = batch_X.to(DEVICE)

        predictions = model(batch_X)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_actual.extend(
            batch_y.numpy()
        )


# ============================================================
# Convert to NumPy Arrays
# ============================================================
predicted_rul = np.array(
    all_predictions,
    dtype=np.float32
)

actual_rul = np.array(
    all_actual,
    dtype=np.float32
)


# ============================================================
# Sanity Checks
# ============================================================
assert len(predicted_rul) == len(actual_rul)

assert np.all(
    np.isfinite(predicted_rul)
), "Predictions contain NaN or infinite values."


# ============================================================
# Save Predictions
# ============================================================
np.savez(
    OUTPUT_PATH,
    actual_rul=actual_rul,
    predicted_rul=predicted_rul
)


# ============================================================
# Sample Predictions
# ============================================================
print("\nSample predictions:\n")

for i in range(
    min(10, len(actual_rul))
):

    print(
        f"Sample {i + 1:02d} | "
        f"Actual RUL: {actual_rul[i]:7.2f} | "
        f"Predicted RUL: {predicted_rul[i]:7.2f}"
    )


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("STRICT ENGINE-WISE PREDICTION COMPLETED")
print("=" * 60)

print(
    f"Total predictions : "
    f"{len(predicted_rul)}"
)

print(
    f"Predictions saved : "
    f"{OUTPUT_PATH}"
)