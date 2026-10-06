import os
import sys
import numpy as np
import torch

# ============================================================
# Project Setup
# ============================================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from models.mamba_model import MambaRULModel


# ============================================================
# Configuration and Paths
# ============================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

X_TEST_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "X_test.npy"
)

Y_TEST_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "y_test.npy"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "models",
    "best_mamba_rul_model.pth"
)

PREDICTION_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "mamba_predictions.npz"
)


# ============================================================
# Load Test Data
# ============================================================
print("=" * 60)
print("MAMBA RUL PREDICTION")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading test data...")

X_test = np.load(X_TEST_PATH).astype(np.float32)
y_test = np.load(Y_TEST_PATH).astype(np.float32)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# ============================================================
# Load Trained Mamba Model
# ============================================================
print("\nLoading best trained model...")

model = MambaRULModel().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

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


# ============================================================
# Generate Predictions
# ============================================================
print("\nGenerating RUL predictions...")

X_tensor = torch.from_numpy(
    X_test
).to(DEVICE)

predictions = []

BATCH_SIZE = 256

with torch.no_grad():

    for start in range(
        0,
        len(X_tensor),
        BATCH_SIZE
    ):

        end = start + BATCH_SIZE

        batch_X = X_tensor[start:end]

        batch_predictions = model(
            batch_X
        )

        predictions.append(
            batch_predictions.cpu().numpy()
        )


predictions = np.concatenate(
    predictions
)


# ============================================================
# Save Predictions
# ============================================================
np.savez(
    PREDICTION_PATH,
    y_true=y_test,
    y_pred=predictions
)


# ============================================================
# Display Sample Predictions
# ============================================================
print("\nSample predictions:")
print("-" * 45)

for i in range(min(10, len(predictions))):

    print(
        f"Sample {i + 1:02d} | "
        f"Actual RUL: {y_test[i]:7.2f} | "
        f"Predicted RUL: {predictions[i]:7.2f}"
    )


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)

print(f"Total predictions : {len(predictions)}")
print(f"Predictions saved : {PREDICTION_PATH}")