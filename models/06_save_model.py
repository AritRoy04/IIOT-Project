import os
import sys
import torch

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.mamba_model import MambaRULModel

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "models",
    "best_mamba_rul_model.pth"
)

EXPORT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "models",
    "final_mamba_rul_model.pth"
)


# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("MAMBA RUL MODEL EXPORT")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Check trained model
# ============================================================
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Trained model not found:\n{CHECKPOINT_PATH}"
    )

print("\nLoading trained Mamba model...")


# ============================================================
# Load checkpoint
# ============================================================
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

print("Checkpoint loaded successfully")


# ============================================================
# Rebuild model
# IMPORTANT: These parameters must match the training model
# ============================================================
model = MambaRULModel().to(device)


# ============================================================
# Load trained weights
# ============================================================
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model weights loaded successfully")


# ============================================================
# Save final deployment checkpoint
# ============================================================
export_data = {
    "model_state_dict": model.state_dict(),

    "model_name": "MambaRULModel",

    "input_features": 25,
    "sequence_length": 30,

    "task": "Remaining Useful Life Prediction",

    "framework": "PyTorch",

    "checkpoint_source": CHECKPOINT_PATH
}

torch.save(export_data, EXPORT_PATH)


# ============================================================
# Verify saved model
# ============================================================
print("\nVerifying exported model...")

saved_model = torch.load(
    EXPORT_PATH,
    map_location=device,
    weights_only=False
)

print("Export verification successful")


# ============================================================
# Summary
# ============================================================
file_size = os.path.getsize(EXPORT_PATH) / (1024 * 1024)

print("\n" + "=" * 60)
print("MODEL EXPORT COMPLETED")
print("=" * 60)

print(f"Model name       : {saved_model['model_name']}")
print(f"Input features   : {saved_model['input_features']}")
print(f"Sequence length  : {saved_model['sequence_length']}")
print(f"Task             : {saved_model['task']}")
print(f"Model file       : {EXPORT_PATH}")
print(f"File size        : {file_size:.2f} MB")