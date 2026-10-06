import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
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
SEED = 42
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 7

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Data Paths
# ============================================================
X_TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "X_train_strict.npy"
)

Y_TRAIN_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "y_train_strict.npy"
)

X_VAL_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "X_val_strict.npy"
)

Y_VAL_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "y_val_strict.npy"
)


# ============================================================
# Output Paths
# ============================================================
MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_mamba_rul_model_strict.pth"
)

HISTORY_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "mamba_training_history_strict.npz"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# Reproducibility
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Header
# ============================================================
print("=" * 60)
print("MAMBA RUL MODEL TRAINING - STRICT ENGINE-WISE SPLIT")
print("=" * 60)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# Load Strict Training and Validation Data
# ============================================================
print("\nLoading strict engine-wise datasets...")

X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)

X_val = np.load(X_VAL_PATH)
y_val = np.load(Y_VAL_PATH)

print("\nTraining data:")
print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")

print("\nValidation data:")
print(f"X_val shape   : {X_val.shape}")
print(f"y_val shape   : {y_val.shape}")


# ============================================================
# Convert Data Types
# ============================================================
X_train = X_train.astype(
    np.float32
)

y_train = y_train.astype(
    np.float32
)

X_val = X_val.astype(
    np.float32
)

y_val = y_val.astype(
    np.float32
)


# ============================================================
# Convert to PyTorch Tensors
# ============================================================
X_train_tensor = torch.from_numpy(
    X_train
)

y_train_tensor = torch.from_numpy(
    y_train
)

X_val_tensor = torch.from_numpy(
    X_val
)

y_val_tensor = torch.from_numpy(
    y_val
)


# ============================================================
# Create Datasets
# ============================================================
train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)


# ============================================================
# Create Data Loaders
# ============================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nStrict split summary:")
print(
    f"Training samples   : "
    f"{len(train_dataset)}"
)

print(
    f"Validation samples : "
    f"{len(val_dataset)}"
)


# ============================================================
# Create Mamba Model
# ============================================================
model = MambaRULModel().to(
    DEVICE
)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"\nTotal model parameters: "
    f"{total_parameters}"
)


# ============================================================
# Loss and Optimizer
# ============================================================
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training Variables
# ============================================================
best_val_loss = float("inf")

patience_counter = 0

train_losses = []
val_losses = []

best_epoch = 0

start_time = time.time()


# ============================================================
# Training Loop
# ============================================================
print("\nStarting strict engine-wise training...\n")

for epoch in range(
    1,
    EPOCHS + 1
):

    # --------------------------------------------------------
    # Training Phase
    # --------------------------------------------------------
    model.train()

    total_train_loss = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(
            DEVICE
        )

        batch_y = batch_y.to(
            DEVICE
        )

        optimizer.zero_grad()

        predictions = model(
            batch_X
        )

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_train_loss += (
            loss.item()
            * batch_X.size(0)
        )


    epoch_train_loss = (
        total_train_loss
        / len(train_dataset)
    )


    # --------------------------------------------------------
    # Validation Phase
    # --------------------------------------------------------
    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(
                DEVICE
            )

            batch_y = batch_y.to(
                DEVICE
            )

            predictions = model(
                batch_X
            )

            loss = criterion(
                predictions,
                batch_y
            )

            total_val_loss += (
                loss.item()
                * batch_X.size(0)
            )


    epoch_val_loss = (
        total_val_loss
        / len(val_dataset)
    )


    # --------------------------------------------------------
    # Save History
    # --------------------------------------------------------
    train_losses.append(
        epoch_train_loss
    )

    val_losses.append(
        epoch_val_loss
    )


    # --------------------------------------------------------
    # Print Epoch Result
    # --------------------------------------------------------
    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Val Loss: {epoch_val_loss:.4f}"
    )


    # --------------------------------------------------------
    # Save Best Model
    # --------------------------------------------------------
    if epoch_val_loss < best_val_loss:

        best_val_loss = (
            epoch_val_loss
        )

        best_epoch = epoch

        patience_counter = 0

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_val_loss":
                    best_val_loss,

                "split_type":
                    "strict_engine_wise",

                "training_engines":
                    64,

                "validation_engines":
                    16,

                "testing_engines":
                    20
            },
            MODEL_PATH
        )

        print(
            f"   -> Best model saved "
            f"(Val Loss: "
            f"{best_val_loss:.4f})"
        )

    else:

        patience_counter += 1

        print(
            f"   -> No improvement "
            f"({patience_counter}/"
            f"{PATIENCE})"
        )


    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------
    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# Save Training History
# ============================================================
np.savez(
    HISTORY_PATH,
    train_loss=np.array(
        train_losses
    ),
    val_loss=np.array(
        val_losses
    ),
    best_epoch=best_epoch,
    best_val_loss=best_val_loss
)


# ============================================================
# Training Summary
# ============================================================
training_time = (
    time.time() - start_time
) / 60

print("\n" + "=" * 60)
print("STRICT ENGINE-WISE TRAINING COMPLETED")
print("=" * 60)

print(
    f"Epochs completed      : "
    f"{len(train_losses)}"
)

print(
    f"Best epoch            : "
    f"{best_epoch}"
)

print(
    f"Best validation loss  : "
    f"{best_val_loss:.4f}"
)

print(
    f"Training time         : "
    f"{training_time:.2f} minutes"
)

print(
    f"Best model saved to   : "
    f"{MODEL_PATH}"
)

print(
    f"History saved to      : "
    f"{HISTORY_PATH}"
)