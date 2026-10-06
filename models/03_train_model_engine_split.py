import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

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
VAL_RATIO = 0.2
PATIENCE = 7

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Paths
# ============================================================
X_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "X_train_engine.npy"
)

Y_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "y_train_engine.npy"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "models"
)

HISTORY_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_mamba_rul_model_engine_split.pth"
)

HISTORY_PATH = os.path.join(
    HISTORY_DIR,
    "mamba_training_history_engine_split.npz"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


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
print("MAMBA RUL MODEL TRAINING - ENGINE-WISE SPLIT")
print("=" * 60)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# Load Training Data
# ============================================================
print("\nLoading engine-wise training data...")

X_train = np.load(X_PATH)
y_train = np.load(Y_PATH)

print(f"X shape: {X_train.shape}")
print(f"y shape: {y_train.shape}")


# ============================================================
# Convert Data Type
# ============================================================
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)


# ============================================================
# Convert to PyTorch Tensors
# ============================================================
X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train)

dataset = TensorDataset(
    X_tensor,
    y_tensor
)


# ============================================================
# Training / Validation Split
# ============================================================
val_size = int(
    len(dataset) * VAL_RATIO
)

train_size = (
    len(dataset) - val_size
)

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print(f"\nTraining samples   : {train_size}")
print(f"Validation samples : {val_size}")


# ============================================================
# Data Loaders
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


# ============================================================
# Create Mamba Model
# ============================================================
model = MambaRULModel().to(DEVICE)

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

start_time = time.time()


# ============================================================
# Training Loop
# ============================================================
print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # Training Phase
    # --------------------------------------------------------
    model.train()

    total_train_loss = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(batch_X)

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_train_loss += (
            loss.item() * batch_X.size(0)
        )

    epoch_train_loss = (
        total_train_loss / len(train_dataset)
    )


    # --------------------------------------------------------
    # Validation Phase
    # --------------------------------------------------------
    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            predictions = model(batch_X)

            loss = criterion(
                predictions,
                batch_y
            )

            total_val_loss += (
                loss.item() * batch_X.size(0)
            )

    epoch_val_loss = (
        total_val_loss / len(val_dataset)
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
    # Print Epoch Results
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

        best_val_loss = epoch_val_loss
        patience_counter = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_val_loss":
                    best_val_loss,

                "split_type":
                    "engine-wise",

                "training_file":
                    "X_train_engine.npy"
            },
            MODEL_PATH
        )

        print(
            f"   -> Best model saved "
            f"(Val Loss: {best_val_loss:.4f})"
        )

    else:

        patience_counter += 1

        print(
            f"   -> No improvement "
            f"({patience_counter}/{PATIENCE})"
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
    train_loss=np.array(train_losses),
    val_loss=np.array(val_losses)
)


# ============================================================
# Training Summary
# ============================================================
training_time = (
    time.time() - start_time
) / 60

print("\n" + "=" * 60)
print("ENGINE-WISE TRAINING COMPLETED")
print("=" * 60)

print(
    f"Epochs completed      : "
    f"{len(train_losses)}"
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