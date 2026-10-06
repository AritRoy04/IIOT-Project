import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


# ============================================================
# Project Setup
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
# File Paths
# ============================================================
X_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "X_train.npy"
)

Y_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "y_train.npy"
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
    "best_mamba_rul_model.pth"
)

HISTORY_PATH = os.path.join(
    HISTORY_DIR,
    "mamba_training_history.npz"
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
# Create Output Directories
# ============================================================
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================
print("=" * 60)
print("MAMBA RUL MODEL TRAINING")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading training data...")

X = np.load(X_PATH).astype(np.float32)
y = np.load(Y_PATH).astype(np.float32)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# Convert to PyTorch Tensors
# ============================================================
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

dataset = TensorDataset(
    X_tensor,
    y_tensor
)


# ============================================================
# Train / Validation Split
# ============================================================
val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print(f"\nTraining samples:   {train_size}")
print(f"Validation samples: {val_size}")


# ============================================================
# Data Loaders
# ============================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# Initialize Mamba Model
# ============================================================
model = MambaRULModel().to(DEVICE)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nModel initialized successfully")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# ============================================================
# Loss Function and Optimizer
# ============================================================
criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)


# ============================================================
# Training History
# ============================================================
train_losses = []
val_losses = []

best_val_loss = float("inf")
patience_counter = 0

start_time = time.time()


# ============================================================
# Training Loop
# ============================================================
print("\nStarting training...")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # Training Phase
    # --------------------------------------------------------
    model.train()

    running_train_loss = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(
            DEVICE,
            non_blocking=True
        )

        batch_y = batch_y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        predictions = model(batch_X)

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_train_loss += (
            loss.item() * batch_X.size(0)
        )

    epoch_train_loss = (
        running_train_loss / train_size
    )


    # --------------------------------------------------------
    # Validation Phase
    # --------------------------------------------------------
    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True
            )

            batch_y = batch_y.to(
                DEVICE,
                non_blocking=True
            )

            predictions = model(batch_X)

            loss = criterion(
                predictions,
                batch_y
            )

            running_val_loss += (
                loss.item() * batch_X.size(0)
            )

    epoch_val_loss = (
        running_val_loss / val_size
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
    # Display Epoch Results
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
                "input_dim":
                    X.shape[2]
            },
            MODEL_PATH
        )

        print(
            f"  -> Best model saved "
            f"(Val Loss: {best_val_loss:.4f})"
        )

    else:

        patience_counter += 1

        print(
            f"  -> No improvement "
            f"({patience_counter}/{PATIENCE})"
        )


    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------
    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")
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
elapsed_time = (
    time.time() - start_time
) / 60

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"Epochs completed     : {len(train_losses)}")
print(f"Best validation loss : {best_val_loss:.4f}")
print(f"Training time        : {elapsed_time:.2f} minutes")
print(f"Best model saved to  : {MODEL_PATH}")
print(f"History saved to     : {HISTORY_PATH}")