import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scaled_features.csv"
)

ORIGINAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_with_rul.csv"
)

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================
WINDOW_SIZE = 30


# ============================================================
# Load Data
# ============================================================
print("=" * 60)
print("SEQUENCE GENERATION")
print("=" * 60)

print("\nLoading data...")

scaled_df = pd.read_csv(INPUT_FILE)
original_df = pd.read_csv(ORIGINAL_FILE)

# Add engine ID back to the scaled dataset
scaled_df["engine_id"] = original_df["engine_id"].values

# Select model input features
# Exclude target and engine identifier
feature_columns = [
    col for col in scaled_df.columns
    if col not in ["RUL", "engine_id"]
]

print(f"Total rows      : {len(scaled_df)}")
print(f"Input features  : {len(feature_columns)}")
print(f"Window size     : {WINDOW_SIZE}")
print(
    f"Unique engines  : "
    f"{scaled_df['engine_id'].nunique()}"
)


# ============================================================
# Generate Sequences
# ============================================================
X_sequences = []
y_sequences = []
sequence_engine_ids = []

print("\nGenerating sequences...")

for engine in scaled_df["engine_id"].unique():

    engine_data = scaled_df[
        scaled_df["engine_id"] == engine
    ]

    features = engine_data[feature_columns].values
    rul = engine_data["RUL"].values

    # Skip engines shorter than the sequence window
    if len(engine_data) < WINDOW_SIZE:
        continue

    for i in range(
        len(engine_data) - WINDOW_SIZE + 1
    ):

        # Input sequence
        X_sequences.append(
            features[i:i + WINDOW_SIZE]
        )

        # RUL at the final time step
        y_sequences.append(
            rul[i + WINDOW_SIZE - 1]
        )

        # Save engine ID for this sequence
        sequence_engine_ids.append(engine)


# ============================================================
# Convert to NumPy Arrays
# ============================================================
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)
sequence_engine_ids = np.array(sequence_engine_ids)


# ============================================================
# Save Generated Data
# ============================================================
np.save(
    OUTPUT_PATH / "X_sequences.npy",
    X_sequences
)

np.save(
    OUTPUT_PATH / "y_sequences.npy",
    y_sequences
)

np.save(
    OUTPUT_PATH / "sequence_engine_ids.npy",
    sequence_engine_ids
)


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SEQUENCE GENERATION COMPLETED")
print("=" * 60)

print(f"Window Size       : {WINDOW_SIZE}")
print(f"Total Sequences   : {len(X_sequences)}")
print(f"Input Shape       : {X_sequences.shape}")
print(f"Target Shape      : {y_sequences.shape}")
print(
    f"Engine ID Shape   : "
    f"{sequence_engine_ids.shape}"
)
print(
    f"Unique Engines    : "
    f"{len(np.unique(sequence_engine_ids))}"
)

print("\nExample Sequence Shape:")
print(X_sequences[0].shape)

print(f"\nFiles saved to:\n{OUTPUT_PATH}")