import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"


# ============================================================
# Configuration
# ============================================================
SEED = 42

# Final engine distribution:
# 64 Train + 16 Validation + 20 Test = 100 Engines
TEST_RATIO = 0.20
VAL_RATIO_FROM_REMAINING = 0.20


# ============================================================
# Load Sequence Data
# ============================================================
print("=" * 60)
print("STRICT ENGINE-WISE TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

print("\nLoading sequence data...")

X = np.load(DATA_PATH / "X_sequences.npy")
y = np.load(DATA_PATH / "y_sequences.npy")
engine_ids = np.load(
    DATA_PATH / "sequence_engine_ids.npy"
)

unique_engines = np.unique(engine_ids)

print(f"X shape         : {X.shape}")
print(f"y shape         : {y.shape}")
print(f"Engine ID shape : {engine_ids.shape}")
print(f"Unique engines  : {len(unique_engines)}")


# ============================================================
# Step 1: Split 100 Engines into 80 Development + 20 Test
# ============================================================
development_engines, test_engines = train_test_split(
    unique_engines,
    test_size=TEST_RATIO,
    random_state=SEED,
    shuffle=True
)


# ============================================================
# Step 2: Split 80 Development Engines into 64 Train + 16 Val
# ============================================================
train_engines, val_engines = train_test_split(
    development_engines,
    test_size=VAL_RATIO_FROM_REMAINING,
    random_state=SEED,
    shuffle=True
)


# ============================================================
# Create Masks
# ============================================================
train_mask = np.isin(
    engine_ids,
    train_engines
)

val_mask = np.isin(
    engine_ids,
    val_engines
)

test_mask = np.isin(
    engine_ids,
    test_engines
)


# ============================================================
# Create Strict Datasets
# ============================================================
X_train_strict = X[train_mask]
y_train_strict = y[train_mask]

X_val_strict = X[val_mask]
y_val_strict = y[val_mask]

X_test_strict = X[test_mask]
y_test_strict = y[test_mask]


# ============================================================
# Leakage Checks
# ============================================================
train_set = set(train_engines)
val_set = set(val_engines)
test_set = set(test_engines)

train_val_overlap = train_set.intersection(val_set)
train_test_overlap = train_set.intersection(test_set)
val_test_overlap = val_set.intersection(test_set)

if train_val_overlap:
    raise RuntimeError(
        f"Train/Validation leakage detected: "
        f"{train_val_overlap}"
    )

if train_test_overlap:
    raise RuntimeError(
        f"Train/Test leakage detected: "
        f"{train_test_overlap}"
    )

if val_test_overlap:
    raise RuntimeError(
        f"Validation/Test leakage detected: "
        f"{val_test_overlap}"
    )


# ============================================================
# Sanity Checks
# ============================================================
assert len(train_engines) == 64
assert len(val_engines) == 16
assert len(test_engines) == 20

assert (
    len(X_train_strict)
    + len(X_val_strict)
    + len(X_test_strict)
    == len(X)
)


# ============================================================
# Save Strict Datasets
# ============================================================
np.save(
    DATA_PATH / "X_train_strict.npy",
    X_train_strict
)

np.save(
    DATA_PATH / "y_train_strict.npy",
    y_train_strict
)

np.save(
    DATA_PATH / "X_val_strict.npy",
    X_val_strict
)

np.save(
    DATA_PATH / "y_val_strict.npy",
    y_val_strict
)

np.save(
    DATA_PATH / "X_test_strict.npy",
    X_test_strict
)

np.save(
    DATA_PATH / "y_test_strict.npy",
    y_test_strict
)


# ============================================================
# Save Engine IDs
# ============================================================
np.save(
    DATA_PATH / "strict_train_engine_ids.npy",
    train_engines
)

np.save(
    DATA_PATH / "strict_val_engine_ids.npy",
    val_engines
)

np.save(
    DATA_PATH / "strict_test_engine_ids.npy",
    test_engines
)


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("STRICT ENGINE-WISE SPLIT COMPLETED")
print("=" * 60)

print("\nEngine distribution:")
print(f"Training engines   : {len(train_engines)}")
print(f"Validation engines : {len(val_engines)}")
print(f"Testing engines    : {len(test_engines)}")

print("\nSequence distribution:")
print(f"Training sequences   : {len(X_train_strict)}")
print(f"Validation sequences : {len(X_val_strict)}")
print(f"Testing sequences    : {len(X_test_strict)}")

print("\nDataset shapes:")
print(f"X_train_strict : {X_train_strict.shape}")
print(f"y_train_strict : {y_train_strict.shape}")

print(f"X_val_strict   : {X_val_strict.shape}")
print(f"y_val_strict   : {y_val_strict.shape}")

print(f"X_test_strict  : {X_test_strict.shape}")
print(f"y_test_strict  : {y_test_strict.shape}")

print("\nLeakage verification:")
print(f"Train / Validation overlap : {len(train_val_overlap)}")
print(f"Train / Test overlap       : {len(train_test_overlap)}")
print(f"Validation / Test overlap  : {len(val_test_overlap)}")

print("\nNo engine-level data leakage detected.")
print(f"\nFiles saved to:\n{DATA_PATH}")