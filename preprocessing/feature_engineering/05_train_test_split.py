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
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# Load Sequence Data
# ============================================================
print("=" * 60)
print("ENGINE-WISE TRAIN-TEST SPLIT")
print("=" * 60)

print("\nLoading sequence data...")

X = np.load(DATA_PATH / "X_sequences.npy")
y = np.load(DATA_PATH / "y_sequences.npy")

engine_ids = np.load(
    DATA_PATH / "sequence_engine_ids.npy"
)

print(f"X shape          : {X.shape}")
print(f"y shape          : {y.shape}")
print(f"Engine ID shape  : {engine_ids.shape}")
print(
    f"Unique engines   : "
    f"{len(np.unique(engine_ids))}"
)


# ============================================================
# Split Unique Engine IDs
# ============================================================
unique_engines = np.unique(engine_ids)

train_engines, test_engines = train_test_split(
    unique_engines,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

print("\nSplitting engines...")

print(f"Training engines : {len(train_engines)}")
print(f"Testing engines  : {len(test_engines)}")


# ============================================================
# Create Sequence Masks
# ============================================================
train_mask = np.isin(
    engine_ids,
    train_engines
)

test_mask = np.isin(
    engine_ids,
    test_engines
)


# ============================================================
# Create Engine-Wise Datasets
# ============================================================
X_train_engine = X[train_mask]
y_train_engine = y[train_mask]

X_test_engine = X[test_mask]
y_test_engine = y[test_mask]

train_sequence_engine_ids = engine_ids[train_mask]
test_sequence_engine_ids = engine_ids[test_mask]


# ============================================================
# Leakage Check
# ============================================================
train_engine_set = set(
    np.unique(train_sequence_engine_ids)
)

test_engine_set = set(
    np.unique(test_sequence_engine_ids)
)

overlap = train_engine_set.intersection(
    test_engine_set
)

if len(overlap) != 0:
    raise RuntimeError(
        f"DATA LEAKAGE DETECTED! "
        f"Overlapping engines: {overlap}"
    )


# ============================================================
# Save Engine-Wise Split
# ============================================================
np.save(
    DATA_PATH / "X_train_engine.npy",
    X_train_engine
)

np.save(
    DATA_PATH / "y_train_engine.npy",
    y_train_engine
)

np.save(
    DATA_PATH / "X_test_engine.npy",
    X_test_engine
)

np.save(
    DATA_PATH / "y_test_engine.npy",
    y_test_engine
)

np.save(
    DATA_PATH / "train_engine_ids.npy",
    train_engines
)

np.save(
    DATA_PATH / "test_engine_ids.npy",
    test_engines
)


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("ENGINE-WISE SPLIT COMPLETED")
print("=" * 60)

print(
    f"Training sequences : "
    f"{len(X_train_engine)}"
)

print(
    f"Testing sequences  : "
    f"{len(X_test_engine)}"
)

print(
    f"Training engines   : "
    f"{len(train_engines)}"
)

print(
    f"Testing engines    : "
    f"{len(test_engines)}"
)

print(
    f"Engine overlap     : "
    f"{len(overlap)}"
)

print("\nTraining data shape:")
print(f"X_train_engine : {X_train_engine.shape}")
print(f"y_train_engine : {y_train_engine.shape}")

print("\nTesting data shape:")
print(f"X_test_engine  : {X_test_engine.shape}")
print(f"y_test_engine  : {y_test_engine.shape}")

print("\nNo engine-level data leakage detected.")

print(f"\nFiles saved to:\n{DATA_PATH}")