import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed"

X_train = np.load(DATA_PATH / "X_train.npy")
X_test = np.load(DATA_PATH / "X_test.npy")
y_train = np.load(DATA_PATH / "y_train.npy")
y_test = np.load(DATA_PATH / "y_test.npy")

print("=" * 60)
print("MAMBA DATASET PREPARATION")
print("=" * 60)

print("\nTraining Data")
print(f"X_train Shape : {X_train.shape}")
print(f"y_train Shape : {y_train.shape}")

print("\nTesting Data")
print(f"X_test Shape : {X_test.shape}")
print(f"y_test Shape : {y_test.shape}")

print("\nSequence Information")
print(f"Sequence Length : {X_train.shape[1]}")
print(f"Features : {X_train.shape[2]}")

print("\nRemaining Useful Life")

print(f"Maximum RUL : {y_train.max()}")
print(f"Minimum RUL : {y_train.min()}")

print("\nData Type")

print(X_train.dtype)

print("\nFirst Training Sample")

print(X_train[0])

print("\nCorresponding RUL")

print(y_train[0])

print("\nDataset Ready for Mamba Training")