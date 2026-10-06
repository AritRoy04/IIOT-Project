import numpy as np
import pandas as pd
import sklearn
import torch

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())