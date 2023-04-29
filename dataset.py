import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, X, y, dtype_X=torch.float32, dtype_y=torch.float32):
        self.X = torch.tensor(X, dtype=dtype_X)
        self.y = torch.tensor(y, dtype=dtype_y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y

def create_lookback_dataset(X, y, lookback):
    X_new = []  # Initialize an empty list to store the new input dataset
    y_new = []  # Initialize an empty list to store the new target dataset

    # Iterate through the input dataset, starting from the first element to (length - lookback)
    for i in range(len(X) - lookback):
        # Create a lookback window from the current element to (current element + lookback)
        X_new.append(X[i:i+lookback])
        # Append the target value at the position (current element + lookback) to the new target dataset
        y_new.append(y[i+lookback])

    # Convert the lists to numpy arrays and return
    return np.array(X_new), np.array(y_new)