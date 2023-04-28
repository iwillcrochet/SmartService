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
    X_new = []
    y_new = []

    for i in range(len(X) - lookback):
        X_new.append(X[i:i+lookback])
        y_new.append(y[i+lookback])

    return np.array(X_new), np.array(y_new)