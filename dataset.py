import torch
from torch.utils.data import Dataset

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
