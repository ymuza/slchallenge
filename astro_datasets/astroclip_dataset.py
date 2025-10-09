import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class AstroclipDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (ndarray): Input features.
            y (ndarray): Target variable.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]