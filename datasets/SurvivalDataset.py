from torch.utils.data import Dataset
import torch

class SurvivalDataset(Dataset):
    def __init__(self, x, t, e):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
        self.e = torch.tensor(e, dtype=torch.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.e[idx], idx  # idx for tracking