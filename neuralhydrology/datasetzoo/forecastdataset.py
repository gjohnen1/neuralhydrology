import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch


class ForecastDataset(Dataset):
    def __init__(self,
                 features,
                 target,
                 window_size,
                 horizon):
        
        super(ForecastDataset, self).__init__()
        self.features = features
        self.target = target
        self.window_size = window_size
        self.horizon = horizon
     

    def __len__(self):
        # Adjust the total count to ensure we have enough data for the last window + horizon
        return len(self.target) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        sample = {}
        # Create input window of features
        sample[f"x_h"] = torch.tensor(self.features[idx+1 : idx+1 + self.window_size - self.horizon], dtype=torch.float32)
        sample[f"x_f"] = torch.tensor(self.features[idx+1 + self.window_size - self.horizon : idx+1 + self.window_size], dtype=torch.float32)     
        sample[f"y"] = torch.tensor(self.target.iloc[idx+1 : idx + self.window_size+1].values, dtype=torch.float32).unsqueeze(-1)
        
        return sample