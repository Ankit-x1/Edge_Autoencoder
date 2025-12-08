import torch 
from torch.utils.data import Dataset 
import pandas as pd 

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, window=50):
        data = pd.read_csv(csv_path).values
        self.data = torch.tensor(data, dtype=torch.float32)
        self.windows = window
    
    def __len__(self):
        return len(self.data) - self.windows
    
    def __getitem__(self, idx):
        return self.data[idx:idx+self.windows]
    