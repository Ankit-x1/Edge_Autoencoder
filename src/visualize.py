import matplotlib.pyplot as plt 
import torch
from lstm_autoencoder import LSTMAutoencoder
from dataset import TimeSeriesDataset

model = LSTMAutoencoder()
model.load_state_dict(torch.load("../models/best_model.pth"))
model.eval()

dataset = TimeSeriesDataset("../data/sample_timeseries.csv")
x = dataset[100].view(1, -1, 1)

with torch.no_grad():
    recon = model(x)

plt.plot(x.flatten(), label="original")
plt.plot(recon.flatten(), label="reconstruction")
plt.legend()
plt.show()
