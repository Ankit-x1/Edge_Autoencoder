import torch
from torch.utils.data import DataLoader
from lstm_autoencoder import LSTMAutoencoder
from dataset import TimeSeriesDataset
import torch.optim as optim
import yaml
import os

config_path = "./config/config.yaml"
data_path = "../data/sample_timeseries.csv"

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")

cfg = yaml.safe_load(open(config_path))

if not os.path.exists(data_path):
    raise FileNotFoundError(f"CSV data file not found: {data_path}")

dataset = TimeSeriesDataset(data_path, window=cfg["window"])

loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

model = LSTMAutoencoder(
    input_dim=1,
    hidden_dim=cfg["hidden_dim"],
    latent_dim=cfg["latent_dim"],
    num_layers=cfg["num_layers"]
)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

for epoch in range(cfg["epochs"]):
    total_loss = 0
    for x in loader:
        x = x.view(x.size(0), x.size(1), 1)  
        out = model(x)
        loss = criterion(out, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{cfg['epochs']}: loss = {total_loss/len(loader)}")

os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/best_model.pth")
print("Model saved!")
