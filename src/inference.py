import torch
import pandas as pd
import argparse
from lstm_autoencoder import LSTMAutoencoder

def load_model(model_path):
    model = LSTMAutoencoder()
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.eval()
    return model

def predict(model, data_window):
    with torch.no_grad():
        x = torch.tensor(data_window, dtype=torch.float32).view(1, -1, 1)
        recon = model(x)
        loss = torch.mean((recon - x) ** 2).item()  
    return loss

def main():
    parser = argparse.ArgumentParser(description="Time Series Anomaly Detection Inference")
    parser.add_argument("--model", default="../models/best_model.pth", help="Path to model file")
    parser.add_argument("--csv", required=True, help="Path to CSV with time series data")
    parser.add_argument("--window", type=int, default=50, help="Window size for model input")
    args = parser.parse_args()

    model = load_model(args.model)

    try:
        data = pd.read_csv(args.csv).values.flatten()
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    except pd.errors.ParserError:
        raise ValueError(f"CSV file is malformed: {args.csv}")

    for i in range(len(data) - args.window):
        window = data[i:i+args.window]
        score = predict(model, window)
        print(f"Window {i}-{i+args.window}: Anomaly Score = {score:.6f}")

if __name__ == "__main__":
    main()
