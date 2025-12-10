import torch 
from lstm_autoencoder import LSTMAutoencoder 
import os 

model = LSTMAutoencoder()
model_path = "../models/best_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model.load_state_dict(torch.load(model_path))


model.eval()

dummy = torch.randn(1,50,1)
os.makedirs("../models", exist_ok=True)
torch.onnx.export(
    model, dummy, "../models/model.onnx",
    input_names = ["input"],
    output_names = ["output"],
    opset_version=13)
print("Exported to ONNX.")
