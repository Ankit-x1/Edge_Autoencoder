import torch 
from lstm_autoencoder import LSTMAutoencoder 

model = LSTMAutoencoder()
model.load_state_dict(torch.load("../models/best_model.pth"))
model.eval()

dummy = torch.randn(1,50,1)
torch.onnx.export(
    model, dummy, "../models/model.onnx",
    input_names = ["input"],
    output_names = ["output"],
    opset_version=13)
print("Exported to ONNX.")
