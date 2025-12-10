import torch
import pytest
from src.lstm_autoencoder import LSTMAutoencoder
from src.dataset import TimeSeriesDataset
import os

def test_model_forward_shape():
    model = LSTMAutoencoder()
    x = torch.randn(2, 50, 1)
    out = model(x)
    assert out.shape == x.shape, "Output shape should match input shape"

def test_dataset_windowing(tmp_path):
    data = "\n".join([str(i) for i in range(60)])
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(data)
    
    dataset = TimeSeriesDataset(str(csv_file), window=10)
    assert len(dataset) == 50, "Dataset length should be total_data - window"
    sample = dataset[0]
    assert sample.shape[0] == 10, "Window length should match"
    
def test_onnx_export(tmp_path):
    import torch.onnx
    from src.lstm_autoencoder import LSTMAutoencoder

    model = LSTMAutoencoder()
    dummy = torch.randn(1,50,1)
    onnx_path = tmp_path / "model.onnx"
    
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names = ["input"],
        output_names = ["output"],
        opset_version=13
    )
    
    assert os.path.exists(onnx_path), "ONNX model should be exported"
