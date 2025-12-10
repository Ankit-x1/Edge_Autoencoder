# EdgeAutoencoder

A clean LSTM Autoencoder project for time series anomaly detection.

## Features
- Modular Python code: dataset, model, training, visualization, inference
- Configurable via YAML
- Export to ONNX for embedded deployment
- Includes unit tests

## Requirements
- Python 3.10+
- PyTorch
- pandas
- matplotlib
- onnxruntime (optional for ONNX inference)
- pytest (for tests)


### Training
```bash
python src/train.py
