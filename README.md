# 🩺 Skin Burn Severity Analyzer

A deep learning project to classify skin burns into 1st-degree, 2nd-degree, or 3rd-degree categories using PyTorch and ResNet18. This repository includes scripts for dataset loading, model training, and prediction to streamline the entire workflow from data preparation to inference.

## Features
- 80/20 train-validation dataset split
- Data augmentation: random horizontal flip, rotation, color jitter, normalization, random erasing
- Transfer learning with ResNet18 pretrained on ImageNet
- CUDA support (uses GPU if available)
- Inference script outputs class probabilities and predicted burn severity

## Model Details
- Base model: ResNet18
- Modified final layer: `nn.Linear(512, 3)`
- Loss: CrossEntropyLoss
- Optimizer: Adam, learning rate 0.001
- Epochs: 20
- Batch size: 32

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/Skin-Burn-Severity-Analyzer.git
cd Skin-Burn-Severity-Analyzer
pip install torch torchvision matplotlib pillow numpy
```

## Prepare your dataset
dataset/
├── 1st degree/
│   ├── img1.jpg
│   ├── img2.jpg
├── 2nd degree/
│   ├── img1.jpg
├── 3rd degree/
│   ├── img1.jpg

## Training
```bash
python train_model.py
```

## Prediction
```bash
python predict.py
```

## Example Output
tensor([[0.05, 0.20, 0.75]])
Prediction: 3rd degree
