# COMP3314 Project: ResNet Reproduction

## Overview

This project reproduces the CIFAR-10 experiments from "Deep Residual Learning for Image Recognition" (He et al., CVPR 2016), including the PlainNet baseline used to illustrate the degradation phenomenon.

## Project Structure

```
resnet-reproduction/
├── models/                   # Model architectures
│   ├── resnet_cifar.py      # ResNet and PlainNet for CIFAR-10
│   └── resnet_imagenet.py   # (Optional) ResNet for ImageNet
├── datasets/                # Data loading utilities
│   └── cifar10_loader.py    # CIFAR-10 dataset and augmentation
├── utils/                   # Training and utility functions
│   ├── train_eval.py        # Training loop and evaluation
│   ├── scheduler.py         # Learning rate scheduling
│   └── plot_curves.py       # Visualization utilities
├── experiments/             # Experiment notebooks and scripts
│   ├── plain_vs_resnet.ipynb         # Training curve comparison
│   └── layer_response_analysis.py    # Layer response analysis
├── results/                 # Output directory
│   ├── checkpoints/         # Trained model weights
│   └── logs/                # Training logs
└── main.py                  # Main training script
```

## Quick Start

### 1. Install dependencies

```bash
py -3 -m pip install -r requirements.txt   # Windows
# or
pip install -r requirements.txt            # Other environments
```

### 2. Sanity check the models

```bash
py -3 smoke_test.py
```

The helper script runs a forward pass on `ResNet20`, `ResNet32`, and `PlainNet20` to verify that everything is wired up correctly. (Create it with the snippet shown in `utils/train_eval.py` docstring or adapt to your preference.)

### 3. Train on CIFAR-10

```bash
py -3 main.py --model ResNet20 --epochs 164 --batch_size 128 --lr 0.1
```

Useful flags:

- `--model`: `ResNet20`, `ResNet32`, `PlainNet20`, `PlainNet32`
- `--device`: `cuda` (default, falls back to CPU if unavailable) or `cpu`
- `--amp`: enable automatic mixed precision when using CUDA
- `--warmup_epochs`: fractional epochs of linear warmup (e.g. `--warmup_epochs 5`)

If a CIFAR-10 download is interrupted or corrupted, delete `./data/cifar-10-python.tar.gz` and rerun the command; it will redownload automatically.

### 4. Inspect logs and checkpoints

Training metrics are stored in `results/logs/`, and model checkpoints in `results/checkpoints/`.

### 5. (Optional) Run analysis notebooks

```bash
jupyter notebook experiments/plain_vs_resnet.ipynb
```

## Paper Alignment

- **Section 3.1-3.3**: Residual block structure
- **Section 4.2**: CIFAR-10 experiment setup
- **Table 6**: Network depth configurations (20, 32, 56, 110)
- **Figure 6**: Training/test error curves
- **Figure 7**: Layer response analysis

## Status / TODO

- [x] Implement BasicBlock and ResNet/PlainNet CIFAR architectures
- [x] Implement CIFAR-10 data loading with augmentation
- [x] Implement training loop, evaluation utilities, and checkpointing
- [x] Implement learning rate scheduler with warmup support
- [ ] Reproduce Figure 6 training/test error curves
- [ ] Reproduce Figure 7 layer response analysis
- [ ] Extend to deeper networks (ResNet56/110) and additional experiments