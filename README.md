# STA-UKAN

> **STA-UKAN: Subseasonal Temperature Forecast Refinement via Multi-source Atmospheric Factor Fusion and Terrain-Aware KAN Network**

## Overview

STA-UKAN is a deep learning framework for subseasonal temperature forecast refinement, combining multi-source atmospheric factor fusion with terrain-aware Kolmogorov-Arnold Network (KAN) architecture.

## Features

- Multi-source atmospheric data fusion (ECMWF, CMA-GFS)
- Terrain-aware downscaling with elevation integration  
- KAN-enhanced UNet architecture for improved prediction
- Support for 2-5 week ahead temperature forecasting

---

## Experimental Environment

All experiments were conducted under the following environment:

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 3090 (24 GB) |
| CUDA | 12.2 |
| Python | 3.10.14 |
| Framework | PyTorch |

---

## Data and Pretrained Checkpoints

Datasets and pretrained checkpoints are hosted on Google Drive:

**[Download from Google Drive](https://drive.google.com/drive/folders/1gLx7fGLR8m1HZW45BBimFTwEstCZQmaI)**

The folder includes:
- Processed training / validation / test datasets
- Pretrained checkpoints for Weeks 2–5

### Recommended Directory Layout

```
STA-UKAN/
├── data/
│   ├── output/           # Processed .npy data files
│   ├── MASK.npy
│   └── final_map1.npy    # Terrain data
└── checkpoints/
    ├── 2week/
    ├── 3week/
    ├── 4week/
    └── 5week/
```

---

## Project Structure

```
STA-UKAN/
├── model/                 # Model architectures
│   ├── model_factory.py   # Model registry and factory
│   ├── kan.py             # KAN layer implementation
│   ├── m4_sta_ukan.py     # STA-UKAN main model
│   ├── edsr.py            # EDSR baseline
│   ├── srcnn.py           # SRCNN baseline
│   ├── srdrn.py           # SRDRN baseline
│   └── srresnet.py        # SRResNet baseline
├── data/
│   └── data_loader.py     # Data loading utilities
├── utils/
│   ├── evaluation.py      # Evaluation metrics
│   ├── training.py        # Training utilities
│   ├── utils.py           # Common utilities
│   └── visualization.py   # Visualization tools
├── checkpoints/           # Pretrained models (external)
├── assets/                # Figures for README
├── inference.py           # Inference script
├── config_5week.yaml      # Experiment configuration
└── README.md
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Inference

```bash
python inference.py --model CUKan2_1_real --week 5week --ckpt checkpoints/5week/model.pt
```

### Training *(Coming Soon)*

Training scripts will be released in a future update.

---

## Supported Models

### Proposed Models
- **STA-UKAN** (CUKan2_1_real): Main proposed model with KAN-enhanced architecture
- **UKan**: Basic UKAN architecture
- **CUKan**: Conditional UKAN variants

### Baseline Models
- **UNet**: Standard UNet architecture
- **EDSR**: Enhanced Deep Residual Networks for Super-Resolution
- **SRCNN**: Super-Resolution CNN
- **SRDRN**: Super-Resolution Dense Residual Network
- **SRResNet**: Super-Resolution Residual Network

---

## Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **MAE** | Mean Absolute Error (95% CI) | ↓ Lower is better |
| **RMSE** | Root Mean Square Error (95% CI) | ↓ Lower is better |
| **MBE** | Mean Bias Error | - |
| **ACC** | Anomaly Correlation Coefficient | ↑ Higher is better |
| **PCC** | Pearson Correlation Coefficient | ↑ Higher is better |
| **SSIM** | Structural Similarity Index | ↑ Higher is better |
| **PSNR** | Peak Signal-to-Noise Ratio | ↑ Higher is better |
| **p-value** | Statistical Significance | < 0.05 |

---

## KAN-based Network References

The KAN-related components are inspired by:

- [UNetKAN](https://github.com/jiaowoguanren0615/UNetKAN)
- [U-KAN](https://github.com/CUHK-AIM-Group/U-KAN)
- [UKAN for crop-field segmentation](https://github.com/DarthReca/crop-field-segmentation-ukan)

---

## Baseline References

| Model | Source |
|-------|--------|
| UNet | [PyTorch-UNet](https://github.com/milesial/Pytorch-UNet) |
| Swin-Unet | [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) |
| EDSR | [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch) |
| SRDRN | [SRDRN](https://github.com/honghong2023/SRDRN) |
| SRCNN | [SRCNN-pytorch](https://github.com/yjn870/SRCNN-pytorch) |
| SRResNet | [pytorch-SRResNet](https://github.com/twtygqyy/pytorch-SRResNet) |
| RCAN | [RCAN](https://github.com/yulunzhang/RCAN) |

---

## Reproducibility Notes

- Identical train/val/test splits across all methods
- Same evaluation scripts for all metrics
- Experiment settings fully specified in config files

---

## Roadmap

- [x] Release model architectures
- [x] Release pretrained checkpoints and datasets
- [ ] Release training scripts
- [ ] Add inference demo notebook
- [ ] Provide Docker image for environment setup
- [ ] Extend to additional meteorological variables

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{whx2025staukan,
  title   = {STA-UKAN: Subseasonal Temperature Forecast Refinement via Multi-source Atmospheric Factor Fusion and Terrain-Aware KAN Network},
  author  = {XXX},
  journal = {XXX},
  year    = {2025}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
