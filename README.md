# STA-UKAN

> **STA-UKAN: Subseasonal Temperature Forecast Refinement via Multi-source Atmospheric Factor Fusion and Terrain-Aware KAN Network**

## Overview

STA-UKAN is a deep learning framework for subseasonal temperature forecast refinement, combining multi-source atmospheric factor fusion with terrain-aware Kolmogorov-Arnold Network (KAN) architecture.

<p align="center">
  <img src="assets/Figure1.png" width="85%">
</p>


## Features

- Multi-source atmospheric data fusion (ECMWF, CMA-GFS)
- Terrain-aware downscaling with elevation integration  
- KAN-enhanced UNet architecture for improved prediction
- Support for 2-5 week ahead temperature forecasting

---

## Experimental Environment

All experiments were conducted and validated on **two independent GPU hardware platforms** to ensure robustness and reproducibility.

### Environment A: NVIDIA GPU Platform
| Component | Specification |
|----------|---------------|
| GPU | NVIDIA RTX 3090 (24 GB) |
| Driver | NVIDIA Driver 535.183.01 |
| CUDA | 12.2 |
| Python | 3.10.14 |
| Framework | PyTorch |

### Environment B: MX GPU Platform 
| Component | Specification |
|----------|---------------|
| GPU | MXC500 |
| Driver Stack | MX-SMI 2.2.8 |
| Kernel Mode Driver | 3.0.11 |
| MACA Version | 3.1.0.14 |
| BIOS Version | 1.27.5.0 |
| Python | 3.10.19 |
| Framework | PyTorch |

> The proposed model and all baseline methods were **trained and evaluated on both GPU hardware platforms**.  
> Although minor numerical differences may arise due to driver stacks and backend implementations, the **relative performance rankings and conclusions remain consistent** across platforms.

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

To comprehensively evaluate the effectiveness of the proposed STA-UKAN framework, we compare it with a wide range of baseline methods, including **traditional statistical approaches**, **machine learning models**, and **deep learning–based methods**.

### 1. Traditional Statistical Methods
- **Ensemble Mean**: Simple multi-model ensemble averaging
- **QM (Quantile Mapping)**: Classical distribution-based bias correction
- **QDM (Quantile Delta Mapping)**: Trend-preserving quantile-based correction

These methods serve as widely used benchmarks in operational bias correction and downscaling studies.

---

### 2. Machine Learning–based Methods
- **RF (Random Forest)**
- **GBR (Gradient Boosting Regressor)**
- **SGDRegressor**
- **AdaBoost**
- **LightGBM**
- **SVM (Support Vector Machine)**
- **XGBoost**

All machine learning models are trained using identical input features and target variables to ensure fair comparison.

---

### 3. Deep Learning–based Methods

#### Proposed Models
- **STA-UKAN**: Main proposed model with terrain-aware KAN-enhanced UNet architecture  
- **UKAN**: Base UKAN architecture without conditional or terrain-aware extensions  
- **CUKAN**: Conditional UKAN variants incorporating auxiliary atmospheric information  

#### Baseline Deep Learning Models
- **UNet**: Standard encoder–decoder UNet architecture  
- **CUNet**: Sub-pixel convolution UNet
- **EDSR**: Enhanced Deep Residual Network  
- **SwinIR**: Swin Transformer–based Image Restoration Network  
- **RCAN**: Residual Channel Attention Network  
- **SRCNN**: Super-Resolution Convolutional Neural Network  
- **SRResNet**: Super-Resolution Residual Network  
- **SRDRN**: Super-Resolution Dense Residual Network  

All deep learning models are trained and evaluated under the same data splits, spatial resolution, and evaluation metrics.

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
@article{better2025staukan,
  title   = {STA-UKAN: Subseasonal Temperature Forecast Refinement via Multi-source Atmospheric Factor Fusion and Terrain-Aware KAN Network},
  author  = {better},
  year    = {2025}
}
```

---

