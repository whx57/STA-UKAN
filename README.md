# <Your-Project-Name> (e.g., STA-UKAN)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv%2FJournal-orange.svg)](#citation)

> One-line summary: Subseasonal temperature forecast refinement via multi-source atmospheric factor fusion and terrain-aware modeling.

## 🔥 News
- [2025-xx-xx] Initial code release.
- [2025-xx-xx] Pretrained weights released.

## ✨ Highlights
- **End-to-end** post-processing / bias-correction for gridded forecasts.
- **Multi-source** predictors (CMA + ECMWF + auxiliary variables).
- **Terrain-aware** module for complex topography.
- Strong performance on **Weeks 2–5** lead times.

## 🧩 Method Overview
- Input: multi-variable gridded predictors + static terrain features.
- Output: refined high-resolution temperature field.
- Architecture: <briefly describe your encoder-decoder / attention / KAN module>.

> Put a framework figure here:
![framework](assets/framework.png)

## 📦 Installation

### 1) Create environment
```bash
conda create -n staukan python=3.10 -y
conda activate staukan
pip install -r requirements.txt
