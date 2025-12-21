# STA-UKAN

**Subseasonal Temperature Forecast Refinement via Multisource Atmospheric Factor Fusion and Terrain-Aware Token-KAN**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

STA-UKAN is a deep learning framework for subseasonal (2-week ahead) temperature forecasting that combines:
- **Multisource Atmospheric Factor Fusion**: Integrates multiple atmospheric variables (temperature, pressure, humidity, wind) using cross-attention mechanisms
- **Terrain-Aware Token-KAN**: Novel architecture combining Kolmogorov-Arnold Networks (KAN) with terrain features for enhanced spatial modeling
- **Token-based Processing**: Efficient sequence modeling for temporal dependencies

## Key Features

- 🌡️ **Subseasonal Temperature Forecasting**: Predicts temperature 1-14 days ahead
- 🌍 **Multisource Data Integration**: Fuses multiple atmospheric factors intelligently
- 🏔️ **Terrain Awareness**: Incorporates topographic features for improved accuracy
- 🧮 **Kolmogorov-Arnold Networks**: Advanced function approximation using KAN layers
- 📊 **Comprehensive Evaluation**: Includes RMSE, MAE, MAPE, R² metrics
- 🎨 **Visualization Tools**: Built-in plotting utilities for results analysis

## Architecture

The STA-UKAN model consists of three main components:

1. **Multisource Fusion Module**
   - Encodes individual atmospheric factors
   - Cross-factor attention for information exchange
   - Produces unified atmospheric representation

2. **Terrain-Aware Token-KAN**
   - Token-based sequence processing
   - Multi-head self-attention
   - KAN layers with terrain modulation
   - Spatio-temporal feature extraction

3. **Forecast Head**
   - Temporal aggregation
   - Dense layers for 14-day forecast generation

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/whx57/STA-UKAN.git
cd STA-UKAN

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import torch
from src.models import STA_UKAN

# Define model configuration
model = STA_UKAN(
    factor_dims={
        'temperature': 10,
        'pressure': 8,
        'humidity': 6,
        'wind': 8
    },
    terrain_dim=5,
    embed_dim=256,
    num_heads=8,
    num_fusion_layers=2,
    num_token_kan_layers=4,
    forecast_horizon=14
)

# Prepare input data
batch_size, seq_len = 32, 30
factor_inputs = {
    'temperature': torch.randn(batch_size, seq_len, 10),
    'pressure': torch.randn(batch_size, seq_len, 8),
    'humidity': torch.randn(batch_size, seq_len, 6),
    'wind': torch.randn(batch_size, seq_len, 8),
}
terrain_features = torch.randn(batch_size, 5)

# Generate forecast
predictions = model(factor_inputs, terrain_features)
print(predictions.shape)  # (32, 14)
```

### Training

```bash
# Train with default configuration
python train.py --config configs/default_config.yaml

# Train with custom settings
python train.py \
    --config configs/default_config.yaml \
    --checkpoint-dir checkpoints \
    --device cuda

# Resume from checkpoint
python train.py \
    --config configs/default_config.yaml \
    --resume checkpoints/latest_checkpoint.pth
```

### Evaluation

```bash
# Evaluate model
python evaluate.py \
    --config configs/default_config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --output-dir outputs
```

### Example Script

```bash
# Run example demonstrating basic usage
python examples/basic_usage.py
```

## Project Structure

```
STA-UKAN/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kan.py              # Kolmogorov-Arnold Network
│   │   ├── token_kan.py        # Terrain-Aware Token-KAN
│   │   ├── fusion.py           # Multisource Fusion
│   │   └── sta_ukan.py         # Main STA-UKAN model
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataloader.py       # Data loading utilities
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training utilities
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── visualization.py    # Plotting utilities
│   └── __init__.py
├── configs/
│   └── default_config.yaml     # Default configuration
├── examples/
│   └── basic_usage.py          # Example usage script
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

## Configuration

Model and training parameters can be configured via YAML files. See `configs/default_config.yaml` for an example:

```yaml
model:
  factor_dims:
    temperature: 10
    pressure: 8
    humidity: 6
    wind: 8
  terrain_dim: 5
  embed_dim: 256
  num_heads: 8
  num_fusion_layers: 2
  num_token_kan_layers: 4
  forecast_horizon: 14
  dropout: 0.1

training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
```

## Model Components

### Kolmogorov-Arnold Network (KAN)

Based on the Kolmogorov-Arnold representation theorem, KAN uses learnable B-spline basis functions for superior function approximation:

```python
from src.models import KAN

kan = KAN(layer_dims=[128, 256, 128, 64], grid_size=5, spline_order=3)
output = kan(input_tensor)
```

### Multisource Fusion

Intelligently combines multiple atmospheric factors:

```python
from src.models import MultisourceFusion

fusion = MultisourceFusion(
    factor_dims={'temp': 10, 'pressure': 8},
    embed_dim=256,
    num_heads=8
)
fused = fusion(factor_inputs)
```

### Terrain-Aware Token-KAN

Processes sequences with terrain awareness:

```python
from src.models import TerrainAwareTokenKAN

token_kan = TerrainAwareTokenKAN(
    input_dim=256,
    embed_dim=256,
    num_heads=8,
    num_layers=4,
    terrain_dim=5
)
output = token_kan(sequences, terrain_features)
```

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **Bias**: Mean prediction error

## Visualization

Built-in visualization tools for analysis:

```python
from src.utils import (
    plot_training_history,
    plot_predictions,
    plot_error_distribution
)

# Plot training curves
plot_training_history(train_losses, val_losses, 'training.png')

# Plot predictions vs targets
plot_predictions(predictions, targets, 'predictions.png')

# Plot error distribution
plot_error_distribution(predictions, targets, 'errors.png')
```

## Data Format

### Input Data

The model expects:

1. **Atmospheric Factors**: Dictionary of tensors
   - Shape: `(batch_size, sequence_length, feature_dim)`
   - Example: `{'temperature': (32, 30, 10), 'pressure': (32, 30, 8)}`

2. **Terrain Features**: Tensor
   - Shape: `(batch_size, terrain_dim)`
   - Example: Elevation, slope, aspect, etc.

3. **Target**: Temperature forecast
   - Shape: `(batch_size, forecast_horizon)`
   - Example: 14-day ahead temperatures

## Citation

If you use this code in your research, please cite:

```bibtex
@article{staukan2024,
  title={STA-UKAN: Subseasonal Temperature Forecast Refinement via Multisource Atmospheric Factor Fusion and Terrain-Aware Token-KAN},
  author={STA-UKAN Team},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Kolmogorov-Arnold Networks (KAN) for advanced function approximation
- PyTorch team for the excellent deep learning framework
- The atmospheric science community for inspiration

## Contact

For questions and feedback, please open an issue on GitHub.