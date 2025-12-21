"""
Example usage of STA-UKAN model.
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import STA_UKAN


def main():
    """Demonstrate basic usage of STA-UKAN model."""
    
    print("STA-UKAN Model Example")
    print("=" * 50)
    
    # Define model configuration
    config = {
        'factor_dims': {
            'temperature': 10,
            'pressure': 8,
            'humidity': 6,
            'wind': 8
        },
        'terrain_dim': 5,
        'embed_dim': 128,  # Smaller for example
        'num_heads': 4,
        'num_fusion_layers': 2,
        'num_token_kan_layers': 3,
        'kan_hidden_dims': [256, 128],
        'forecast_horizon': 14,
        'dropout': 0.1
    }
    
    # Create model
    print("\n1. Creating STA-UKAN model...")
    model = STA_UKAN(**config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {num_params:,}")
    
    # Create sample input data
    print("\n2. Creating sample input data...")
    batch_size = 4
    seq_len = 30
    
    factor_inputs = {
        'temperature': torch.randn(batch_size, seq_len, 10),
        'pressure': torch.randn(batch_size, seq_len, 8),
        'humidity': torch.randn(batch_size, seq_len, 6),
        'wind': torch.randn(batch_size, seq_len, 8),
    }
    terrain_features = torch.randn(batch_size, 5)
    
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Factor dimensions: {config['factor_dims']}")
    print(f"   Terrain dimension: {config['terrain_dim']}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    model.eval()
    with torch.no_grad():
        predictions = model(factor_inputs, terrain_features)
    
    print(f"   Output shape: {predictions.shape}")
    print(f"   Expected shape: (batch_size={batch_size}, forecast_horizon={config['forecast_horizon']})")
    
    # Display sample predictions
    print("\n4. Sample predictions (first sample):")
    print(f"   {predictions[0].numpy()}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    
    # Model architecture summary
    print("\nModel Architecture Summary:")
    print("-" * 50)
    print(f"Input: Multisource atmospheric factors + terrain features")
    print(f"  - Factors: {list(config['factor_dims'].keys())}")
    print(f"  - Sequence length: {seq_len} time steps")
    print(f"\n1. Multisource Fusion Module:")
    print(f"   - Embedding dimension: {config['embed_dim']}")
    print(f"   - Fusion layers: {config['num_fusion_layers']}")
    print(f"   - Attention heads: {config['num_heads']}")
    print(f"\n2. Terrain-Aware Token-KAN:")
    print(f"   - Token-KAN layers: {config['num_token_kan_layers']}")
    print(f"   - KAN hidden dims: {config['kan_hidden_dims']}")
    print(f"\n3. Forecast Head:")
    print(f"   - Output: {config['forecast_horizon']}-day temperature forecast")
    print("-" * 50)


if __name__ == '__main__':
    main()
