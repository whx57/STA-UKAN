"""
Integration test for STA-UKAN model.
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import STA_UKAN, create_sta_ukan_model, KAN, MultisourceFusion, TerrainAwareTokenKAN
from data import AtmosphericDataset, DataNormalizer, create_dataloader
from utils import Trainer, evaluate_model


def test_kan():
    """Test KAN module."""
    print("Testing KAN module...")
    kan = KAN(layer_dims=[64, 128, 64, 32], grid_size=5, spline_order=3)
    x = torch.randn(8, 64)
    output = kan(x)
    assert output.shape == (8, 32), f"Expected shape (8, 32), got {output.shape}"
    print("✓ KAN module test passed")


def test_multisource_fusion():
    """Test Multisource Fusion module."""
    print("Testing Multisource Fusion module...")
    fusion = MultisourceFusion(
        factor_dims={'temp': 10, 'pressure': 8, 'humidity': 6},
        embed_dim=64,
        num_heads=4,
        num_fusion_layers=2
    )
    
    factor_inputs = {
        'temp': torch.randn(4, 20, 10),
        'pressure': torch.randn(4, 20, 8),
        'humidity': torch.randn(4, 20, 6),
    }
    
    output = fusion(factor_inputs)
    assert output.shape == (4, 20, 64), f"Expected shape (4, 20, 64), got {output.shape}"
    print("✓ Multisource Fusion module test passed")


def test_terrain_aware_token_kan():
    """Test Terrain-Aware Token-KAN module."""
    print("Testing Terrain-Aware Token-KAN module...")
    token_kan = TerrainAwareTokenKAN(
        input_dim=64,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        terrain_dim=5,
        kan_hidden_dims=[128, 64]
    )
    
    x = torch.randn(4, 20, 64)
    terrain = torch.randn(4, 5)
    output = token_kan(x, terrain)
    
    assert output.shape == (4, 20, 64), f"Expected shape (4, 20, 64), got {output.shape}"
    print("✓ Terrain-Aware Token-KAN module test passed")


def test_sta_ukan_model():
    """Test complete STA-UKAN model."""
    print("Testing complete STA-UKAN model...")
    
    config = {
        'factor_dims': {
            'temperature': 10,
            'pressure': 8,
            'humidity': 6,
            'wind': 8
        },
        'terrain_dim': 5,
        'embed_dim': 64,
        'num_heads': 4,
        'num_fusion_layers': 2,
        'num_token_kan_layers': 2,
        'kan_hidden_dims': [128, 64],
        'forecast_horizon': 14,
        'dropout': 0.1
    }
    
    model = create_sta_ukan_model(config)
    
    # Test forward pass
    batch_size, seq_len = 4, 30
    factor_inputs = {
        'temperature': torch.randn(batch_size, seq_len, 10),
        'pressure': torch.randn(batch_size, seq_len, 8),
        'humidity': torch.randn(batch_size, seq_len, 6),
        'wind': torch.randn(batch_size, seq_len, 8),
    }
    terrain_features = torch.randn(batch_size, 5)
    
    output = model(factor_inputs, terrain_features)
    assert output.shape == (batch_size, 14), f"Expected shape (4, 14), got {output.shape}"
    
    # Test predict method
    predictions = model.predict(factor_inputs, terrain_features)
    assert predictions.shape == (batch_size, 14), f"Expected shape (4, 14), got {predictions.shape}"
    
    print("✓ STA-UKAN model test passed")


def test_data_loader():
    """Test data loading utilities."""
    print("Testing data loading utilities...")
    
    # Create synthetic data
    num_samples = 100
    seq_len = 30
    forecast_horizon = 14
    
    factor_data = {
        'temperature': np.random.randn(num_samples, seq_len, 10).astype(np.float32),
        'pressure': np.random.randn(num_samples, seq_len, 8).astype(np.float32),
    }
    terrain_data = np.random.randn(num_samples, 5).astype(np.float32)
    target_data = np.random.randn(num_samples, forecast_horizon).astype(np.float32)
    
    # Test dataset
    dataset = AtmosphericDataset(
        factor_data=factor_data,
        terrain_data=terrain_data,
        target_data=target_data,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon
    )
    
    assert len(dataset) == num_samples, f"Expected {num_samples} samples, got {len(dataset)}"
    
    # Test data normalizer
    normalizer = DataNormalizer()
    normalizer.fit(factor_data, terrain_data, target_data)
    
    normalized_factors, normalized_terrain, normalized_target = normalizer.transform(
        factor_data, terrain_data, target_data
    )
    
    # Test inverse transform
    denormalized_target = normalizer.inverse_transform_target(normalized_target)
    assert np.allclose(denormalized_target, target_data, rtol=1e-5), "Inverse transform failed"
    
    print("✓ Data loading utilities test passed")


def test_backward_pass():
    """Test backward pass and gradient computation."""
    print("Testing backward pass...")
    
    model = STA_UKAN(
        factor_dims={'temp': 10, 'pressure': 8},
        terrain_dim=5,
        embed_dim=64,
        num_heads=4,
        forecast_horizon=14
    )
    
    # Create input
    factor_inputs = {
        'temp': torch.randn(2, 20, 10, requires_grad=True),
        'pressure': torch.randn(2, 20, 8, requires_grad=True),
    }
    terrain_features = torch.randn(2, 5, requires_grad=True)
    target = torch.randn(2, 14)
    
    # Forward pass
    output = model(factor_inputs, terrain_features)
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check that at least some gradients exist
    num_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    
    assert num_params_with_grad > 0, "No gradients computed"
    print(f"  Gradients computed for {num_params_with_grad}/{total_params} parameters")
    
    print("✓ Backward pass test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("STA-UKAN Integration Tests")
    print("=" * 60)
    print()
    
    try:
        test_kan()
        test_multisource_fusion()
        test_terrain_aware_token_kan()
        test_sta_ukan_model()
        test_data_loader()
        test_backward_pass()
        
        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
        return 0
    
    except Exception as e:
        print()
        print("=" * 60)
        print(f"Test failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
