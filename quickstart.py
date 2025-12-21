#!/usr/bin/env python
"""
Quick start script for STA-UKAN.
Demonstrates model usage with synthetic data.
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import create_sta_ukan_model
from data import DataNormalizer, create_dataloader
from utils import Trainer, evaluate_model


def generate_synthetic_data(num_samples, seq_len, forecast_horizon):
    """Generate synthetic atmospheric data for demonstration."""
    print(f"Generating {num_samples} synthetic samples...")
    
    factor_data = {
        'temperature': np.random.randn(num_samples, seq_len, 10).astype(np.float32),
        'pressure': np.random.randn(num_samples, seq_len, 8).astype(np.float32),
        'humidity': np.random.randn(num_samples, seq_len, 6).astype(np.float32),
        'wind': np.random.randn(num_samples, seq_len, 8).astype(np.float32),
    }
    terrain_data = np.random.randn(num_samples, 5).astype(np.float32)
    target_data = np.random.randn(num_samples, forecast_horizon).astype(np.float32)
    
    return factor_data, terrain_data, target_data


def main():
    """Run quick start demonstration."""
    print("=" * 70)
    print("STA-UKAN Quick Start")
    print("=" * 70)
    print()
    
    # Configuration
    config = {
        'factor_dims': {
            'temperature': 10,
            'pressure': 8,
            'humidity': 6,
            'wind': 8
        },
        'terrain_dim': 5,
        'embed_dim': 128,
        'num_heads': 4,
        'num_fusion_layers': 2,
        'num_token_kan_layers': 3,
        'kan_hidden_dims': [256, 128],
        'forecast_horizon': 14,
        'dropout': 0.1
    }
    
    # Data parameters
    seq_len = 30
    batch_size = 16
    num_epochs = 5
    
    # Generate synthetic data
    print("Step 1: Generating synthetic data")
    print("-" * 70)
    train_factor, train_terrain, train_target = generate_synthetic_data(
        num_samples=200, seq_len=seq_len, forecast_horizon=14
    )
    val_factor, val_terrain, val_target = generate_synthetic_data(
        num_samples=50, seq_len=seq_len, forecast_horizon=14
    )
    
    # Normalize data
    print("\nStep 2: Normalizing data")
    print("-" * 70)
    normalizer = DataNormalizer()
    normalizer.fit(train_factor, train_terrain, train_target)
    
    train_factor, train_terrain, train_target = normalizer.transform(
        train_factor, train_terrain, train_target
    )
    val_factor, val_terrain, val_target = normalizer.transform(
        val_factor, val_terrain, val_target
    )
    print("Data normalized successfully")
    
    # Create data loaders
    print("\nStep 3: Creating data loaders")
    print("-" * 70)
    train_loader = create_dataloader(
        train_factor, train_terrain, train_target,
        batch_size=batch_size, seq_len=seq_len,
        forecast_horizon=14, shuffle=True, num_workers=0
    )
    val_loader = create_dataloader(
        val_factor, val_terrain, val_target,
        batch_size=batch_size, seq_len=seq_len,
        forecast_horizon=14, shuffle=False, num_workers=0
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nStep 4: Creating STA-UKAN model")
    print("-" * 70)
    model = create_sta_ukan_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Setup training
    print("\nStep 5: Setting up training")
    print("-" * 70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir='/tmp/sta_ukan_quickstart',
        log_interval=5
    )
    
    # Train model
    print(f"\nStep 6: Training for {num_epochs} epochs")
    print("-" * 70)
    trainer.train(num_epochs=num_epochs)
    
    # Evaluate model
    print("\nStep 7: Evaluating model")
    print("-" * 70)
    metrics, predictions, targets = evaluate_model(
        model, val_loader, device=device,
        denormalizer=normalizer.inverse_transform_target
    )
    
    print("\nValidation Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:10s}: {value:8.4f}")
    
    print("\n" + "=" * 70)
    print("Quick start completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Replace synthetic data with real atmospheric data")
    print("  2. Adjust model configuration in configs/default_config.yaml")
    print("  3. Use train.py for full training with more options")
    print("  4. Use evaluate.py to evaluate trained models")
    print("=" * 70)


if __name__ == '__main__':
    main()
