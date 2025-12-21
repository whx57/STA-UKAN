"""
Training script for STA-UKAN model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import yaml
import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import create_sta_ukan_model
from data import create_dataloader, DataNormalizer
from utils import Trainer, evaluate_model, plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train STA-UKAN model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Path to checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_synthetic_data(num_samples=1000, seq_len=30, forecast_horizon=14):
    """
    Create synthetic data for testing.
    
    In practice, replace this with real atmospheric data loading.
    """
    # Synthetic atmospheric factors
    factor_data = {
        'temperature': np.random.randn(num_samples, seq_len, 10).astype(np.float32),
        'pressure': np.random.randn(num_samples, seq_len, 8).astype(np.float32),
        'humidity': np.random.randn(num_samples, seq_len, 6).astype(np.float32),
        'wind': np.random.randn(num_samples, seq_len, 8).astype(np.float32),
    }
    
    # Synthetic terrain features
    terrain_data = np.random.randn(num_samples, 5).astype(np.float32)
    
    # Synthetic target temperatures
    target_data = np.random.randn(num_samples, forecast_horizon).astype(np.float32)
    
    return factor_data, terrain_data, target_data


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data
    print('Loading data...')
    # In practice, load real data here
    # For now, using synthetic data
    train_factor_data, train_terrain_data, train_target_data = create_synthetic_data(
        num_samples=config['data']['train_samples'],
        seq_len=config['data']['seq_len'],
        forecast_horizon=config['model']['forecast_horizon']
    )
    
    val_factor_data, val_terrain_data, val_target_data = create_synthetic_data(
        num_samples=config['data']['val_samples'],
        seq_len=config['data']['seq_len'],
        forecast_horizon=config['model']['forecast_horizon']
    )
    
    # Normalize data
    normalizer = DataNormalizer()
    normalizer.fit(train_factor_data, train_terrain_data, train_target_data)
    
    train_factor_data, train_terrain_data, train_target_data = normalizer.transform(
        train_factor_data, train_terrain_data, train_target_data
    )
    
    val_factor_data, val_terrain_data, val_target_data = normalizer.transform(
        val_factor_data, val_terrain_data, val_target_data
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_factor_data,
        train_terrain_data,
        train_target_data,
        batch_size=config['training']['batch_size'],
        seq_len=config['data']['seq_len'],
        forecast_horizon=config['model']['forecast_horizon'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    val_loader = create_dataloader(
        val_factor_data,
        val_terrain_data,
        val_target_data,
        batch_size=config['training']['batch_size'],
        seq_len=config['data']['seq_len'],
        forecast_horizon=config['model']['forecast_horizon'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    
    # Create model
    print('Creating model...')
    model = create_sta_ukan_model(config['model'])
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Create loss function and optimizer
    criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training'].get('min_lr', 1e-6)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=config['training'].get('log_interval', 10)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        start_epoch = trainer.load_checkpoint(args.resume) + 1
    
    # Train model
    print('Starting training...')
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    # Plot training history
    plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
    plot_training_history(trainer.train_losses, trainer.val_losses, save_path=plot_path)
    
    # Evaluate on validation set
    print('Evaluating model...')
    metrics, predictions, targets = evaluate_model(
        model,
        val_loader,
        device=device,
        denormalizer=normalizer.inverse_transform_target
    )
    
    print('\nValidation Metrics:')
    for metric_name, value in metrics.items():
        print(f'{metric_name}: {value:.4f}')
    
    print(f'\nTraining completed. Best validation loss: {trainer.best_val_loss:.4f}')
    print(f'Checkpoints saved to: {args.checkpoint_dir}')


if __name__ == '__main__':
    main()
