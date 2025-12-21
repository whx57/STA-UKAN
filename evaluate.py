"""
Evaluation script for STA-UKAN model.
"""

import torch
import argparse
import yaml
import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import create_sta_ukan_model
from data import create_dataloader, DataNormalizer
from utils import evaluate_model, plot_predictions, plot_error_distribution


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate STA-UKAN model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Path to output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_synthetic_data(num_samples=200, seq_len=30, forecast_horizon=14):
    """
    Create synthetic test data.
    
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
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print('Loading test data...')
    # In practice, load real test data here
    test_factor_data, test_terrain_data, test_target_data = create_synthetic_data(
        num_samples=config['data']['test_samples'],
        seq_len=config['data']['seq_len'],
        forecast_horizon=config['model']['forecast_horizon']
    )
    
    # Normalize data (in practice, use statistics from training set)
    normalizer = DataNormalizer()
    normalizer.fit(test_factor_data, test_terrain_data, test_target_data)
    
    test_factor_data, test_terrain_data, test_target_data = normalizer.transform(
        test_factor_data, test_terrain_data, test_target_data
    )
    
    # Create test data loader
    test_loader = create_dataloader(
        test_factor_data,
        test_terrain_data,
        test_target_data,
        batch_size=config['training']['batch_size'],
        seq_len=config['data']['seq_len'],
        forecast_horizon=config['model']['forecast_horizon'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Create model
    print('Creating model...')
    model = create_sta_ukan_model(config['model'])
    
    # Load checkpoint
    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate model
    print('Evaluating model...')
    metrics, predictions, targets = evaluate_model(
        model,
        test_loader,
        device=device,
        denormalizer=normalizer.inverse_transform_target
    )
    
    # Print metrics
    print('\nTest Metrics:')
    print('-' * 40)
    for metric_name, value in metrics.items():
        print(f'{metric_name:15s}: {value:8.4f}')
    print('-' * 40)
    
    # Save predictions
    pred_path = os.path.join(args.output_dir, 'predictions.npz')
    np.savez(pred_path, predictions=predictions, targets=targets)
    print(f'\nPredictions saved to: {pred_path}')
    
    # Plot predictions
    pred_plot_path = os.path.join(args.output_dir, 'predictions_plot.png')
    plot_predictions(predictions, targets, save_path=pred_plot_path)
    
    # Plot error distribution
    error_plot_path = os.path.join(args.output_dir, 'error_distribution.png')
    plot_error_distribution(predictions, targets, save_path=error_plot_path)
    
    print(f'\nEvaluation completed. Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
