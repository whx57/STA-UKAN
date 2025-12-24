"""
STA-UKAN Inference Script
=========================
Load pretrained checkpoint and run inference on test data.

Usage:
    python inference.py --model CUKan2_1_real --week 5week --ckpt checkpoints/5week/model.pt
"""

import argparse
import os
import torch
import numpy as np

from data.data_loader import load_data, prepare_dataloaders
from model.model_factory import get_model
from utils.utils import set_seed
from utils.evaluation import test_model
from model.net import MaskedCharbonnierLoss
from config_loader import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='STA-UKAN Inference')
    parser.add_argument('--model', type=str, default='CUKan2_1_real',
                        help='Model name (e.g., UNet, CUKan2_1_real)')
    parser.add_argument('--week', type=str, default='5week',
                        choices=['2week', '3week', '4week', '5week'],
                        help='Prediction week (2week-5week)')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--output_dir', type=str, default='output_inference',
                        help='Output directory for results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction results to file')
    return parser.parse_args()


def run_inference(args):
    """Run inference with pretrained model."""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config()
    
    # Find model configuration
    model_cfg = next((m for m in config['models'] if m.get('name', '') == args.model), None)
    if model_cfg is None:
        raise ValueError(f"Model '{args.model}' not found in config. Available models: {[m['name'] for m in config['models']]}")
    
    model_name = model_cfg['name']
    model_params = model_cfg.get('params', {})
    
    print(f"Model: {model_name}")
    print(f"Week: {args.week}")
    print(f"Checkpoint: {args.ckpt}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data paths
    x_path = f'data/output/x3_{args.week}.npy'
    y_path = f'data/output/y3_{args.week}.npy'
    m_path = 'data/MASK.npy'
    map_path = 'data/final_map1.npy'
    
    # Check data files exist
    for path in [x_path, y_path, m_path, map_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
    
    # Load data
    print("Loading data...")
    X, Y, MASK = load_data(x_path, y_path, m_path)
    
    # Prepare test dataloader
    _, _, test_loader = prepare_dataloaders(
        X, Y, MASK, args.batch_size, map_path,
        split_mode='year',
        train_sets=[21, 22, 23]
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Build model
    print("Building model...")
    model = get_model(model_name, **model_params)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}...")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    
    checkpoint = torch.load(args.ckpt, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Loss function
    criterion = MaskedCharbonnierLoss(MASK)
    
    # Run evaluation
    print("\n" + "="*50)
    print("Running inference on test set...")
    print("="*50)
    
    logs_dir = args.output_dir
    config_name = f"inference_{args.model}_{args.week}"
    
    test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        model_name=f"{model_name}_inference",
        config_name=config_name,
        logs_dir=logs_dir,
        device=device,
        week=args.week
    )
    
    # Optionally save predictions
    if args.save_predictions:
        print("\nSaving predictions...")
        save_predictions(model, test_loader, device, args.output_dir, args.week, model_name)
    
    print(f"\nInference completed! Results saved to: {args.output_dir}")


def save_predictions(model, test_loader, device, output_dir, week, model_name):
    """Save model predictions to numpy file."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, masks in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    pred_path = os.path.join(output_dir, f'{week}_{model_name}_predictions.npy')
    label_path = os.path.join(output_dir, f'{week}_{model_name}_labels.npy')
    
    np.save(pred_path, predictions)
    np.save(label_path, labels)
    
    print(f"Predictions saved to: {pred_path}")
    print(f"Labels saved to: {label_path}")
    print(f"Predictions shape: {predictions.shape}")


if __name__ == '__main__':
    args = parse_args()
    run_inference(args)
