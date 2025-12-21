"""
Evaluation metrics for temperature forecasting.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(predictions, targets):
    """
    Root Mean Squared Error.
    
    Args:
        predictions: Predicted values
        targets: True values
    Returns:
        RMSE score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return np.sqrt(mean_squared_error(targets, predictions))


def mae(predictions, targets):
    """
    Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        targets: True values
    Returns:
        MAE score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return mean_absolute_error(targets, predictions)


def mape(predictions, targets, epsilon=1e-8):
    """
    Mean Absolute Percentage Error.
    
    Args:
        predictions: Predicted values
        targets: True values
        epsilon: Small value to avoid division by zero
    Returns:
        MAPE score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return np.mean(np.abs((targets - predictions) / (np.abs(targets) + epsilon))) * 100


def r2_score(predictions, targets):
    """
    R-squared (coefficient of determination).
    
    Args:
        predictions: Predicted values
        targets: True values
    Returns:
        R2 score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    
    return 1 - (ss_res / (ss_tot + 1e-8))


def bias(predictions, targets):
    """
    Bias (mean error).
    
    Args:
        predictions: Predicted values
        targets: True values
    Returns:
        Bias score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    return np.mean(predictions - targets)


def skill_score(predictions, targets, baseline_predictions):
    """
    Skill score comparing model to baseline.
    
    Args:
        predictions: Model predicted values
        targets: True values
        baseline_predictions: Baseline predicted values
    Returns:
        Skill score
    """
    model_mse = mean_squared_error(targets, predictions)
    baseline_mse = mean_squared_error(targets, baseline_predictions)
    
    return 1 - (model_mse / (baseline_mse + 1e-8))


def evaluate_model(model, dataloader, device='cuda', denormalizer=None):
    """
    Evaluate model on a dataset.
    
    Args:
        model: STA-UKAN model
        dataloader: Data loader
        device: Device to use
        denormalizer: Optional denormalizer function
    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for factor_inputs, terrain_features, targets in dataloader:
            # Move data to device
            factor_inputs = {
                name: data.to(device)
                for name, data in factor_inputs.items()
            }
            terrain_features = terrain_features.to(device)
            targets = targets.to(device)
            
            # Predict
            predictions = model(factor_inputs, terrain_features)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Denormalize if needed
    if denormalizer is not None:
        all_predictions = denormalizer(all_predictions.numpy())
        all_targets = denormalizer(all_targets.numpy())
    
    # Compute metrics
    metrics = {
        'RMSE': rmse(all_predictions, all_targets),
        'MAE': mae(all_predictions, all_targets),
        'MAPE': mape(all_predictions, all_targets),
        'R2': r2_score(all_predictions, all_targets),
        'Bias': bias(all_predictions, all_targets),
    }
    
    return metrics, all_predictions, all_targets
