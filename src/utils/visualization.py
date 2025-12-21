"""
Visualization utilities for STA-UKAN.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_training_history(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss history.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved training history plot to {save_path}')
    
    plt.close()


def plot_predictions(predictions, targets, days_ahead=None, save_path=None):
    """
    Plot predicted vs actual temperatures.
    
    Args:
        predictions: Predicted temperature values (n_samples, forecast_horizon)
        targets: True temperature values (n_samples, forecast_horizon)
        days_ahead: Optional list of day indices to plot
        save_path: Optional path to save the figure
    """
    if days_ahead is None:
        days_ahead = [0, 6, 13]  # First day, week, and 2 weeks
    
    fig, axes = plt.subplots(1, len(days_ahead), figsize=(5 * len(days_ahead), 4))
    if len(days_ahead) == 1:
        axes = [axes]
    
    for idx, day in enumerate(days_ahead):
        ax = axes[idx]
        
        pred_day = predictions[:, day]
        target_day = targets[:, day]
        
        # Scatter plot
        ax.scatter(target_day, pred_day, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(target_day.min(), pred_day.min())
        max_val = max(target_day.max(), pred_day.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel('Actual Temperature', fontsize=11)
        ax.set_ylabel('Predicted Temperature', fontsize=11)
        ax.set_title(f'Day {day + 1} Forecast', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved predictions plot to {save_path}')
    
    plt.close()


def plot_forecast_horizon_metrics(metrics_by_day, save_path=None):
    """
    Plot metrics across forecast horizon.
    
    Args:
        metrics_by_day: Dictionary mapping day index to metrics dict
        save_path: Optional path to save the figure
    """
    days = sorted(metrics_by_day.keys())
    metric_names = list(metrics_by_day[days[0]].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metric_names[:4]):
        ax = axes[idx]
        values = [metrics_by_day[day][metric_name] for day in days]
        
        ax.plot(days, values, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Forecast Day', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} vs Forecast Horizon', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved forecast horizon metrics plot to {save_path}')
    
    plt.close()


def plot_error_distribution(predictions, targets, save_path=None):
    """
    Plot distribution of prediction errors.
    
    Args:
        predictions: Predicted values
        targets: True values
        save_path: Optional path to save the figure
    """
    errors = (predictions - targets).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved error distribution plot to {save_path}')
    
    plt.close()


def plot_attention_weights(attention_weights, save_path=None):
    """
    Plot attention weight heatmap.
    
    Args:
        attention_weights: Attention weight matrix
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        attention_weights,
        cmap='viridis',
        cbar=True,
        square=True,
        linewidths=0.5,
        annot=False
    )
    
    plt.xlabel('Key Position', fontsize=11)
    plt.ylabel('Query Position', fontsize=11)
    plt.title('Attention Weights Heatmap', fontsize=12, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved attention weights plot to {save_path}')
    
    plt.close()
