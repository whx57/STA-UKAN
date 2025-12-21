"""
Data loader for atmospheric data and terrain features.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class AtmosphericDataset(Dataset):
    """Dataset for atmospheric factors and terrain features."""
    
    def __init__(
        self,
        factor_data,
        terrain_data,
        target_data,
        seq_len=30,
        forecast_horizon=14,
        transform=None
    ):
        """
        Args:
            factor_data: Dictionary of atmospheric factor arrays
                        {name: (num_samples, time_steps, feature_dim)}
            terrain_data: Terrain feature array (num_samples, terrain_dim)
            target_data: Target temperature array (num_samples, forecast_horizon)
            seq_len: Length of input sequence
            forecast_horizon: Number of days to forecast
            transform: Optional data transformation
        """
        self.factor_data = factor_data
        self.terrain_data = terrain_data
        self.target_data = target_data
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.transform = transform
        
        # Validate data shapes
        self.num_samples = len(terrain_data)
        for name, data in factor_data.items():
            assert len(data) == self.num_samples, \
                f"Factor {name} has {len(data)} samples, expected {self.num_samples}"
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            factor_inputs: Dictionary of factor tensors
            terrain_features: Terrain tensor
            target: Target temperature tensor
        """
        # Get factor inputs
        factor_inputs = {}
        for name, data in self.factor_data.items():
            factor_inputs[name] = torch.FloatTensor(data[idx, :self.seq_len, :])
        
        # Get terrain features
        terrain_features = torch.FloatTensor(self.terrain_data[idx])
        
        # Get target
        target = torch.FloatTensor(self.target_data[idx, :self.forecast_horizon])
        
        if self.transform:
            factor_inputs, terrain_features, target = self.transform(
                factor_inputs, terrain_features, target
            )
        
        return factor_inputs, terrain_features, target


class DataNormalizer:
    """Normalizer for atmospheric data."""
    
    def __init__(self):
        self.factor_means = {}
        self.factor_stds = {}
        self.terrain_mean = None
        self.terrain_std = None
        self.target_mean = None
        self.target_std = None
        
    def fit(self, factor_data, terrain_data, target_data):
        """
        Compute normalization statistics.
        
        Args:
            factor_data: Dictionary of atmospheric factor arrays
            terrain_data: Terrain feature array
            target_data: Target temperature array
        """
        # Compute factor statistics
        for name, data in factor_data.items():
            self.factor_means[name] = np.mean(data, axis=(0, 1), keepdims=True)
            self.factor_stds[name] = np.std(data, axis=(0, 1), keepdims=True) + 1e-8
        
        # Compute terrain statistics
        self.terrain_mean = np.mean(terrain_data, axis=0, keepdims=True)
        self.terrain_std = np.std(terrain_data, axis=0, keepdims=True) + 1e-8
        
        # Compute target statistics
        self.target_mean = np.mean(target_data)
        self.target_std = np.std(target_data) + 1e-8
        
    def transform(self, factor_data, terrain_data, target_data):
        """
        Normalize data.
        
        Args:
            factor_data: Dictionary of atmospheric factor arrays
            terrain_data: Terrain feature array
            target_data: Target temperature array
        Returns:
            Normalized data
        """
        # Normalize factors
        normalized_factors = {}
        for name, data in factor_data.items():
            normalized_factors[name] = (
                (data - self.factor_means[name]) / self.factor_stds[name]
            )
        
        # Normalize terrain
        normalized_terrain = (
            (terrain_data - self.terrain_mean) / self.terrain_std
        )
        
        # Normalize target
        normalized_target = (target_data - self.target_mean) / self.target_std
        
        return normalized_factors, normalized_terrain, normalized_target
    
    def inverse_transform_target(self, normalized_target):
        """
        Denormalize target predictions.
        
        Args:
            normalized_target: Normalized target array
        Returns:
            Original scale target array
        """
        return normalized_target * self.target_std + self.target_mean


def create_dataloader(
    factor_data,
    terrain_data,
    target_data,
    batch_size=32,
    seq_len=30,
    forecast_horizon=14,
    shuffle=True,
    num_workers=4
):
    """
    Create DataLoader for training/evaluation.
    
    Args:
        factor_data: Dictionary of atmospheric factor arrays
        terrain_data: Terrain feature array
        target_data: Target temperature array
        batch_size: Batch size
        seq_len: Length of input sequence
        forecast_horizon: Number of days to forecast
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    Returns:
        DataLoader
    """
    dataset = AtmosphericDataset(
        factor_data=factor_data,
        terrain_data=terrain_data,
        target_data=target_data,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
