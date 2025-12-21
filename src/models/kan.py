"""
Kolmogorov-Arnold Network (KAN) implementation for STA-UKAN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BSpline(nn.Module):
    """B-Spline basis functions for KAN."""
    
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(BSpline, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Initialize grid points
        h = 2.0 / grid_size
        grid = torch.linspace(-1 - h * spline_order, 1 + h * spline_order, 
                            grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)
        
        # Learnable spline coefficients
        self.coefficients = nn.Parameter(
            torch.randn(in_features, out_features, grid_size + spline_order) * 0.1
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.shape[0]
        x = x.unsqueeze(-1)  # (batch_size, in_features, 1)
        
        # Normalize input to [-1, 1]
        x_normalized = torch.tanh(x)
        
        # Compute B-spline basis
        bases = self._compute_basis(x_normalized)
        
        # Apply spline transformation
        output = torch.einsum('bik,iok->bo', bases, self.coefficients)
        
        return output
    
    def _compute_basis(self, x):
        """Compute B-spline basis functions."""
        # Simplified basis computation
        grid = self.grid.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_points)
        x = x.expand(-1, -1, self.grid.shape[0])  # (batch, in_features, grid_points)
        
        # Gaussian-like basis functions (simplified)
        bases = torch.exp(-((x - grid) ** 2) / 0.5)
        bases = bases / (bases.sum(dim=-1, keepdim=True) + 1e-8)
        
        return bases[:, :, :self.grid_size + self.spline_order]


class KANLayer(nn.Module):
    """Single layer of Kolmogorov-Arnold Network."""
    
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # B-spline transformation
        self.spline = BSpline(in_features, out_features, grid_size, spline_order)
        
        # Residual connection weights
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Apply spline transformation
        output = self.spline(x)
        
        return output


class KAN(nn.Module):
    """Multi-layer Kolmogorov-Arnold Network."""
    
    def __init__(self, layer_dims, grid_size=5, spline_order=3):
        """
        Args:
            layer_dims: List of layer dimensions, e.g., [64, 128, 64, 32]
            grid_size: Size of the spline grid
            spline_order: Order of B-spline
        """
        super(KAN, self).__init__()
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        
        # Build KAN layers
        self.layers = nn.ModuleList([
            KANLayer(layer_dims[i], layer_dims[i+1], grid_size, spline_order)
            for i in range(self.num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(layer_dims[i+1])
            for i in range(self.num_layers)
        ])
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, layer_dims[0])
        Returns:
            Output tensor of shape (batch_size, layer_dims[-1])
        """
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            x = layer(x)
            x = norm(x)
            
            # Apply activation except for the last layer
            if i < self.num_layers - 1:
                x = F.gelu(x)
        
        return x
