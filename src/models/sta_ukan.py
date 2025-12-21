"""
Main STA-UKAN model for subseasonal temperature forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion import MultisourceFusion
from .token_kan import TerrainAwareTokenKAN


class STA_UKAN(nn.Module):
    """
    STA-UKAN: Subseasonal Temperature Forecast Refinement via 
    Multisource Atmospheric Factor Fusion and Terrain-Aware Token-KAN.
    """
    
    def __init__(
        self,
        factor_dims,
        terrain_dim,
        embed_dim=256,
        num_heads=8,
        num_fusion_layers=2,
        num_token_kan_layers=4,
        kan_hidden_dims=None,
        forecast_horizon=14,
        dropout=0.1
    ):
        """
        Args:
            factor_dims: Dictionary of atmospheric factor dimensions
                        e.g., {'temperature': 10, 'pressure': 8, 'humidity': 6}
            terrain_dim: Dimension of terrain features
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_fusion_layers: Number of fusion layers
            num_token_kan_layers: Number of Token-KAN layers
            kan_hidden_dims: Hidden dimensions for KAN layers
            forecast_horizon: Number of days to forecast
            dropout: Dropout rate
        """
        super(STA_UKAN, self).__init__()
        self.factor_dims = factor_dims
        self.terrain_dim = terrain_dim
        self.embed_dim = embed_dim
        self.forecast_horizon = forecast_horizon
        
        if kan_hidden_dims is None:
            kan_hidden_dims = [embed_dim * 2, embed_dim]
        
        # Multisource atmospheric factor fusion
        self.fusion_module = MultisourceFusion(
            factor_dims=factor_dims,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_fusion_layers=num_fusion_layers,
            dropout=dropout
        )
        
        # Terrain-aware Token-KAN
        self.token_kan = TerrainAwareTokenKAN(
            input_dim=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_token_kan_layers,
            terrain_dim=terrain_dim,
            kan_hidden_dims=kan_hidden_dims,
            dropout=dropout
        )
        
        # Temporal aggregation
        self.temporal_pooling = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        
        # Forecast head
        self.forecast_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, forecast_horizon),
        )
        
    def forward(self, factor_inputs, terrain_features):
        """
        Args:
            factor_inputs: Dictionary of atmospheric factor tensors
                          {name: (batch_size, seq_len, factor_dim)}
            terrain_features: Terrain features (batch_size, terrain_dim)
        Returns:
            Forecast: (batch_size, forecast_horizon)
        """
        # Fuse multisource atmospheric factors
        fused_features = self.fusion_module(factor_inputs)
        
        # Apply terrain-aware Token-KAN
        kan_features = self.token_kan(fused_features, terrain_features)
        
        # Temporal aggregation (mean pooling over sequence)
        pooled_features = kan_features.mean(dim=1)  # (batch_size, embed_dim)
        pooled_features = self.temporal_pooling(pooled_features)
        
        # Generate forecast
        forecast = self.forecast_head(pooled_features)
        
        return forecast
    
    def predict(self, factor_inputs, terrain_features):
        """
        Generate temperature forecast predictions.
        
        Args:
            factor_inputs: Dictionary of atmospheric factor tensors
            terrain_features: Terrain features
        Returns:
            Forecast predictions
        """
        self.eval()
        with torch.no_grad():
            forecast = self.forward(factor_inputs, terrain_features)
        return forecast


def create_sta_ukan_model(config):
    """
    Create STA-UKAN model from configuration.
    
    Args:
        config: Configuration dictionary
    Returns:
        STA-UKAN model
    """
    model = STA_UKAN(
        factor_dims=config.get('factor_dims', {
            'temperature': 10,
            'pressure': 8,
            'humidity': 6,
            'wind': 8
        }),
        terrain_dim=config.get('terrain_dim', 5),
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_fusion_layers=config.get('num_fusion_layers', 2),
        num_token_kan_layers=config.get('num_token_kan_layers', 4),
        kan_hidden_dims=config.get('kan_hidden_dims', None),
        forecast_horizon=config.get('forecast_horizon', 14),
        dropout=config.get('dropout', 0.1)
    )
    
    return model
