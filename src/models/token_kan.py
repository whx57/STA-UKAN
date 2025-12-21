"""
Token-KAN module with terrain awareness for STA-UKAN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan import KAN


class TerrainEncoder(nn.Module):
    """Encoder for terrain features."""
    
    def __init__(self, terrain_dim, hidden_dim):
        super(TerrainEncoder, self).__init__()
        self.terrain_dim = terrain_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(terrain_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
    def forward(self, terrain_features):
        """
        Args:
            terrain_features: (batch_size, terrain_dim)
        Returns:
            Encoded terrain: (batch_size, hidden_dim)
        """
        return self.encoder(terrain_features)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
        Returns:
            Output: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        output = self.out_proj(attn_output)
        
        return output


class TokenKANBlock(nn.Module):
    """Token-KAN block combining attention and KAN layers."""
    
    def __init__(self, embed_dim, num_heads, kan_dims, dropout=0.1):
        super(TokenKANBlock, self).__init__()
        self.embed_dim = embed_dim
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        # KAN transformation
        self.kan = KAN(kan_dims)
        self.kan_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, terrain_embed=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            terrain_embed: Optional terrain embedding (batch_size, embed_dim)
        Returns:
            Output: (batch_size, seq_len, embed_dim)
        """
        # Attention with residual
        attn_out = self.attention(x)
        x = x + self.dropout(attn_out)
        x = self.attn_norm(x)
        
        # Apply terrain modulation if provided
        if terrain_embed is not None:
            terrain_embed = terrain_embed.unsqueeze(1)  # (batch, 1, embed_dim)
            x = x * torch.sigmoid(terrain_embed)
        
        # KAN transformation with residual
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(batch_size * seq_len, self.embed_dim)
        kan_out = self.kan(x_flat)
        kan_out = kan_out.reshape(batch_size, seq_len, self.embed_dim)
        
        x = x + self.dropout(kan_out)
        x = self.kan_norm(x)
        
        return x


class TerrainAwareTokenKAN(nn.Module):
    """Terrain-Aware Token-KAN module."""
    
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        num_layers,
        terrain_dim,
        kan_hidden_dims,
        dropout=0.1
    ):
        """
        Args:
            input_dim: Dimension of input features
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of Token-KAN blocks
            terrain_dim: Dimension of terrain features
            kan_hidden_dims: Hidden dimensions for KAN layers
            dropout: Dropout rate
        """
        super(TerrainAwareTokenKAN, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Terrain encoder
        self.terrain_encoder = TerrainEncoder(terrain_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, embed_dim) * 0.02)
        
        # Token-KAN blocks
        kan_dims = [embed_dim] + kan_hidden_dims + [embed_dim]
        self.blocks = nn.ModuleList([
            TokenKANBlock(embed_dim, num_heads, kan_dims, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, terrain_features):
        """
        Args:
            x: Input features (batch_size, seq_len, input_dim)
            terrain_features: Terrain features (batch_size, terrain_dim)
        Returns:
            Output: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Encode terrain
        terrain_embed = self.terrain_encoder(terrain_features)
        
        # Apply Token-KAN blocks
        for block in self.blocks:
            x = block(x, terrain_embed)
        
        return x
