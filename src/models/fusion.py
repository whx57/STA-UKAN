"""
Multisource atmospheric factor fusion module for STA-UKAN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorEncoder(nn.Module):
    """Encoder for individual atmospheric factors."""
    
    def __init__(self, input_dim, hidden_dim):
        super(FactorEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            Encoded features: (batch_size, seq_len, hidden_dim)
        """
        return self.encoder(x)


class CrossFactorAttention(nn.Module):
    """Cross-attention between different atmospheric factors."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossFactorAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        """
        Args:
            query: (batch_size, seq_len, embed_dim)
            key_value: (batch_size, seq_len, embed_dim)
        Returns:
            Output: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        q = self.q_proj(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        output = self.out_proj(attn_output)
        
        return output


class MultisourceFusion(nn.Module):
    """Multisource atmospheric factor fusion module."""
    
    def __init__(
        self,
        factor_dims,
        embed_dim,
        num_heads,
        num_fusion_layers=2,
        dropout=0.1
    ):
        """
        Args:
            factor_dims: Dictionary mapping factor names to their dimensions
                        e.g., {'temperature': 10, 'pressure': 8, 'humidity': 6}
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_fusion_layers: Number of fusion layers
            dropout: Dropout rate
        """
        super(MultisourceFusion, self).__init__()
        self.factor_names = list(factor_dims.keys())
        self.embed_dim = embed_dim
        
        # Factor encoders
        self.factor_encoders = nn.ModuleDict({
            name: FactorEncoder(dim, embed_dim)
            for name, dim in factor_dims.items()
        })
        
        # Cross-factor attention layers
        self.cross_attentions = nn.ModuleList([
            nn.ModuleDict({
                name: CrossFactorAttention(embed_dim, num_heads, dropout)
                for name in self.factor_names
            })
            for _ in range(num_fusion_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({
                name: nn.LayerNorm(embed_dim)
                for name in self.factor_names
            })
            for _ in range(num_fusion_layers)
        ])
        
        # Fusion projection
        self.fusion_proj = nn.Linear(embed_dim * len(self.factor_names), embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, factor_inputs):
        """
        Args:
            factor_inputs: Dictionary of factor tensors
                          {name: (batch_size, seq_len, factor_dim)}
        Returns:
            Fused features: (batch_size, seq_len, embed_dim)
        """
        # Encode each factor
        encoded_factors = {}
        for name in self.factor_names:
            encoded_factors[name] = self.factor_encoders[name](factor_inputs[name])
        
        # Apply cross-factor attention layers
        for cross_attn_dict, norm_dict in zip(self.cross_attentions, self.layer_norms):
            updated_factors = {}
            
            for name in self.factor_names:
                # Aggregate all other factors
                other_factors = [
                    encoded_factors[other_name]
                    for other_name in self.factor_names
                    if other_name != name
                ]
                
                # If only one factor, use self-attention
                if len(other_factors) == 0:
                    aggregated = encoded_factors[name]
                else:
                    aggregated = torch.stack(other_factors, dim=0).mean(dim=0)
                
                # Cross-attention
                attn_out = cross_attn_dict[name](encoded_factors[name], aggregated)
                updated = encoded_factors[name] + self.dropout(attn_out)
                updated_factors[name] = norm_dict[name](updated)
            
            encoded_factors = updated_factors
        
        # Concatenate and fuse
        all_factors = torch.cat([encoded_factors[name] for name in self.factor_names], dim=-1)
        fused = self.fusion_proj(all_factors)
        fused = self.fusion_norm(fused)
        
        return fused
