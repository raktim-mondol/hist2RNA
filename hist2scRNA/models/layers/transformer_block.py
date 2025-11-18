"""
Transformer Block

Combines multi-head self-attention and feed-forward network with residual connections.
"""

import torch.nn as nn
from .attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and feed-forward network
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        return x
