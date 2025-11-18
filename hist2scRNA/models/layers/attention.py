"""
Multi-Head Self-Attention Mechanism

Implements the attention mechanism for Vision Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for Vision Transformer
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n_tokens, embed_dim = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, n_tokens, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, n_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = attn_probs @ v  # (batch_size, num_heads, n_tokens, head_dim)
        attn_output = attn_output.transpose(1, 2)  # (batch_size, n_tokens, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, n_tokens, embed_dim)

        # Final projection
        output = self.proj(attn_output)
        output = self.dropout(output)

        return output
