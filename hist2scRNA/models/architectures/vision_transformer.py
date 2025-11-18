"""
Vision Transformer for Histopathology Image Feature Extraction

Implements a Vision Transformer (ViT) specifically designed for
histopathology image analysis.
"""

import torch
import torch.nn as nn
from ..layers import PatchEmbedding, TransformerBlock


class VisionTransformer(nn.Module):
    """
    Vision Transformer for histopathology image feature extraction
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        # Class token for global image representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return both class token and patch tokens
        return x[:, 0], x[:, 1:]  # (batch_size, embed_dim), (batch_size, n_patches, embed_dim)
