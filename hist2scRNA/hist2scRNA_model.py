"""
hist2scRNA: State-of-the-art model for single-cell RNA-seq prediction from histopathology images

This model combines:
1. Vision Transformer (ViT) for patch-level feature extraction
2. Graph Neural Networks (GNN) for spatial relationship modeling
3. Spatial attention mechanisms
4. Zero-Inflated Negative Binomial (ZINB) loss for single-cell data sparsity

Based on recent SOTA methods: GHIST, Hist2ST, TransformerST, and HisToGene
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import math


class PatchEmbedding(nn.Module):
    """
    Convert image patches to embeddings using convolutional layers
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Convolutional layer to create patch embeddings
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.proj(x)  # (batch_size, embed_dim, n_patches^0.5, n_patches^0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


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


class SpatialGraphAttention(nn.Module):
    """
    Graph Attention Network for modeling spatial relationships between spots
    """
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.1):
        super(SpatialGraphAttention, self).__init__()

        self.gat1 = GATConv(in_channels, out_channels, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(out_channels * num_heads, out_channels, heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Second GAT layer
        x = self.gat2(x, edge_index)

        return x


class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial Loss for single-cell RNA-seq data
    Handles sparsity in single-cell data
    """
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, mu, theta, pi, target):
        """
        Args:
            mu: mean of negative binomial (batch_size, n_genes)
            theta: dispersion parameter (batch_size, n_genes)
            pi: zero-inflation probability (batch_size, n_genes)
            target: true expression values (batch_size, n_genes)
        """
        eps = 1e-10

        # Negative binomial component
        theta = torch.clamp(theta, min=eps)
        t1 = torch.lgamma(theta + eps) + torch.lgamma(target + 1.0) - torch.lgamma(target + theta + eps)
        t2 = (theta + target) * torch.log(1.0 + (mu / (theta + eps))) + (target * (torch.log(theta + eps) - torch.log(mu + eps)))
        nb_case = t1 + t2 - torch.log(1.0 - pi + eps)

        # Zero-inflation component
        zero_nb = torch.pow(theta / (theta + mu + eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)

        # Combine cases
        result = torch.where(target < eps, zero_case, nb_case)

        return result.mean()


class hist2scRNA(nn.Module):
    """
    State-of-the-art model for single-cell RNA-seq prediction from histopathology images

    Architecture:
    1. Vision Transformer for patch-level feature extraction
    2. Spatial Graph Attention Network for modeling spatial relationships
    3. Gene expression decoder with ZINB distribution
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, vit_depth=6, vit_heads=12,
                 gnn_hidden=512, gnn_heads=4,
                 n_genes=2000, n_cell_types=10,
                 use_spatial_graph=True, dropout=0.1):
        super(hist2scRNA, self).__init__()

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.use_spatial_graph = use_spatial_graph

        # Vision Transformer for feature extraction
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            dropout=dropout
        )

        # Spatial Graph Attention (if enabled)
        if use_spatial_graph:
            self.spatial_gnn = SpatialGraphAttention(
                in_channels=embed_dim,
                out_channels=gnn_hidden,
                num_heads=gnn_heads,
                dropout=dropout
            )
            decoder_input_dim = gnn_hidden
        else:
            decoder_input_dim = embed_dim

        # Gene expression decoder
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ZINB parameters prediction
        self.mu_decoder = nn.Linear(2048, n_genes)  # Mean
        self.theta_decoder = nn.Linear(2048, n_genes)  # Dispersion
        self.pi_decoder = nn.Linear(2048, n_genes)  # Zero-inflation probability

        # Cell type prediction (optional, for multi-task learning)
        self.cell_type_head = nn.Linear(2048, n_cell_types)

    def forward(self, x, edge_index=None, batch=None):
        """
        Args:
            x: input images (batch_size, channels, height, width) or spot features
            edge_index: spatial graph edges (2, n_edges) for GNN
            batch: batch assignment for graph data
        """
        # Extract features using Vision Transformer
        if len(x.shape) == 4:  # Image input
            cls_token, patch_tokens = self.vit(x)  # (batch_size, embed_dim), (batch_size, n_patches, embed_dim)
            # Use class token as spot representation
            spot_features = cls_token
        else:  # Already extracted features
            spot_features = x

        # Apply spatial graph attention if enabled
        if self.use_spatial_graph and edge_index is not None:
            spot_features = self.spatial_gnn(spot_features, edge_index, batch)

        # Decode to gene expression
        h = self.decoder(spot_features)

        # Predict ZINB parameters
        mu = torch.exp(self.mu_decoder(h))  # Mean (positive)
        theta = torch.exp(self.theta_decoder(h))  # Dispersion (positive)
        pi = torch.sigmoid(self.pi_decoder(h))  # Zero-inflation probability (0-1)

        # Cell type prediction
        cell_type_logits = self.cell_type_head(h)

        return {
            'mu': mu,
            'theta': theta,
            'pi': pi,
            'cell_type_logits': cell_type_logits,
            'features': h
        }

    def predict(self, x, edge_index=None, batch=None):
        """
        Predict gene expression (using mean of distribution)
        """
        output = self.forward(x, edge_index, batch)
        return output['mu']


class hist2scRNA_Lightweight(nn.Module):
    """
    Lightweight version of hist2scRNA for faster training and inference
    Uses smaller Vision Transformer and simplified architecture
    """
    def __init__(self, feature_dim=2048, n_genes=2000, n_cell_types=10, dropout=0.1):
        super(hist2scRNA_Lightweight, self).__init__()

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Spatial attention
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=1024,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Gene expression decoder
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ZINB parameters prediction
        self.mu_decoder = nn.Linear(2048, n_genes)
        self.theta_decoder = nn.Linear(2048, n_genes)
        self.pi_decoder = nn.Linear(2048, n_genes)

        # Cell type prediction
        self.cell_type_head = nn.Linear(2048, n_cell_types)

    def forward(self, x, spatial_positions=None):
        """
        Args:
            x: pre-extracted features (batch_size, feature_dim) or (batch_size, n_spots, feature_dim)
            spatial_positions: optional spatial coordinates for attention
        """
        # Project features
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add spot dimension

        h = self.feature_proj(x)

        # Apply spatial attention
        h, _ = self.spatial_attn(h, h, h)
        h = h.squeeze(1) if h.shape[1] == 1 else h.mean(dim=1)

        # Decode to gene expression
        h = self.decoder(h)

        # Predict ZINB parameters
        mu = torch.exp(self.mu_decoder(h))
        theta = torch.exp(self.theta_decoder(h))
        pi = torch.sigmoid(self.pi_decoder(h))

        # Cell type prediction
        cell_type_logits = self.cell_type_head(h)

        return {
            'mu': mu,
            'theta': theta,
            'pi': pi,
            'cell_type_logits': cell_type_logits,
            'features': h
        }

    def predict(self, x, spatial_positions=None):
        """Predict gene expression"""
        output = self.forward(x, spatial_positions)
        return output['mu']


if __name__ == "__main__":
    # Test the models
    print("Testing hist2scRNA model...")

    # Full model
    model = hist2scRNA(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        vit_depth=6,
        vit_heads=6,
        n_genes=2000,
        n_cell_types=10,
        use_spatial_graph=True
    )

    # Test with dummy image
    dummy_img = torch.randn(4, 3, 224, 224)

    # Create dummy spatial graph (4 spots in a 2x2 grid)
    edge_index = torch.tensor([
        [0, 0, 1, 2],
        [1, 2, 3, 3]
    ], dtype=torch.long)

    output = model(dummy_img, edge_index)
    print(f"Output shapes:")
    print(f"  mu: {output['mu'].shape}")
    print(f"  theta: {output['theta'].shape}")
    print(f"  pi: {output['pi'].shape}")
    print(f"  cell_type_logits: {output['cell_type_logits'].shape}")

    # Test lightweight model
    print("\nTesting hist2scRNA_Lightweight model...")
    model_light = hist2scRNA_Lightweight(feature_dim=2048, n_genes=2000, n_cell_types=10)

    dummy_features = torch.randn(4, 2048)
    output_light = model_light(dummy_features)
    print(f"Lightweight output shapes:")
    print(f"  mu: {output_light['mu'].shape}")
    print(f"  theta: {output_light['theta'].shape}")

    print("\nModels created successfully!")
