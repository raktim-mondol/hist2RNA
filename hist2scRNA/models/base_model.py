"""
Base hist2scRNA Model

State-of-the-art model for single-cell RNA-seq prediction from histopathology images.

This model combines:
1. Vision Transformer (ViT) for patch-level feature extraction
2. Graph Neural Networks (GNN) for spatial relationship modeling
3. Spatial attention mechanisms
4. Zero-Inflated Negative Binomial (ZINB) loss for single-cell data sparsity

Based on recent SOTA methods: GHIST, Hist2ST, TransformerST, and HisToGene
"""

import torch
import torch.nn as nn
from .architectures import VisionTransformer, SpatialGraphAttention


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
