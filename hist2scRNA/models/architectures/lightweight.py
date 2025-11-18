"""
Lightweight hist2scRNA Model

A simplified version of hist2scRNA for faster training and inference.
Uses smaller architecture and simplified components.
"""

import torch
import torch.nn as nn


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
