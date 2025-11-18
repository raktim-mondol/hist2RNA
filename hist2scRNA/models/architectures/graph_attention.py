"""
Spatial Graph Attention Network

Implements Graph Attention Networks (GAT) for modeling spatial relationships
between tissue spots in spatial transcriptomics data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


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
