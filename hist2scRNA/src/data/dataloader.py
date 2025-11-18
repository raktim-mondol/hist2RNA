"""
DataLoader utilities and custom collate functions

Provides utilities for batching spatial transcriptomics data
with proper handling of graph structures.
"""

import torch


def collate_spatial_batch(batch):
    """
    Custom collate function for spatial data
    Builds a batch with graph structure
    """
    images = torch.stack([item['image'] for item in batch])
    expressions = torch.stack([item['expression'] for item in batch])
    cell_types = torch.stack([item['cell_type'] for item in batch])
    coordinates = torch.stack([item['coordinate'] for item in batch])
    indices = torch.tensor([item['idx'] for item in batch])

    return {
        'images': images,
        'expressions': expressions,
        'cell_types': cell_types,
        'coordinates': coordinates,
        'indices': indices
    }


def build_batch_edge_index(edge_index_full, batch_indices):
    """
    Build edge index for a batch of spots

    Args:
        edge_index_full: full edge index tensor (2, n_edges)
        batch_indices: indices of spots in the batch

    Returns:
        batch_edge_index: filtered edge index for the batch
    """
    # Create mapping from global indices to batch indices
    index_map = {idx.item(): i for i, idx in enumerate(batch_indices)}

    # Filter edges that are within the batch
    batch_edges = []
    for i in range(edge_index_full.shape[1]):
        src, dst = edge_index_full[0, i].item(), edge_index_full[1, i].item()
        if src in index_map and dst in index_map:
            batch_edges.append([index_map[src], index_map[dst]])

    if len(batch_edges) > 0:
        batch_edge_index = torch.tensor(batch_edges, dtype=torch.long).t()
    else:
        # Return empty edge index if no edges in batch
        batch_edge_index = torch.zeros((2, 0), dtype=torch.long)

    return batch_edge_index
