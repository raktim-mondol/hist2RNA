"""
Data handling modules for hist2scRNA

Includes dataset classes, dataloaders, transforms, and data generators.
"""

from .dataset import SpatialTranscriptomicsDataset
from .dataloader import collate_spatial_batch, build_batch_edge_index
from .transforms import get_default_transforms, HistoNormalize, RandomHistoAugmentation
from .generators import (
    generate_spatial_coordinates,
    generate_spatial_edges,
    generate_cell_type_labels,
    generate_gene_expression_zinb,
    generate_histopathology_patches,
    save_dummy_data
)

__all__ = [
    'SpatialTranscriptomicsDataset',
    'collate_spatial_batch',
    'build_batch_edge_index',
    'get_default_transforms',
    'HistoNormalize',
    'RandomHistoAugmentation',
    'generate_spatial_coordinates',
    'generate_spatial_edges',
    'generate_cell_type_labels',
    'generate_gene_expression_zinb',
    'generate_histopathology_patches',
    'save_dummy_data',
]
