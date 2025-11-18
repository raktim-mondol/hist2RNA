"""
Dataset classes for spatial transcriptomics data

Provides PyTorch Dataset implementation for loading spatial transcriptomics
data with histopathology images, gene expression, and spatial information.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import os
import json


class SpatialTranscriptomicsDataset(Dataset):
    """
    Dataset for spatial transcriptomics with histopathology images
    """
    def __init__(self, data_dir, transform=None, use_images=True):
        """
        Args:
            data_dir: directory containing the data
            transform: optional transform for images
            use_images: whether to load images or use pre-extracted features
        """
        self.data_dir = data_dir
        self.transform = transform
        self.use_images = use_images

        # Load data
        self.expression = pd.read_csv(os.path.join(data_dir, 'gene_expression.csv'), index_col='spot_id')
        self.coordinates = pd.read_csv(os.path.join(data_dir, 'spatial_coordinates.csv'))
        self.cell_types = pd.read_csv(os.path.join(data_dir, 'cell_types.csv'))
        self.edges = pd.read_csv(os.path.join(data_dir, 'spatial_edges.csv'))

        # Load metadata
        with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)

        self.spot_ids = self.expression.index.tolist()
        self.n_spots = len(self.spot_ids)
        self.n_genes = self.expression.shape[1]

        # Build edge index tensor
        self.edge_index = torch.tensor(self.edges.values.T, dtype=torch.long)

        print(f"Loaded dataset from {data_dir}")
        print(f"  - Spots: {self.n_spots}")
        print(f"  - Genes: {self.n_genes}")
        print(f"  - Edges: {self.edge_index.shape[1]}")

    def __len__(self):
        return self.n_spots

    def __getitem__(self, idx):
        spot_id = self.spot_ids[idx]

        # Load image
        if self.use_images:
            img_path = os.path.join(self.data_dir, 'patches', f'{spot_id}.png')
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            else:
                # Default transform: convert to tensor and normalize
                image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        else:
            # Return dummy tensor if not using images
            image = torch.zeros(3, 224, 224)

        # Load gene expression
        expression = torch.tensor(self.expression.loc[spot_id].values, dtype=torch.float32)

        # Load cell type
        cell_type = torch.tensor(self.cell_types[self.cell_types['spot_id'] == spot_id]['cell_type'].values[0], dtype=torch.long)

        # Load coordinates
        coord = torch.tensor(
            self.coordinates[self.coordinates['spot_id'] == spot_id][['x', 'y']].values[0],
            dtype=torch.float32
        )

        return {
            'image': image,
            'expression': expression,
            'cell_type': cell_type,
            'coordinate': coord,
            'spot_id': spot_id,
            'idx': idx
        }
