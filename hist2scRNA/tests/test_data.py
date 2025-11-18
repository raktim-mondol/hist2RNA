"""
Tests for data handling modules
"""

import torch
import pytest
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    generate_spatial_coordinates,
    generate_spatial_edges,
    generate_cell_type_labels,
    save_dummy_data,
    SpatialTranscriptomicsDataset,
    collate_spatial_batch
)


def test_spatial_coordinates():
    """Test spatial coordinate generation"""
    n_spots = 50
    coordinates = generate_spatial_coordinates(n_spots, grid_size=10, noise_level=0.1)

    assert coordinates.shape == (n_spots, 2)
    assert coordinates.dtype == float
    print(f"Generated {n_spots} spatial coordinates - PASSED")


def test_spatial_edges():
    """Test spatial edge generation"""
    n_spots = 50
    coordinates = generate_spatial_coordinates(n_spots)
    edge_index = generate_spatial_edges(coordinates, k_neighbors=6)

    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0
    print(f"Generated {edge_index.shape[1]} spatial edges - PASSED")


def test_cell_type_labels():
    """Test cell type label generation"""
    n_spots = 50
    n_cell_types = 5
    coordinates = generate_spatial_coordinates(n_spots)

    cell_types = generate_cell_type_labels(
        n_spots, n_cell_types, spatial_clustering=True, coordinates=coordinates
    )

    assert cell_types.shape == (n_spots,)
    assert cell_types.min() >= 0
    assert cell_types.max() < n_cell_types
    print(f"Generated cell type labels - PASSED")


def test_dummy_data_generation():
    """Test complete dummy data generation"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Generate dummy data
        result = save_dummy_data(
            n_spots=20,
            n_genes=100,
            n_cell_types=3,
            output_dir=temp_dir
        )

        # Check files exist
        assert os.path.exists(os.path.join(temp_dir, 'gene_expression.csv'))
        assert os.path.exists(os.path.join(temp_dir, 'spatial_coordinates.csv'))
        assert os.path.exists(os.path.join(temp_dir, 'cell_types.csv'))
        assert os.path.exists(os.path.join(temp_dir, 'metadata.json'))
        assert os.path.exists(os.path.join(temp_dir, 'patches'))

        print("Dummy data generation - PASSED")

        # Test dataset loading
        dataset = SpatialTranscriptomicsDataset(temp_dir, use_images=True)

        assert len(dataset) == 20
        assert dataset.n_genes == 100

        # Test __getitem__
        sample = dataset[0]
        assert 'image' in sample
        assert 'expression' in sample
        assert 'cell_type' in sample

        print("Dataset loading - PASSED")

        # Test collate function
        batch = [dataset[i] for i in range(4)]
        collated = collate_spatial_batch(batch)

        assert 'images' in collated
        assert 'expressions' in collated
        assert collated['images'].shape[0] == 4

        print("Batch collation - PASSED")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Running data tests...\n")

    test_spatial_coordinates()
    test_spatial_edges()
    test_cell_type_labels()
    test_dummy_data_generation()

    print("\nAll data tests passed!")
