"""
Dummy data generators for testing

Generates synthetic spatial transcriptomics data for model testing and development.
"""

import numpy as np
import torch
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from scipy.spatial import KDTree
from sklearn.mixture import GaussianMixture
import json


def generate_spatial_coordinates(n_spots, grid_size=10, noise_level=0.1):
    """
    Generate spatial coordinates for spots in a grid pattern with noise

    Args:
        n_spots: number of spots
        grid_size: size of the grid
        noise_level: amount of random noise to add to coordinates

    Returns:
        coordinates: (n_spots, 2) array of (x, y) coordinates
    """
    grid_dim = int(np.ceil(np.sqrt(n_spots)))
    x = np.linspace(0, grid_size, grid_dim)
    y = np.linspace(0, grid_size, grid_dim)

    xx, yy = np.meshgrid(x, y)
    coordinates = np.column_stack([xx.ravel(), yy.ravel()])

    # Take only n_spots
    coordinates = coordinates[:n_spots]

    # Add noise
    coordinates += np.random.randn(n_spots, 2) * noise_level

    return coordinates


def generate_spatial_edges(coordinates, k_neighbors=6):
    """
    Generate edges for spatial graph based on k-nearest neighbors

    Args:
        coordinates: (n_spots, 2) spatial coordinates
        k_neighbors: number of nearest neighbors

    Returns:
        edge_index: (2, n_edges) edge list
    """
    n_spots = coordinates.shape[0]
    tree = KDTree(coordinates)

    edges = []
    for i in range(n_spots):
        # Find k+1 nearest neighbors (including itself)
        distances, indices = tree.query(coordinates[i], k=min(k_neighbors + 1, n_spots))

        # Add edges to all neighbors except itself
        for j in indices[1:]:
            edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index


def generate_cell_type_labels(n_spots, n_cell_types=10, spatial_clustering=True, coordinates=None):
    """
    Generate cell type labels with optional spatial clustering

    Args:
        n_spots: number of spots
        n_cell_types: number of distinct cell types
        spatial_clustering: whether cell types should be spatially clustered
        coordinates: spatial coordinates for clustering

    Returns:
        cell_types: (n_spots,) array of cell type labels
    """
    if spatial_clustering and coordinates is not None:
        # Use Gaussian mixture-like approach for spatial clustering
        gmm = GaussianMixture(n_components=n_cell_types, random_state=42)
        cell_types = gmm.fit_predict(coordinates)
    else:
        # Random cell type assignment
        cell_types = np.random.randint(0, n_cell_types, size=n_spots)

    return cell_types


def generate_gene_expression_zinb(n_spots, n_genes, cell_types, n_cell_types,
                                   mean_expression=5.0, dispersion=2.0,
                                   zero_inflation=0.7, cell_type_effect=3.0):
    """
    Generate single-cell gene expression using Zero-Inflated Negative Binomial distribution

    Args:
        n_spots: number of spots/cells
        n_genes: number of genes
        cell_types: cell type labels for each spot
        n_cell_types: total number of cell types
        mean_expression: base mean expression level
        dispersion: overdispersion parameter
        zero_inflation: probability of zero inflation
        cell_type_effect: strength of cell type-specific expression

    Returns:
        expression: (n_spots, n_genes) gene expression matrix
    """
    expression = np.zeros((n_spots, n_genes))

    # Generate cell type-specific gene programs
    genes_per_type = n_genes // n_cell_types

    for spot_idx in range(n_spots):
        cell_type = cell_types[spot_idx]

        for gene_idx in range(n_genes):
            # Determine if this gene is a marker for this cell type
            marker_type = gene_idx // genes_per_type
            is_marker = (marker_type == cell_type)

            # Base mean expression
            mu = mean_expression

            # Increase expression for marker genes
            if is_marker:
                mu *= cell_type_effect

            # Add random variation
            mu *= np.random.lognormal(0, 0.5)

            # Zero inflation
            if np.random.rand() < zero_inflation:
                expression[spot_idx, gene_idx] = 0
            else:
                # Sample from negative binomial
                var = mu + (mu ** 2) / dispersion
                p = mu / var
                n = mu * p / (1 - p)

                # Ensure valid parameters
                if p > 0 and p < 1 and n > 0:
                    expression[spot_idx, gene_idx] = nbinom.rvs(n=n, p=1-p)
                else:
                    expression[spot_idx, gene_idx] = np.random.poisson(mu)

    return expression


def generate_histopathology_patches(n_spots, img_size=224, output_dir='./patches'):
    """
    Generate synthetic histopathology image patches

    Args:
        n_spots: number of patches to generate
        img_size: size of each patch
        output_dir: directory to save patches

    Returns:
        patch_paths: list of paths to generated patches
    """
    os.makedirs(output_dir, exist_ok=True)

    patch_paths = []

    for i in range(n_spots):
        # Create synthetic H&E-like image
        background = np.random.randint(200, 240, size=(img_size, img_size, 3), dtype=np.uint8)
        background[:, :, 0] = np.random.randint(220, 245, size=(img_size, img_size))
        background[:, :, 1] = np.random.randint(180, 220, size=(img_size, img_size))
        background[:, :, 2] = np.random.randint(200, 230, size=(img_size, img_size))

        # Add nuclei
        n_nuclei = np.random.randint(20, 50)
        for _ in range(n_nuclei):
            x = np.random.randint(10, img_size - 10)
            y = np.random.randint(10, img_size - 10)
            radius = np.random.randint(3, 8)

            Y, X = np.ogrid[:img_size, :img_size]
            dist = np.sqrt((X - x)**2 + (Y - y)**2)
            mask = dist <= radius

            background[mask, 0] = np.random.randint(100, 150)
            background[mask, 1] = np.random.randint(80, 130)
            background[mask, 2] = np.random.randint(150, 200)

        # Add texture
        noise = np.random.randint(-10, 10, size=(img_size, img_size, 3), dtype=np.int16)
        img = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Save image
        img_pil = Image.fromarray(img)
        patch_path = os.path.join(output_dir, f'spot_{i:04d}.png')
        img_pil.save(patch_path)
        patch_paths.append(patch_path)

    return patch_paths


def save_dummy_data(n_spots=100, n_genes=2000, n_cell_types=10, output_dir='./dummy_data'):
    """
    Generate and save complete dummy single-cell RNA-seq dataset

    Args:
        n_spots: number of spatial spots/cells
        n_genes: number of genes
        n_cell_types: number of cell types
        output_dir: directory to save all data
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating dummy single-cell RNA-seq data...")
    print(f"  - Number of spots: {n_spots}")
    print(f"  - Number of genes: {n_genes}")
    print(f"  - Number of cell types: {n_cell_types}")

    # 1. Generate spatial coordinates
    print("\n1. Generating spatial coordinates...")
    coordinates = generate_spatial_coordinates(n_spots, grid_size=10, noise_level=0.2)

    coord_df = pd.DataFrame(coordinates, columns=['x', 'y'])
    coord_df['spot_id'] = [f'spot_{i:04d}' for i in range(n_spots)]
    coord_df.to_csv(os.path.join(output_dir, 'spatial_coordinates.csv'), index=False)
    print(f"   Saved to: {output_dir}/spatial_coordinates.csv")

    # 2. Generate spatial edges
    print("\n2. Generating spatial graph edges...")
    edge_index = generate_spatial_edges(coordinates, k_neighbors=6)

    edge_df = pd.DataFrame(edge_index.t().numpy(), columns=['source', 'target'])
    edge_df.to_csv(os.path.join(output_dir, 'spatial_edges.csv'), index=False)
    print(f"   Saved to: {output_dir}/spatial_edges.csv")
    print(f"   Number of edges: {edge_index.shape[1]}")

    # 3. Generate cell type labels
    print("\n3. Generating cell type labels...")
    cell_types = generate_cell_type_labels(n_spots, n_cell_types, spatial_clustering=True, coordinates=coordinates)

    celltype_df = pd.DataFrame({
        'spot_id': [f'spot_{i:04d}' for i in range(n_spots)],
        'cell_type': cell_types
    })
    celltype_df.to_csv(os.path.join(output_dir, 'cell_types.csv'), index=False)
    print(f"   Saved to: {output_dir}/cell_types.csv")

    # 4. Generate gene expression data
    print("\n4. Generating gene expression matrix...")
    expression = generate_gene_expression_zinb(
        n_spots=n_spots,
        n_genes=n_genes,
        cell_types=cell_types,
        n_cell_types=n_cell_types,
        mean_expression=5.0,
        dispersion=2.0,
        zero_inflation=0.7,
        cell_type_effect=3.0
    )

    gene_names = [f'Gene_{i:04d}' for i in range(n_genes)]
    spot_ids = [f'spot_{i:04d}' for i in range(n_spots)]

    expr_df = pd.DataFrame(expression, columns=gene_names, index=spot_ids)
    expr_df.index.name = 'spot_id'
    expr_df.to_csv(os.path.join(output_dir, 'gene_expression.csv'))
    print(f"   Saved to: {output_dir}/gene_expression.csv")

    sparsity = (expression == 0).sum() / expression.size
    print(f"\n   Expression statistics:")
    print(f"     - Sparsity: {sparsity:.2%}")
    print(f"     - Mean expression: {expression.mean():.2f}")
    print(f"     - Max expression: {expression.max():.0f}")

    # 5. Generate histopathology patches
    print("\n5. Generating synthetic histopathology patches...")
    patch_paths = generate_histopathology_patches(n_spots, img_size=224, output_dir=os.path.join(output_dir, 'patches'))
    print(f"   Saved {len(patch_paths)} patches to: {output_dir}/patches/")

    # 6. Create metadata
    print("\n6. Creating metadata...")
    metadata = {
        'n_spots': n_spots,
        'n_genes': n_genes,
        'n_cell_types': n_cell_types,
        'sparsity': float(sparsity),
        'mean_expression': float(expression.mean()),
        'max_expression': float(expression.max()),
        'description': 'Dummy single-cell RNA-seq data for testing hist2scRNA model'
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved to: {output_dir}/metadata.json")

    print("\n" + "="*60)
    print("Dummy data generation complete!")
    print("="*60)

    return {
        'coordinates': coordinates,
        'edge_index': edge_index,
        'cell_types': cell_types,
        'expression': expression,
        'metadata': metadata
    }
