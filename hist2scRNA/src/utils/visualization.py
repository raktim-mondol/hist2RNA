"""
Visualization utilities for hist2scRNA

Provides plotting functions for training curves, predictions, and analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(train_losses, val_losses, save_path):
    """
    Plot training and validation loss curves

    Args:
        train_losses: dict with 'total', 'zinb', 'ce' keys
        val_losses: dict with 'total', 'zinb', 'ce' keys
        save_path: path to save the plot
    """
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses['total'], label='Train', marker='o', markersize=3)
    plt.plot(val_losses['total'], label='Validation', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(train_losses['zinb'], label='Train', marker='o', markersize=3)
    plt.plot(val_losses['zinb'], label='Validation', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('ZINB Loss')
    plt.title('ZINB Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(train_losses['ce'], label='Train', marker='o', markersize=3)
    plt.plot(val_losses['ce'], label='Validation', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Cell Type Loss')
    plt.title('Cell Type Classification Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_predictions_vs_true(y_true, y_pred, save_path, n_samples=1000):
    """
    Plot predicted vs true gene expression

    Args:
        y_true: true expression values
        y_pred: predicted expression values
        save_path: path to save the plot
        n_samples: number of samples to plot
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Subsample for visualization
    if len(y_true_flat) > n_samples:
        idx = np.random.choice(len(y_true_flat), n_samples, replace=False)
        y_true_flat = y_true_flat[idx]
        y_pred_flat = y_pred_flat[idx]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.5, s=10)

    # Add diagonal line
    max_val = max(np.max(y_true_flat), np.max(y_pred_flat))
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')

    plt.xlabel('True Expression', fontsize=12)
    plt.ylabel('Predicted Expression', fontsize=12)
    plt.title('Gene Expression: Predicted vs True', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction plot to {save_path}")


def plot_spatial_expression(coordinates, expression, gene_idx, save_path, title=None):
    """
    Plot spatial distribution of gene expression

    Args:
        coordinates: (n_spots, 2) array of spatial coordinates
        expression: (n_spots, n_genes) array of expression values
        gene_idx: index of gene to plot
        save_path: path to save the plot
        title: optional title for the plot
    """
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=expression[:, gene_idx],
        cmap='viridis',
        s=100,
        alpha=0.8
    )

    plt.colorbar(scatter, label='Expression Level')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(title or f'Spatial Expression Pattern - Gene {gene_idx}')
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial expression plot to {save_path}")
