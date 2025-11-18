"""
Evaluation metrics for single-cell RNA-seq prediction

Provides comprehensive metrics for evaluating gene expression predictions
including correlation, regression metrics, and zero-inflation analysis.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix
)
from typing import Dict, Optional
import pandas as pd


def compute_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute overall regression metrics across all genes and spots
    """
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Remove any NaN or Inf values
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    y_true_flat = y_true_flat[mask]
    y_pred_flat = y_pred_flat[mask]

    metrics = {
        'mse': float(mean_squared_error(y_true_flat, y_pred_flat)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
        'mae': float(mean_absolute_error(y_true_flat, y_pred_flat)),
        'r2': float(r2_score(y_true_flat, y_pred_flat)),
        'pearson_r': float(stats.pearsonr(y_true_flat, y_pred_flat)[0]),
        'pearson_p': float(stats.pearsonr(y_true_flat, y_pred_flat)[1]),
        'spearman_r': float(stats.spearmanr(y_true_flat, y_pred_flat)[0]),
        'spearman_p': float(stats.spearmanr(y_true_flat, y_pred_flat)[1]),
    }

    return metrics


def compute_genewise_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              gene_names: Optional[list] = None) -> Dict:
    """
    Compute metrics for each gene separately (correlation across spots)
    """
    n_spots, n_genes = y_true.shape

    if gene_names is None:
        gene_names = [f'Gene_{i}' for i in range(n_genes)]

    gene_metrics = []

    for i in range(n_genes):
        y_true_gene = y_true[:, i]
        y_pred_gene = y_pred[:, i]

        # Skip if all values are the same (no variance)
        if np.var(y_true_gene) < 1e-10 or np.var(y_pred_gene) < 1e-10:
            continue

        metrics = {
            'gene': gene_names[i] if i < len(gene_names) else f'Gene_{i}',
            'mse': float(mean_squared_error(y_true_gene, y_pred_gene)),
            'mae': float(mean_absolute_error(y_true_gene, y_pred_gene)),
            'pearson_r': float(stats.pearsonr(y_true_gene, y_pred_gene)[0]),
            'spearman_r': float(stats.spearmanr(y_true_gene, y_pred_gene)[0]),
            'mean_true': float(np.mean(y_true_gene)),
            'mean_pred': float(np.mean(y_pred_gene)),
            'std_true': float(np.std(y_true_gene)),
            'std_pred': float(np.std(y_pred_gene)),
        }
        gene_metrics.append(metrics)

    # Aggregate statistics
    if len(gene_metrics) > 0:
        df = pd.DataFrame(gene_metrics)
        summary = {
            'mean_pearson_r': float(df['pearson_r'].mean()),
            'median_pearson_r': float(df['pearson_r'].median()),
            'mean_spearman_r': float(df['spearman_r'].mean()),
            'median_spearman_r': float(df['spearman_r'].median()),
            'mean_mse': float(df['mse'].mean()),
            'mean_mae': float(df['mae'].mean()),
            'per_gene_metrics': gene_metrics[:100]  # Save top 100 genes
        }
    else:
        summary = {}

    return summary


def compute_spotwise_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              spot_ids: Optional[list] = None) -> Dict:
    """
    Compute metrics for each spot separately (correlation across genes)
    """
    n_spots, n_genes = y_true.shape

    if spot_ids is None:
        spot_ids = [f'Spot_{i}' for i in range(n_spots)]

    spot_correlations = []

    for i in range(n_spots):
        y_true_spot = y_true[i, :]
        y_pred_spot = y_pred[i, :]

        # Skip if all values are the same
        if np.var(y_true_spot) < 1e-10 or np.var(y_pred_spot) < 1e-10:
            continue

        pearson_r = float(stats.pearsonr(y_true_spot, y_pred_spot)[0])
        spearman_r = float(stats.spearmanr(y_true_spot, y_pred_spot)[0])

        spot_correlations.append({
            'spot': spot_ids[i] if i < len(spot_ids) else f'Spot_{i}',
            'pearson_r': pearson_r,
            'spearman_r': spearman_r,
        })

    # Aggregate statistics
    if len(spot_correlations) > 0:
        df = pd.DataFrame(spot_correlations)
        summary = {
            'mean_pearson_r': float(df['pearson_r'].mean()),
            'median_pearson_r': float(df['pearson_r'].median()),
            'mean_spearman_r': float(df['spearman_r'].mean()),
            'median_spearman_r': float(df['spearman_r'].median()),
            'min_pearson_r': float(df['pearson_r'].min()),
            'max_pearson_r': float(df['pearson_r'].max()),
        }
    else:
        summary = {}

    return summary


def compute_zero_inflation_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                    threshold: float = 0.5) -> Dict:
    """
    Compute metrics related to zero-inflation (dropout events)
    """
    # Binary classification: zero vs non-zero
    y_true_binary = (y_true > threshold).astype(int).flatten()
    y_pred_binary = (y_pred > threshold).astype(int).flatten()

    # Overall zero rate
    zero_rate_true = float(np.mean(y_true == 0))
    zero_rate_pred = float(np.mean(y_pred < threshold))

    # Classification metrics for zero vs non-zero
    accuracy = float(accuracy_score(y_true_binary, y_pred_binary))

    # Sensitivity/Specificity for detecting non-zeros
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))

    sensitivity = float(tp / (tp + fn + 1e-10))  # True positive rate
    specificity = float(tn / (tn + fp + 1e-10))  # True negative rate

    # F1 score for non-zero class
    f1 = float(f1_score(y_true_binary, y_pred_binary, average='binary'))

    # Mean Absolute Error for non-zero values only
    mask_nonzero = y_true.flatten() > 0
    if np.sum(mask_nonzero) > 0:
        mae_nonzero = float(mean_absolute_error(
            y_true.flatten()[mask_nonzero],
            y_pred.flatten()[mask_nonzero]
        ))
    else:
        mae_nonzero = 0.0

    metrics = {
        'zero_rate_true': zero_rate_true,
        'zero_rate_pred': zero_rate_pred,
        'zero_rate_diff': abs(zero_rate_true - zero_rate_pred),
        'zero_detection_accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'mae_nonzero_only': mae_nonzero,
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
        }
    }

    return metrics


def compute_cell_type_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute cell type classification metrics

    Args:
        y_true: True labels (n_spots,)
        y_pred: Predicted logits (n_spots, n_classes) or labels (n_spots,)
    """
    # Convert logits to labels if needed
    if len(y_pred.shape) == 2:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = y_pred

    # Basic metrics
    accuracy = float(accuracy_score(y_true, y_pred_labels))
    f1_macro = float(f1_score(y_true, y_pred_labels, average='macro'))
    f1_weighted = float(f1_score(y_true, y_pred_labels, average='weighted'))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_labels)

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm.tolist(),
    }

    return metrics


def analyze_expression_levels(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Analyze prediction performance at different expression levels
    """
    # Define expression level bins
    bins = [0, 1, 5, 10, 50, np.inf]
    bin_labels = ['0-1', '1-5', '5-10', '10-50', '50+']

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    level_metrics = []

    for i in range(len(bins) - 1):
        mask = (y_true_flat >= bins[i]) & (y_true_flat < bins[i+1])

        if np.sum(mask) > 10:  # Need enough samples
            y_true_bin = y_true_flat[mask]
            y_pred_bin = y_pred_flat[mask]

            metrics = {
                'expression_level': bin_labels[i],
                'n_values': int(np.sum(mask)),
                'mse': float(mean_squared_error(y_true_bin, y_pred_bin)),
                'mae': float(mean_absolute_error(y_true_bin, y_pred_bin)),
                'pearson_r': float(stats.pearsonr(y_true_bin, y_pred_bin)[0]) if np.var(y_true_bin) > 1e-10 else 0.0,
                'mean_true': float(np.mean(y_true_bin)),
                'mean_pred': float(np.mean(y_pred_bin)),
            }
            level_metrics.append(metrics)

    return {'by_expression_level': level_metrics}
