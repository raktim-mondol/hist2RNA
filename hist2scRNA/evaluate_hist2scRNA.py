"""
Evaluation script for hist2scRNA model

Comprehensive evaluation metrics for single-cell RNA-seq prediction from histopathology images.
Includes:
- Regression metrics (MSE, MAE, R², Pearson/Spearman correlation)
- Single-cell specific metrics (gene-wise, spot-wise correlations)
- Zero-inflation metrics (prediction of zeros vs non-zeros)
- Cell type classification metrics (accuracy, F1-score)
- Visualization functions
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
from typing import Dict, Tuple, Optional
import json
import os


class scRNAEvaluator:
    """
    Comprehensive evaluator for single-cell RNA-seq prediction models
    """

    def __init__(self, save_dir: str = './evaluation_results'):
        """
        Args:
            save_dir: Directory to save evaluation results and plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)

    def evaluate_all(self,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     cell_type_true: Optional[np.ndarray] = None,
                     cell_type_pred: Optional[np.ndarray] = None,
                     gene_names: Optional[list] = None,
                     spot_ids: Optional[list] = None) -> Dict:
        """
        Perform comprehensive evaluation

        Args:
            y_true: True gene expression (n_spots, n_genes)
            y_pred: Predicted gene expression (n_spots, n_genes)
            cell_type_true: True cell type labels (n_spots,)
            cell_type_pred: Predicted cell type logits or labels (n_spots, n_cell_types) or (n_spots,)
            gene_names: List of gene names
            spot_ids: List of spot identifiers

        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {}

        # 1. Overall regression metrics
        print("Computing overall regression metrics...")
        results['overall'] = self.compute_overall_metrics(y_true, y_pred)

        # 2. Gene-wise metrics
        print("Computing gene-wise metrics...")
        results['gene_wise'] = self.compute_genewise_metrics(y_true, y_pred, gene_names)

        # 3. Spot-wise metrics
        print("Computing spot-wise metrics...")
        results['spot_wise'] = self.compute_spotwise_metrics(y_true, y_pred, spot_ids)

        # 4. Zero-inflation metrics
        print("Computing zero-inflation metrics...")
        results['zero_inflation'] = self.compute_zero_inflation_metrics(y_true, y_pred)

        # 5. Cell type classification metrics (if provided)
        if cell_type_true is not None and cell_type_pred is not None:
            print("Computing cell type classification metrics...")
            results['cell_type'] = self.compute_cell_type_metrics(cell_type_true, cell_type_pred)

        # 6. Expression level analysis
        print("Computing expression level analysis...")
        results['expression_levels'] = self.analyze_expression_levels(y_true, y_pred)

        # Save results
        self.save_results(results)

        # Generate visualizations
        print("Generating visualizations...")
        self.generate_plots(y_true, y_pred, results, gene_names)

        # Print summary
        self.print_summary(results)

        return results

    def compute_overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
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

    def compute_genewise_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
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
                'per_gene_metrics': gene_metrics[:100]  # Save top 100 genes to avoid huge files
            }
        else:
            summary = {}

        return summary

    def compute_spotwise_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
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

    def compute_zero_inflation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
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
            'sensitivity': sensitivity,  # Ability to detect non-zeros
            'specificity': specificity,  # Ability to detect zeros
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

    def compute_cell_type_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
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

        # Per-class metrics
        report = classification_report(y_true, y_pred_labels, output_dict=True)

        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
        }

        return metrics

    def analyze_expression_levels(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
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

    def save_results(self, results: Dict, filename: str = 'evaluation_results.json'):
        """
        Save evaluation results to JSON file
        """
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")

    def print_summary(self, results: Dict):
        """
        Print a summary of evaluation results
        """
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)

        # Overall metrics
        if 'overall' in results:
            print("\n[Overall Metrics]")
            print(f"  MSE:           {results['overall']['mse']:.4f}")
            print(f"  RMSE:          {results['overall']['rmse']:.4f}")
            print(f"  MAE:           {results['overall']['mae']:.4f}")
            print(f"  R²:            {results['overall']['r2']:.4f}")
            print(f"  Pearson r:     {results['overall']['pearson_r']:.4f}")
            print(f"  Spearman ρ:    {results['overall']['spearman_r']:.4f}")

        # Gene-wise metrics
        if 'gene_wise' in results and results['gene_wise']:
            print("\n[Gene-wise Metrics]")
            print(f"  Mean Pearson r:   {results['gene_wise'].get('mean_pearson_r', 'N/A'):.4f}")
            print(f"  Median Pearson r: {results['gene_wise'].get('median_pearson_r', 'N/A'):.4f}")
            print(f"  Mean Spearman ρ:  {results['gene_wise'].get('mean_spearman_r', 'N/A'):.4f}")

        # Spot-wise metrics
        if 'spot_wise' in results and results['spot_wise']:
            print("\n[Spot-wise Metrics]")
            print(f"  Mean Pearson r:   {results['spot_wise'].get('mean_pearson_r', 'N/A'):.4f}")
            print(f"  Median Pearson r: {results['spot_wise'].get('median_pearson_r', 'N/A'):.4f}")

        # Zero-inflation metrics
        if 'zero_inflation' in results:
            print("\n[Zero-Inflation Metrics]")
            print(f"  Zero rate (true): {results['zero_inflation']['zero_rate_true']:.4f}")
            print(f"  Zero rate (pred): {results['zero_inflation']['zero_rate_pred']:.4f}")
            print(f"  Zero detection accuracy: {results['zero_inflation']['zero_detection_accuracy']:.4f}")
            print(f"  Sensitivity:      {results['zero_inflation']['sensitivity']:.4f}")
            print(f"  Specificity:      {results['zero_inflation']['specificity']:.4f}")
            print(f"  F1 score:         {results['zero_inflation']['f1_score']:.4f}")

        # Cell type metrics
        if 'cell_type' in results:
            print("\n[Cell Type Classification]")
            print(f"  Accuracy:       {results['cell_type']['accuracy']:.4f}")
            print(f"  F1 (macro):     {results['cell_type']['f1_macro']:.4f}")
            print(f"  F1 (weighted):  {results['cell_type']['f1_weighted']:.4f}")

        print("\n" + "="*80)

    def generate_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                       results: Dict, gene_names: Optional[list] = None):
        """
        Generate visualization plots
        """
        # 1. Overall scatter plot (true vs predicted)
        self.plot_scatter(y_true, y_pred, 'overall_scatter.png')

        # 2. Gene-wise correlation distribution
        if 'gene_wise' in results and 'per_gene_metrics' in results['gene_wise']:
            self.plot_genewise_correlations(results['gene_wise'], 'genewise_correlations.png')

        # 3. Zero-inflation analysis
        if 'zero_inflation' in results:
            self.plot_zero_inflation(y_true, y_pred, 'zero_inflation_analysis.png')

        # 4. Expression level analysis
        if 'expression_levels' in results:
            self.plot_expression_levels(results['expression_levels'], 'expression_levels.png')

        # 5. Cell type confusion matrix
        if 'cell_type' in results:
            self.plot_confusion_matrix(results['cell_type'], 'cell_type_confusion.png')

    def plot_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, filename: str):
        """
        Create scatter plot of true vs predicted values
        """
        plt.figure(figsize=(10, 10))

        # Subsample for visualization if too many points
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        if len(y_true_flat) > 10000:
            idx = np.random.choice(len(y_true_flat), 10000, replace=False)
            y_true_flat = y_true_flat[idx]
            y_pred_flat = y_pred_flat[idx]

        plt.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=1)

        # Add diagonal line
        max_val = max(np.max(y_true_flat), np.max(y_pred_flat))
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')

        plt.xlabel('True Expression', fontsize=14)
        plt.ylabel('Predicted Expression', fontsize=14)
        plt.title('True vs Predicted Gene Expression', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)

        filepath = os.path.join(self.save_dir, 'plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scatter plot saved to {filepath}")

    def plot_genewise_correlations(self, gene_metrics: Dict, filename: str):
        """
        Plot distribution of gene-wise correlations
        """
        if 'per_gene_metrics' not in gene_metrics:
            return

        df = pd.DataFrame(gene_metrics['per_gene_metrics'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pearson correlation
        axes[0].hist(df['pearson_r'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(df['pearson_r'].mean(), color='r', linestyle='--',
                        linewidth=2, label=f'Mean: {df["pearson_r"].mean():.3f}')
        axes[0].set_xlabel('Pearson Correlation', fontsize=12)
        axes[0].set_ylabel('Number of Genes', fontsize=12)
        axes[0].set_title('Distribution of Gene-wise Pearson Correlations', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Spearman correlation
        axes[1].hist(df['spearman_r'], bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[1].axvline(df['spearman_r'].mean(), color='r', linestyle='--',
                        linewidth=2, label=f'Mean: {df["spearman_r"].mean():.3f}')
        axes[1].set_xlabel('Spearman Correlation', fontsize=12)
        axes[1].set_ylabel('Number of Genes', fontsize=12)
        axes[1].set_title('Distribution of Gene-wise Spearman Correlations', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, 'plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gene-wise correlation plot saved to {filepath}")

    def plot_zero_inflation(self, y_true: np.ndarray, y_pred: np.ndarray, filename: str):
        """
        Plot zero-inflation analysis
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of expression values
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        axes[0].hist(y_true_flat, bins=50, alpha=0.5, label='True', edgecolor='black')
        axes[0].hist(y_pred_flat, bins=50, alpha=0.5, label='Predicted', edgecolor='black')
        axes[0].set_xlabel('Expression Value', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Expression Values', fontsize=14)
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Zero vs non-zero comparison
        zero_true = np.sum(y_true_flat == 0)
        nonzero_true = np.sum(y_true_flat > 0)
        zero_pred = np.sum(y_pred_flat < 0.5)
        nonzero_pred = np.sum(y_pred_flat >= 0.5)

        x = np.arange(2)
        width = 0.35
        axes[1].bar(x - width/2, [zero_true, nonzero_true], width, label='True', alpha=0.8)
        axes[1].bar(x + width/2, [zero_pred, nonzero_pred], width, label='Predicted', alpha=0.8)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Zero vs Non-zero Values', fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['Zero', 'Non-zero'])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, 'plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Zero-inflation plot saved to {filepath}")

    def plot_expression_levels(self, level_metrics: Dict, filename: str):
        """
        Plot performance at different expression levels
        """
        if 'by_expression_level' not in level_metrics:
            return

        df = pd.DataFrame(level_metrics['by_expression_level'])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # MAE by expression level
        axes[0, 0].bar(df['expression_level'], df['mae'])
        axes[0, 0].set_ylabel('MAE', fontsize=12)
        axes[0, 0].set_title('MAE by Expression Level', fontsize=14)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Pearson correlation by expression level
        axes[0, 1].bar(df['expression_level'], df['pearson_r'], color='green')
        axes[0, 1].set_ylabel('Pearson r', fontsize=12)
        axes[0, 1].set_title('Correlation by Expression Level', fontsize=14)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Number of values per level
        axes[1, 0].bar(df['expression_level'], df['n_values'], color='orange')
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('Sample Count by Expression Level', fontsize=14)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Mean true vs mean predicted
        x = np.arange(len(df))
        width = 0.35
        axes[1, 1].bar(x - width/2, df['mean_true'], width, label='True', alpha=0.8)
        axes[1, 1].bar(x + width/2, df['mean_pred'], width, label='Predicted', alpha=0.8)
        axes[1, 1].set_ylabel('Mean Expression', fontsize=12)
        axes[1, 1].set_title('Mean Expression by Level', fontsize=14)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(df['expression_level'], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, 'plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Expression levels plot saved to {filepath}")

    def plot_confusion_matrix(self, cell_type_metrics: Dict, filename: str):
        """
        Plot confusion matrix for cell type classification
        """
        cm = np.array(cell_type_metrics['confusion_matrix'])

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted Cell Type', fontsize=12)
        plt.ylabel('True Cell Type', fontsize=12)
        plt.title('Cell Type Classification Confusion Matrix', fontsize=14)

        filepath = os.path.join(self.save_dir, 'plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved to {filepath}")


def evaluate_model(model, dataloader, device, save_dir='./evaluation_results'):
    """
    Evaluate a trained model on a dataset

    Args:
        model: Trained hist2scRNA model
        dataloader: DataLoader with test data
        device: torch device
        save_dir: Directory to save results

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_true_expressions = []
    all_true_cell_types = []
    all_pred_cell_types = []

    print("Running model inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            expressions = batch['expression'].to(device)
            cell_types = batch['cell_type'].to(device)
            edge_index = batch.get('edge_index', None)

            if edge_index is not None:
                edge_index = edge_index.to(device)

            # Forward pass
            output = model(images, edge_index)

            # Collect predictions (use mu as prediction)
            all_predictions.append(output['mu'].cpu().numpy())
            all_true_expressions.append(expressions.cpu().numpy())
            all_true_cell_types.append(cell_types.cpu().numpy())
            all_pred_cell_types.append(output['cell_type_logits'].cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Concatenate all batches
    y_pred = np.concatenate(all_predictions, axis=0)
    y_true = np.concatenate(all_true_expressions, axis=0)
    cell_type_true = np.concatenate(all_true_cell_types, axis=0)
    cell_type_pred = np.concatenate(all_pred_cell_types, axis=0)

    print(f"Evaluation data shape: {y_true.shape}")
    print(f"Predictions shape: {y_pred.shape}")

    # Create evaluator and run evaluation
    evaluator = scRNAEvaluator(save_dir=save_dir)
    results = evaluator.evaluate_all(
        y_true=y_true,
        y_pred=y_pred,
        cell_type_true=cell_type_true,
        cell_type_pred=cell_type_pred
    )

    return results


if __name__ == '__main__':
    """
    Example usage with dummy data
    """
    print("Running evaluation example with dummy data...")

    # Generate dummy data
    n_spots = 100
    n_genes = 500
    n_cell_types = 5

    # Simulate ZINB-like data (high sparsity)
    np.random.seed(42)
    y_true = np.random.negative_binomial(5, 0.3, size=(n_spots, n_genes)).astype(float)
    # Add zeros (dropout)
    dropout_mask = np.random.rand(n_spots, n_genes) < 0.7
    y_true[dropout_mask] = 0

    # Simulate predictions (with some noise)
    y_pred = y_true + np.random.normal(0, 2, size=(n_spots, n_genes))
    y_pred = np.clip(y_pred, 0, None)  # Expression can't be negative

    # Simulate cell types
    cell_type_true = np.random.randint(0, n_cell_types, size=n_spots)
    cell_type_pred_logits = np.random.randn(n_spots, n_cell_types)
    # Make predictions somewhat correlated with true labels
    for i in range(n_spots):
        cell_type_pred_logits[i, cell_type_true[i]] += 2

    # Create evaluator
    evaluator = scRNAEvaluator(save_dir='./evaluation_example')

    # Run evaluation
    results = evaluator.evaluate_all(
        y_true=y_true,
        y_pred=y_pred,
        cell_type_true=cell_type_true,
        cell_type_pred=cell_type_pred_logits,
        gene_names=[f'GENE{i}' for i in range(n_genes)],
        spot_ids=[f'spot_{i:04d}' for i in range(n_spots)]
    )

    print("\nEvaluation complete! Check './evaluation_example' directory for results and plots.")
