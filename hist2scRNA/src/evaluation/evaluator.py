"""
Evaluator class for hist2scRNA model

Provides comprehensive evaluation with metrics computation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json
from typing import Dict, Optional
import torch

from .metrics import (
    compute_overall_metrics,
    compute_genewise_metrics,
    compute_spotwise_metrics,
    compute_zero_inflation_metrics,
    compute_cell_type_metrics,
    analyze_expression_levels
)


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
        results['overall'] = compute_overall_metrics(y_true, y_pred)

        # 2. Gene-wise metrics
        print("Computing gene-wise metrics...")
        results['gene_wise'] = compute_genewise_metrics(y_true, y_pred, gene_names)

        # 3. Spot-wise metrics
        print("Computing spot-wise metrics...")
        results['spot_wise'] = compute_spotwise_metrics(y_true, y_pred, spot_ids)

        # 4. Zero-inflation metrics
        print("Computing zero-inflation metrics...")
        results['zero_inflation'] = compute_zero_inflation_metrics(y_true, y_pred)

        # 5. Cell type classification metrics (if provided)
        if cell_type_true is not None and cell_type_pred is not None:
            print("Computing cell type classification metrics...")
            results['cell_type'] = compute_cell_type_metrics(cell_type_true, cell_type_pred)

        # 6. Expression level analysis
        print("Computing expression level analysis...")
        results['expression_levels'] = analyze_expression_levels(y_true, y_pred)

        # Save results
        self.save_results(results)

        # Generate visualizations
        print("Generating visualizations...")
        self.generate_plots(y_true, y_pred, results, gene_names)

        # Print summary
        self.print_summary(results)

        return results

    def save_results(self, results: Dict, filename: str = 'evaluation_results.json'):
        """Save evaluation results to JSON file"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")

    def print_summary(self, results: Dict):
        """Print a summary of evaluation results"""
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
            print(f"  Mean Pearson r:   {results['gene_wise'].get('mean_pearson_r', 0):.4f}")
            print(f"  Median Pearson r: {results['gene_wise'].get('median_pearson_r', 0):.4f}")

        # Spot-wise metrics
        if 'spot_wise' in results and results['spot_wise']:
            print("\n[Spot-wise Metrics]")
            print(f"  Mean Pearson r:   {results['spot_wise'].get('mean_pearson_r', 0):.4f}")
            print(f"  Median Pearson r: {results['spot_wise'].get('median_pearson_r', 0):.4f}")

        # Zero-inflation metrics
        if 'zero_inflation' in results:
            print("\n[Zero-Inflation Metrics]")
            print(f"  Zero rate (true): {results['zero_inflation']['zero_rate_true']:.4f}")
            print(f"  Zero rate (pred): {results['zero_inflation']['zero_rate_pred']:.4f}")
            print(f"  F1 score:         {results['zero_inflation']['f1_score']:.4f}")

        # Cell type metrics
        if 'cell_type' in results:
            print("\n[Cell Type Classification]")
            print(f"  Accuracy:       {results['cell_type']['accuracy']:.4f}")
            print(f"  F1 (macro):     {results['cell_type']['f1_macro']:.4f}")

        print("\n" + "="*80)

    def generate_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                       results: Dict, gene_names: Optional[list] = None):
        """Generate visualization plots"""
        # 1. Overall scatter plot
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

    def plot_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, filename: str):
        """Create scatter plot of true vs predicted values"""
        plt.figure(figsize=(10, 10))

        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        if len(y_true_flat) > 10000:
            idx = np.random.choice(len(y_true_flat), 10000, replace=False)
            y_true_flat = y_true_flat[idx]
            y_pred_flat = y_pred_flat[idx]

        plt.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=1)

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
        """Plot distribution of gene-wise correlations"""
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
        """Plot zero-inflation analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # Histogram of expression values
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
        """Plot performance at different expression levels"""
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
    from ..data import build_batch_edge_index

    model.eval()

    all_predictions = []
    all_true_expressions = []
    all_true_cell_types = []
    all_pred_cell_types = []
    edge_index_full = dataloader.dataset.dataset.edge_index if hasattr(dataloader.dataset, 'dataset') else dataloader.dataset.edge_index

    print("Running model inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(device)
            expressions = batch['expressions'].to(device)
            cell_types = batch['cell_types'].to(device)
            indices = batch['indices']

            # Build batch edge index
            batch_edge_index = build_batch_edge_index(edge_index_full, indices).to(device)

            # Forward pass
            output = model(images, batch_edge_index)

            # Collect predictions
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
