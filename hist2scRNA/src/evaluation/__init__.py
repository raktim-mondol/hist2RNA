"""
Evaluation module for hist2scRNA

Provides comprehensive evaluation metrics and visualization tools.
"""

from .evaluator import scRNAEvaluator, evaluate_model
from .metrics import (
    compute_overall_metrics,
    compute_genewise_metrics,
    compute_spotwise_metrics,
    compute_zero_inflation_metrics,
    compute_cell_type_metrics,
    analyze_expression_levels
)

__all__ = [
    'scRNAEvaluator',
    'evaluate_model',
    'compute_overall_metrics',
    'compute_genewise_metrics',
    'compute_spotwise_metrics',
    'compute_zero_inflation_metrics',
    'compute_cell_type_metrics',
    'analyze_expression_levels',
]
