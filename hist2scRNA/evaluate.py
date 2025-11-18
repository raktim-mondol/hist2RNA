"""
Main evaluation script for hist2scRNA model

This script provides a command-line interface for evaluating trained hist2scRNA models
with comprehensive metrics and visualizations.

Example usage:
    python evaluate.py --checkpoint ./models/checkpoints/best_model.pt --data_dir ./dummy_data/small
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader, Subset

from models import hist2scRNA, hist2scRNA_Lightweight
from src.data import SpatialTranscriptomicsDataset, collate_spatial_batch, get_default_transforms
from src.evaluation import evaluate_model
from src.utils import get_device, set_seed


def main(args):
    """Main evaluation function"""
    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)

    print("\n" + "="*80)
    print("hist2scRNA Evaluation")
    print("="*80)

    # Load dataset
    print(f"\nLoading data from {args.data_dir}...")
    transform = get_default_transforms(augment=False)
    dataset = SpatialTranscriptomicsDataset(args.data_dir, transform=transform, use_images=True)

    # Create data loader for test set (or full dataset if no split specified)
    if args.test_indices:
        # Load specific test indices
        test_indices = torch.load(args.test_indices)
        test_dataset = Subset(dataset, test_indices)
    else:
        # Use full dataset
        test_dataset = dataset

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_spatial_batch,
        num_workers=args.num_workers
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=device)

    # Create model (need to infer architecture from checkpoint or args)
    if args.model_type == 'full':
        model = hist2scRNA(
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            vit_depth=args.vit_depth,
            vit_heads=args.vit_heads,
            n_genes=dataset.n_genes,
            n_cell_types=dataset.metadata['n_cell_types'],
            use_spatial_graph=args.use_spatial_graph,
            dropout=args.dropout
        )
    else:
        model = hist2scRNA_Lightweight(
            feature_dim=2048,
            n_genes=dataset.n_genes,
            n_cell_types=dataset.metadata['n_cell_types'],
            dropout=args.dropout
        )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'val_metrics' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_metrics']['loss']:.4f}")

    # Evaluate
    print(f"\nRunning evaluation...")
    os.makedirs(args.output_dir, exist_ok=True)

    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=args.output_dir
    )

    print("\n" + "="*80)
    print("Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)

    # Save detailed results
    import json
    results_path = os.path.join(args.output_dir, 'detailed_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate hist2scRNA model')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the data')
    parser.add_argument('--test_indices', type=str, default=None,
                        help='Path to test indices file (optional)')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='full', choices=['full', 'lightweight'],
                        help='Model type')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size for Vision Transformer')
    parser.add_argument('--embed_dim', type=int, default=384,
                        help='Embedding dimension')
    parser.add_argument('--vit_depth', type=int, default=6,
                        help='Number of transformer blocks')
    parser.add_argument('--vit_heads', type=int, default=6,
                        help='Number of attention heads')
    parser.add_argument('--use_spatial_graph', action='store_true', default=True,
                        help='Use spatial graph attention')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Evaluation
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    # Output
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Run evaluation
    main(args)
