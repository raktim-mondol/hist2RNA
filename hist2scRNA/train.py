"""
Main training script for hist2scRNA model

This script provides a command-line interface for training the hist2scRNA model
with configurable hyperparameters.

Example usage:
    python train.py --config experiments/config/default.yaml
    python train.py --data_dir ./dummy_data/medium --epochs 100
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from models import hist2scRNA, hist2scRNA_Lightweight, ZINBLoss
from src.data import SpatialTranscriptomicsDataset, collate_spatial_batch, get_default_transforms
from src.training import hist2scRNATrainer
from src.utils import (
    Config, set_seed, get_device, create_output_dirs,
    count_parameters, plot_training_curves
)


def main(args):
    """Main training function"""
    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)

    # Create output directories
    create_output_dirs(args.output_dir, args.checkpoint_dir)

    print("\n" + "="*80)
    print("hist2scRNA Training")
    print("="*80)

    # Load dataset
    print(f"\nLoading data from {args.data_dir}...")
    transform = get_default_transforms(augment=args.augment)
    dataset = SpatialTranscriptomicsDataset(args.data_dir, transform=transform, use_images=True)

    # Split dataset
    train_size = int(args.train_split * len(dataset))
    val_size = int(args.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"\nDataset splits:")
    print(f"  - Train: {len(train_dataset)}")
    print(f"  - Validation: {len(val_dataset)}")
    print(f"  - Test: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_spatial_batch,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_spatial_batch,
        num_workers=args.num_workers
    )

    # Create model
    print(f"\nCreating model: {args.model_type}")
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

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Loss functions
    criterion_zinb = ZINBLoss()
    criterion_ce = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.scheduler_patience
    )

    # Create trainer
    trainer = hist2scRNATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        edge_index_full=dataset.edge_index,
        criterion_zinb=criterion_zinb,
        criterion_ce=criterion_ce,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        alpha=args.alpha,
        checkpoint_dir=args.checkpoint_dir
    )

    # Train
    train_losses, val_losses = trainer.train(epochs=args.epochs)

    # Plot training curves
    plot_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, plot_path)

    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Outputs saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train hist2scRNA model')

    # Data
    parser.add_argument('--data_dir', type=str, default='./dummy_data/small',
                        help='Directory containing the data')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Fraction of data for training')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of data for validation')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation')

    # Model
    parser.add_argument('--model_type', type=str, default='full', choices=['full', 'lightweight'],
                        help='Model type to use')
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

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Weight for cell type classification loss')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    # Output
    parser.add_argument('--output_dir', type=str, default='./output_scrna',
                        help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./models/checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    # Config file (overrides command-line arguments)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration YAML file')

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = Config.from_yaml(args.config)
        # Override args with config values
        args.img_size = config.model.img_size
        args.embed_dim = config.model.embed_dim
        args.epochs = config.training.epochs
        args.batch_size = config.training.batch_size
        # ... (you can add more overrides)

    # Run training
    main(args)
