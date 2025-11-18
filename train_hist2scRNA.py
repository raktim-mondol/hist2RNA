"""
Training script for hist2scRNA model

This script trains the state-of-the-art single-cell RNA-seq prediction model
on spatial transcriptomics data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import json
from tqdm import tqdm
import argparse

from hist2scRNA_model import hist2scRNA, hist2scRNA_Lightweight, ZINBLoss


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


def collate_spatial_batch(batch):
    """
    Custom collate function for spatial data
    Builds a batch with graph structure
    """
    images = torch.stack([item['image'] for item in batch])
    expressions = torch.stack([item['expression'] for item in batch])
    cell_types = torch.stack([item['cell_type'] for item in batch])
    coordinates = torch.stack([item['coordinate'] for item in batch])
    indices = torch.tensor([item['idx'] for item in batch])

    return {
        'images': images,
        'expressions': expressions,
        'cell_types': cell_types,
        'coordinates': coordinates,
        'indices': indices
    }


def build_batch_edge_index(edge_index_full, batch_indices):
    """
    Build edge index for a batch of spots
    """
    # Create mapping from global indices to batch indices
    index_map = {idx.item(): i for i, idx in enumerate(batch_indices)}

    # Filter edges that are within the batch
    batch_edges = []
    for i in range(edge_index_full.shape[1]):
        src, dst = edge_index_full[0, i].item(), edge_index_full[1, i].item()
        if src in index_map and dst in index_map:
            batch_edges.append([index_map[src], index_map[dst]])

    if len(batch_edges) > 0:
        batch_edge_index = torch.tensor(batch_edges, dtype=torch.long).t()
    else:
        # Return empty edge index if no edges in batch
        batch_edge_index = torch.zeros((2, 0), dtype=torch.long)

    return batch_edge_index


def train_epoch(model, dataloader, edge_index_full, optimizer, criterion_zinb, criterion_ce, device, alpha=0.1):
    """
    Train for one epoch

    Args:
        model: the model
        dataloader: training data loader
        edge_index_full: full edge index for the graph
        optimizer: optimizer
        criterion_zinb: ZINB loss for gene expression
        criterion_ce: cross-entropy loss for cell type classification
        device: device to use
        alpha: weight for cell type loss

    Returns:
        average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_zinb_loss = 0
    total_ce_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch['images'].to(device)
        expressions = batch['expressions'].to(device)
        cell_types = batch['cell_types'].to(device)
        indices = batch['indices']

        # Build batch edge index
        batch_edge_index = build_batch_edge_index(edge_index_full, indices).to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(images, batch_edge_index)

        # ZINB loss for gene expression
        zinb_loss = criterion_zinb(output['mu'], output['theta'], output['pi'], expressions)

        # Cell type classification loss
        ce_loss = criterion_ce(output['cell_type_logits'], cell_types)

        # Combined loss
        loss = zinb_loss + alpha * ce_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_zinb_loss += zinb_loss.item()
        total_ce_loss += ce_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_zinb_loss = total_zinb_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)

    return avg_loss, avg_zinb_loss, avg_ce_loss


def evaluate(model, dataloader, edge_index_full, criterion_zinb, criterion_ce, device, alpha=0.1):
    """
    Evaluate the model

    Returns:
        metrics dictionary
    """
    model.eval()
    total_loss = 0
    total_zinb_loss = 0
    total_ce_loss = 0

    all_predictions = []
    all_targets = []
    all_cell_type_preds = []
    all_cell_type_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device)
            expressions = batch['expressions'].to(device)
            cell_types = batch['cell_types'].to(device)
            indices = batch['indices']

            # Build batch edge index
            batch_edge_index = build_batch_edge_index(edge_index_full, indices).to(device)

            # Forward pass
            output = model(images, batch_edge_index)

            # ZINB loss
            zinb_loss = criterion_zinb(output['mu'], output['theta'], output['pi'], expressions)

            # Cell type loss
            ce_loss = criterion_ce(output['cell_type_logits'], cell_types)

            # Combined loss
            loss = zinb_loss + alpha * ce_loss

            total_loss += loss.item()
            total_zinb_loss += zinb_loss.item()
            total_ce_loss += ce_loss.item()

            # Collect predictions
            all_predictions.append(output['mu'].cpu().numpy())
            all_targets.append(expressions.cpu().numpy())
            all_cell_type_preds.append(output['cell_type_logits'].argmax(dim=1).cpu().numpy())
            all_cell_type_targets.append(cell_types.cpu().numpy())

    # Concatenate all predictions
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    all_cell_type_preds = np.concatenate(all_cell_type_preds)
    all_cell_type_targets = np.concatenate(all_cell_type_targets)

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)

    # Per-spot correlation
    spot_correlations = []
    for i in range(all_predictions.shape[0]):
        corr, _ = spearmanr(all_predictions[i], all_targets[i])
        if not np.isnan(corr):
            spot_correlations.append(corr)

    # Per-gene correlation
    gene_correlations = []
    for j in range(all_predictions.shape[1]):
        corr, _ = spearmanr(all_predictions[:, j], all_targets[:, j])
        if not np.isnan(corr):
            gene_correlations.append(corr)

    # Cell type accuracy
    cell_type_accuracy = (all_cell_type_preds == all_cell_type_targets).mean()

    metrics = {
        'loss': total_loss / len(dataloader),
        'zinb_loss': total_zinb_loss / len(dataloader),
        'ce_loss': total_ce_loss / len(dataloader),
        'mse': mse,
        'mae': mae,
        'mean_spot_corr': np.mean(spot_correlations),
        'mean_gene_corr': np.mean(gene_correlations),
        'cell_type_accuracy': cell_type_accuracy
    }

    return metrics, all_predictions, all_targets


def plot_training_curves(train_losses, val_losses, save_path):
    """
    Plot training and validation loss curves
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses['total'], label='Train')
    plt.plot(val_losses['total'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_losses['zinb'], label='Train')
    plt.plot(val_losses['zinb'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('ZINB Loss')
    plt.title('ZINB Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(train_losses['ce'], label='Train')
    plt.plot(val_losses['ce'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Cell Type Loss')
    plt.title('Cell Type Classification Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training curves to {save_path}")


def main(args):
    """
    Main training function
    """
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading data from {args.data_dir}...")
    dataset = SpatialTranscriptomicsDataset(args.data_dir, use_images=True)

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_spatial_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_spatial_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_spatial_batch)

    # Create model
    print(f"\nCreating model...")
    if args.model_type == 'full':
        model = hist2scRNA(
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            vit_depth=args.vit_depth,
            vit_heads=args.vit_heads,
            n_genes=dataset.n_genes,
            n_cell_types=dataset.metadata['n_cell_types'],
            use_spatial_graph=True,
            dropout=args.dropout
        )
    else:
        model = hist2scRNA_Lightweight(
            feature_dim=2048,
            n_genes=dataset.n_genes,
            n_cell_types=dataset.metadata['n_cell_types'],
            dropout=args.dropout
        )

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss functions
    criterion_zinb = ZINBLoss()
    criterion_ce = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    train_losses = {'total': [], 'zinb': [], 'ce': []}
    val_losses = {'total': [], 'zinb': [], 'ce': []}

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, train_zinb_loss, train_ce_loss = train_epoch(
            model, train_loader, dataset.edge_index, optimizer,
            criterion_zinb, criterion_ce, device, alpha=args.alpha
        )

        train_losses['total'].append(train_loss)
        train_losses['zinb'].append(train_zinb_loss)
        train_losses['ce'].append(train_ce_loss)

        print(f"Train - Loss: {train_loss:.4f}, ZINB: {train_zinb_loss:.4f}, CE: {train_ce_loss:.4f}")

        # Validate
        val_metrics, _, _ = evaluate(
            model, val_loader, dataset.edge_index,
            criterion_zinb, criterion_ce, device, alpha=args.alpha
        )

        val_losses['total'].append(val_metrics['loss'])
        val_losses['zinb'].append(val_metrics['zinb_loss'])
        val_losses['ce'].append(val_metrics['ce_loss'])

        print(f"Val   - Loss: {val_metrics['loss']:.4f}, ZINB: {val_metrics['zinb_loss']:.4f}, CE: {val_metrics['ce_loss']:.4f}")
        print(f"Val   - Spot Corr: {val_metrics['mean_spot_corr']:.4f}, Gene Corr: {val_metrics['mean_gene_corr']:.4f}")
        print(f"Val   - Cell Type Acc: {val_metrics['cell_type_accuracy']:.4f}")

        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_metrics': val_metrics
            }, args.checkpoint_path)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")

    # Plot training curves
    plot_training_curves(train_losses, val_losses, args.output_dir + '/training_curves.png')

    # Load best model and evaluate on test set
    print(f"\n{'='*60}")
    print("Evaluating best model on test set...")
    print(f"{'='*60}")

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics, test_predictions, test_targets = evaluate(
        model, test_loader, dataset.edge_index,
        criterion_zinb, criterion_ce, device, alpha=args.alpha
    )

    print(f"\nTest Results:")
    print(f"  - Loss: {test_metrics['loss']:.4f}")
    print(f"  - MSE: {test_metrics['mse']:.4f}")
    print(f"  - MAE: {test_metrics['mae']:.4f}")
    print(f"  - Mean Spot Correlation: {test_metrics['mean_spot_corr']:.4f}")
    print(f"  - Mean Gene Correlation: {test_metrics['mean_gene_corr']:.4f}")
    print(f"  - Cell Type Accuracy: {test_metrics['cell_type_accuracy']:.4f}")

    # Save results
    results = {
        'args': vars(args),
        'test_metrics': test_metrics,
        'best_val_metrics': checkpoint['val_metrics']
    }

    with open(args.output_dir + '/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train hist2scRNA model')

    # Data
    parser.add_argument('--data_dir', type=str, default='./dummy_data/small',
                        help='Directory containing the data')

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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--output_dir', type=str, default='./output_scrna',
                        help='Output directory')
    parser.add_argument('--checkpoint_path', type=str, default='./output_scrna/best_model.pt',
                        help='Path to save best model')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run training
    main(args)
