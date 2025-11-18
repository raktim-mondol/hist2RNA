"""
Trainer class for hist2scRNA model

Handles the complete training pipeline including training loops,
validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os


class hist2scRNATrainer:
    """
    Trainer for hist2scRNA model
    """
    def __init__(self, model, train_loader, val_loader, edge_index_full,
                 criterion_zinb, criterion_ce, optimizer, scheduler=None,
                 device='cuda', alpha=0.1, checkpoint_dir='./checkpoints'):
        """
        Args:
            model: hist2scRNA model
            train_loader: training data loader
            val_loader: validation data loader
            edge_index_full: full edge index for spatial graph
            criterion_zinb: ZINB loss function
            criterion_ce: Cross-entropy loss for cell type classification
            optimizer: optimizer
            scheduler: learning rate scheduler (optional)
            device: device to use for training
            alpha: weight for cell type classification loss
            checkpoint_dir: directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.edge_index_full = edge_index_full
        self.criterion_zinb = criterion_zinb
        self.criterion_ce = criterion_ce
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.alpha = alpha
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.train_losses = {'total': [], 'zinb': [], 'ce': []}
        self.val_losses = {'total': [], 'zinb': [], 'ce': []}
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        """Train for one epoch"""
        from ..data import build_batch_edge_index

        self.model.train()
        total_loss = 0
        total_zinb_loss = 0
        total_ce_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} - Training")
        for batch in pbar:
            images = batch['images'].to(self.device)
            expressions = batch['expressions'].to(self.device)
            cell_types = batch['cell_types'].to(self.device)
            indices = batch['indices']

            # Build batch edge index
            batch_edge_index = build_batch_edge_index(self.edge_index_full, indices).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(images, batch_edge_index)

            # ZINB loss for gene expression
            zinb_loss = self.criterion_zinb(output['mu'], output['theta'], output['pi'], expressions)

            # Cell type classification loss
            ce_loss = self.criterion_ce(output['cell_type_logits'], cell_types)

            # Combined loss
            loss = zinb_loss + self.alpha * ce_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_zinb_loss += zinb_loss.item()
            total_ce_loss += ce_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'zinb': f'{zinb_loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_zinb_loss = total_zinb_loss / len(self.train_loader)
        avg_ce_loss = total_ce_loss / len(self.train_loader)

        return avg_loss, avg_zinb_loss, avg_ce_loss

    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        from ..data import build_batch_edge_index
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from scipy.stats import spearmanr

        self.model.eval()
        total_loss = 0
        total_zinb_loss = 0
        total_ce_loss = 0

        all_predictions = []
        all_targets = []
        all_cell_type_preds = []
        all_cell_type_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} - Validation")
            for batch in pbar:
                images = batch['images'].to(self.device)
                expressions = batch['expressions'].to(self.device)
                cell_types = batch['cell_types'].to(self.device)
                indices = batch['indices']

                # Build batch edge index
                batch_edge_index = build_batch_edge_index(self.edge_index_full, indices).to(self.device)

                # Forward pass
                output = self.model(images, batch_edge_index)

                # ZINB loss
                zinb_loss = self.criterion_zinb(output['mu'], output['theta'], output['pi'], expressions)

                # Cell type loss
                ce_loss = self.criterion_ce(output['cell_type_logits'], cell_types)

                # Combined loss
                loss = zinb_loss + self.alpha * ce_loss

                total_loss += loss.item()
                total_zinb_loss += zinb_loss.item()
                total_ce_loss += ce_loss.item()

                # Collect predictions
                all_predictions.append(output['mu'].cpu().numpy())
                all_targets.append(expressions.cpu().numpy())
                all_cell_type_preds.append(output['cell_type_logits'].argmax(dim=1).cpu().numpy())
                all_cell_type_targets.append(cell_types.cpu().numpy())

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'zinb': f'{zinb_loss.item():.4f}',
                    'ce': f'{ce_loss.item():.4f}'
                })

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

        # Cell type accuracy
        cell_type_accuracy = (all_cell_type_preds == all_cell_type_targets).mean()

        metrics = {
            'loss': total_loss / len(self.val_loader),
            'zinb_loss': total_zinb_loss / len(self.val_loader),
            'ce_loss': total_ce_loss / len(self.val_loader),
            'mse': mse,
            'mae': mae,
            'mean_spot_corr': np.mean(spot_correlations) if spot_correlations else 0.0,
            'cell_type_accuracy': cell_type_accuracy
        }

        return metrics

    def train(self, epochs):
        """
        Main training loop

        Args:
            epochs: number of epochs to train

        Returns:
            train_losses: dictionary of training losses
            val_losses: dictionary of validation losses
        """
        print(f"\nStarting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*60}")

            # Train
            train_loss, train_zinb_loss, train_ce_loss = self.train_epoch(epoch)
            self.train_losses['total'].append(train_loss)
            self.train_losses['zinb'].append(train_zinb_loss)
            self.train_losses['ce'].append(train_ce_loss)

            print(f"Train - Loss: {train_loss:.4f}, ZINB: {train_zinb_loss:.4f}, CE: {train_ce_loss:.4f}")

            # Validate
            val_metrics = self.validate_epoch(epoch)
            self.val_losses['total'].append(val_metrics['loss'])
            self.val_losses['zinb'].append(val_metrics['zinb_loss'])
            self.val_losses['ce'].append(val_metrics['ce_loss'])

            print(f"Val   - Loss: {val_metrics['loss']:.4f}, ZINB: {val_metrics['zinb_loss']:.4f}, CE: {val_metrics['ce_loss']:.4f}")
            print(f"Val   - Spot Corr: {val_metrics['mean_spot_corr']:.4f}, Cell Type Acc: {val_metrics['cell_type_accuracy']:.4f}")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['loss'])

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  -> Saved best model (val_loss: {self.best_val_loss:.4f})")

        return self.train_losses, self.val_losses

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint
