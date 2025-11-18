"""
Tests for hist2scRNA model architectures
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import hist2scRNA, hist2scRNA_Lightweight, ZINBLoss


def test_full_model_creation():
    """Test creation of full hist2scRNA model"""
    model = hist2scRNA(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        vit_depth=4,
        vit_heads=6,
        n_genes=500,
        n_cell_types=5,
        use_spatial_graph=True,
        dropout=0.1
    )

    assert model is not None
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    print(f"Full model parameters: {n_params:,}")


def test_full_model_forward():
    """Test forward pass of full model"""
    model = hist2scRNA(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        vit_depth=2,
        vit_heads=6,
        n_genes=500,
        n_cell_types=5,
        use_spatial_graph=True,
        dropout=0.1
    )

    # Create dummy input
    dummy_img = torch.randn(2, 3, 224, 224)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    # Forward pass
    output = model(dummy_img, edge_index)

    # Check outputs
    assert 'mu' in output
    assert 'theta' in output
    assert 'pi' in output
    assert 'cell_type_logits' in output

    assert output['mu'].shape == (2, 500)
    assert output['theta'].shape == (2, 500)
    assert output['pi'].shape == (2, 500)
    assert output['cell_type_logits'].shape == (2, 5)

    # Check value ranges
    assert (output['mu'] > 0).all()
    assert (output['theta'] > 0).all()
    assert (output['pi'] >= 0).all() and (output['pi'] <= 1).all()

    print("Full model forward pass: PASSED")


def test_lightweight_model():
    """Test lightweight model"""
    model = hist2scRNA_Lightweight(
        feature_dim=2048,
        n_genes=500,
        n_cell_types=5,
        dropout=0.1
    )

    # Forward pass
    dummy_features = torch.randn(2, 2048)
    output = model(dummy_features)

    assert output['mu'].shape == (2, 500)
    assert output['theta'].shape == (2, 500)
    assert output['pi'].shape == (2, 500)

    print("Lightweight model: PASSED")


def test_zinb_loss():
    """Test ZINB loss function"""
    criterion = ZINBLoss()

    # Create dummy data
    mu = torch.exp(torch.randn(2, 500))
    theta = torch.exp(torch.randn(2, 500))
    pi = torch.sigmoid(torch.randn(2, 500))
    target = torch.randint(0, 10, (2, 500)).float()

    # Compute loss
    loss = criterion(mu, theta, pi, target)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0

    print(f"ZINB loss: {loss.item():.4f} - PASSED")


def test_backward_pass():
    """Test backward pass and gradient computation"""
    model = hist2scRNA(
        img_size=224,
        patch_size=16,
        embed_dim=256,
        vit_depth=2,
        vit_heads=4,
        n_genes=100,
        n_cell_types=5,
        use_spatial_graph=False,
        dropout=0.1
    )

    criterion = ZINBLoss()

    # Create dummy data
    dummy_img = torch.randn(2, 3, 224, 224)
    target = torch.randint(0, 10, (2, 100)).float()

    # Forward pass
    output = model(dummy_img)
    loss = criterion(output['mu'], output['theta'], output['pi'], target)

    # Backward pass
    loss.backward()

    # Check gradients
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.parameters() if p.requires_grad)

    assert has_gradients

    print("Backward pass: PASSED")


if __name__ == "__main__":
    print("Running model tests...\n")

    test_full_model_creation()
    test_full_model_forward()
    test_lightweight_model()
    test_zinb_loss()
    test_backward_pass()

    print("\nAll model tests passed!")
