"""
Quick test script for hist2scRNA model
Tests model architecture and forward pass without requiring training data
"""

import torch
import sys

print("Testing hist2scRNA model architecture...")
print("=" * 60)

try:
    from hist2scRNA_model import hist2scRNA, hist2scRNA_Lightweight, ZINBLoss
    print("✓ Successfully imported model modules")
except Exception as e:
    print(f"✗ Failed to import modules: {e}")
    sys.exit(1)

# Test 1: Full model creation
print("\n1. Testing full hist2scRNA model...")
try:
    model = hist2scRNA(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        vit_depth=4,  # Smaller for testing
        vit_heads=6,
        n_genes=500,
        n_cell_types=5,
        use_spatial_graph=True,
        dropout=0.1
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully")
    print(f"  - Parameters: {n_params:,}")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    sys.exit(1)

# Test 2: Forward pass with dummy image
print("\n2. Testing forward pass with dummy image...")
try:
    dummy_img = torch.randn(2, 3, 224, 224)  # 2 images

    # Create dummy spatial graph (2 spots connected)
    edge_index = torch.tensor([
        [0],
        [1]
    ], dtype=torch.long)

    output = model(dummy_img, edge_index)

    print(f"✓ Forward pass successful")
    print(f"  - mu shape: {output['mu'].shape} (expected: [2, 500])")
    print(f"  - theta shape: {output['theta'].shape} (expected: [2, 500])")
    print(f"  - pi shape: {output['pi'].shape} (expected: [2, 500])")
    print(f"  - cell_type_logits shape: {output['cell_type_logits'].shape} (expected: [2, 5])")

    # Verify shapes
    assert output['mu'].shape == (2, 500), "Incorrect mu shape"
    assert output['theta'].shape == (2, 500), "Incorrect theta shape"
    assert output['pi'].shape == (2, 500), "Incorrect pi shape"
    assert output['cell_type_logits'].shape == (2, 5), "Incorrect cell_type_logits shape"

    # Verify value ranges
    assert (output['mu'] > 0).all(), "mu should be positive"
    assert (output['theta'] > 0).all(), "theta should be positive"
    assert (output['pi'] >= 0).all() and (output['pi'] <= 1).all(), "pi should be in [0, 1]"

    print("✓ Output shapes and value ranges are correct")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: ZINB Loss
print("\n3. Testing ZINB Loss...")
try:
    criterion = ZINBLoss()

    # Create dummy targets
    dummy_expression = torch.randint(0, 10, (2, 500)).float()

    # Compute loss
    loss = criterion(output['mu'], output['theta'], output['pi'], dummy_expression)

    print(f"✓ ZINB Loss computed successfully")
    print(f"  - Loss value: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    print("✓ Loss is valid (not NaN or Inf)")

except Exception as e:
    print(f"✗ ZINB Loss test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Lightweight model
print("\n4. Testing lightweight hist2scRNA model...")
try:
    model_light = hist2scRNA_Lightweight(
        feature_dim=2048,
        n_genes=500,
        n_cell_types=5,
        dropout=0.1
    )

    n_params_light = sum(p.numel() for p in model_light.parameters())
    print(f"✓ Lightweight model created successfully")
    print(f"  - Parameters: {n_params_light:,}")

    # Test forward pass
    dummy_features = torch.randn(2, 2048)
    output_light = model_light(dummy_features)

    print(f"✓ Lightweight forward pass successful")
    print(f"  - mu shape: {output_light['mu'].shape}")

    assert output_light['mu'].shape == (2, 500), "Incorrect lightweight mu shape"

except Exception as e:
    print(f"✗ Lightweight model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Backward pass (gradient check)
print("\n5. Testing backward pass...")
try:
    model.zero_grad()
    loss.backward()

    # Check if gradients are computed
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.parameters() if p.requires_grad)

    if has_gradients:
        print("✓ Backward pass successful (gradients computed)")
    else:
        print("⚠ Warning: No gradients computed")

except Exception as e:
    print(f"✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nModel summary:")
print(f"  - Full model parameters: {n_params:,}")
print(f"  - Lightweight model parameters: {n_params_light:,}")
print(f"  - Parameter reduction: {(1 - n_params_light/n_params)*100:.1f}%")
print("\nThe hist2scRNA model is ready for training!")
