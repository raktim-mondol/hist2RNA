# Quick Start Guide: hist2scRNA Single-Cell RNA-seq Prediction

## 5-Minute Quick Start

This guide will get you up and running with hist2scRNA in just a few minutes.

📊 **Want to understand the architecture?** See [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) for comprehensive visual diagrams.

## Step 1: Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/raktim-mondol/hist2RNA.git
cd hist2RNA

# Install dependencies
pip install -r requirements.txt

# Note: If torch_geometric installation fails, use:
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch_geometric
```

## Step 2: Generate Test Data (1 minute)

```bash
# Generate dummy single-cell data for testing
python generate_dummy_scrna_data.py
```

This creates two datasets:
- `dummy_data/small/` - Quick testing (50 spots, 500 genes)
- `dummy_data/medium/` - Realistic testing (200 spots, 2000 genes)

## Step 3: Test Model Architecture (30 seconds)

```bash
# Verify the model is working correctly
python test_scrna_model.py
```

Expected output:
```
Testing hist2scRNA model architecture...
============================================================
✓ Successfully imported model modules

1. Testing full hist2scRNA model...
✓ Model created successfully
  - Parameters: 45,234,560

2. Testing forward pass with dummy image...
✓ Forward pass successful
  - mu shape: torch.Size([2, 500])
  - theta shape: torch.Size([2, 500])
  - pi shape: torch.Size([2, 500])
  - cell_type_logits shape: torch.Size([2, 5])
✓ Output shapes and value ranges are correct

...

ALL TESTS PASSED! ✓
```

## Step 4: Train Your First Model (1.5 minutes)

```bash
# Quick training on small dataset
python train_hist2scRNA.py \
    --data_dir ./dummy_data/small \
    --epochs 10 \
    --batch_size 8 \
    --model_type full \
    --output_dir ./output_scrna_quickstart
```

## Understanding the Output

After training, you'll find:

```
output_scrna_quickstart/
├── best_model.pt              # Trained model checkpoint
├── training_curves.png        # Training/validation loss plots
└── results.json              # Final metrics and parameters
```

### Sample Results

```json
{
  "test_metrics": {
    "loss": 2.3456,
    "mse": 15.67,
    "mae": 2.89,
    "mean_spot_corr": 0.72,
    "mean_gene_corr": 0.65,
    "cell_type_accuracy": 0.85
  }
}
```

## What's Next?

### For Research Use

Train on realistic dataset:
```bash
python train_hist2scRNA.py \
    --data_dir ./dummy_data/medium \
    --epochs 100 \
    --batch_size 4 \
    --model_type full \
    --lr 0.0001 \
    --output_dir ./output_scrna_research
```

### For Production Use

Use the lightweight model:
```bash
python train_hist2scRNA.py \
    --data_dir ./dummy_data/medium \
    --epochs 50 \
    --batch_size 16 \
    --model_type lightweight \
    --output_dir ./output_scrna_production
```

### Using Your Own Data

1. Prepare your data in the required format (see SCRNA_README.md)
2. Place images in `your_data/patches/`
3. Create CSV files for expression, coordinates, edges, and cell types
4. Train:

```bash
python train_hist2scRNA.py \
    --data_dir ./your_data \
    --epochs 100 \
    --output_dir ./output_your_data
```

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or use lightweight model
```bash
python train_hist2scRNA.py --batch_size 2 --model_type lightweight
```

### Issue 2: torch_geometric Not Found
```
ModuleNotFoundError: No module named 'torch_geometric'
```
**Solution:** Install PyTorch Geometric manually
```bash
pip install torch_geometric
```

### Issue 3: Training is Too Slow
**Solutions:**
- Use GPU: Model will automatically use CUDA if available
- Reduce model size: `--embed_dim 256 --vit_depth 4`
- Use lightweight model: `--model_type lightweight`
- Increase batch size: `--batch_size 16` (if memory allows)

## Key Parameters Explained

| Parameter | What it does | Default | When to change |
|-----------|-------------|---------|----------------|
| `--epochs` | Training iterations | 50 | Increase for better performance |
| `--batch_size` | Samples per batch | 8 | Decrease if OOM, increase for speed |
| `--lr` | Learning rate | 0.0001 | Decrease if loss unstable |
| `--model_type` | Model variant | full | Use 'lightweight' for speed |
| `--embed_dim` | Model size | 384 | Increase for larger capacity |
| `--vit_depth` | Transformer layers | 6 | Increase for better performance |

## Performance Tips

### 🚀 For Faster Training
```bash
python train_hist2scRNA.py \
    --model_type lightweight \
    --batch_size 16 \
    --embed_dim 256 \
    --vit_depth 4
```

### 🎯 For Best Accuracy
```bash
python train_hist2scRNA.py \
    --model_type full \
    --batch_size 4 \
    --embed_dim 768 \
    --vit_depth 12 \
    --epochs 200
```

### 💾 For Limited GPU Memory
```bash
python train_hist2scRNA.py \
    --model_type lightweight \
    --batch_size 2 \
    --embed_dim 256
```

## Interpreting Results

### Good Model Performance
- Mean spot correlation > 0.7
- Mean gene correlation > 0.6
- Cell type accuracy > 0.8
- Validation loss decreasing

### Signs of Overfitting
- Training loss much lower than validation loss
- Validation loss increasing while training loss decreases
- **Solution:** Add more dropout, reduce model size, or get more data

### Signs of Underfitting
- Both losses high and not decreasing
- Poor correlations (< 0.5)
- **Solution:** Increase model capacity, train longer, or reduce regularization

## Advanced Usage

### Custom Data Augmentation

```python
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Use in dataset
dataset = SpatialTranscriptomicsDataset(
    data_dir='./your_data',
    transform=custom_transform
)
```

### Loading Trained Model for Inference

```python
import torch
from hist2scRNA_model import hist2scRNA

# Load model
model = hist2scRNA(n_genes=2000, n_cell_types=10)
checkpoint = torch.load('output_scrna/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model.predict(images, edge_index)
```

## Comparison with Bulk RNA Model

| Feature | hist2RNA (Bulk) | hist2scRNA (Single-cell) |
|---------|----------------|-------------------------|
| Input | H&E patches | H&E patches + spatial graph |
| Output | Bulk gene expression | Single-cell expression + cell type |
| Architecture | CNN | Vision Transformer + GNN |
| Loss | MSE | ZINB (handles sparsity) |
| Spatial modeling | Patch aggregation | Graph attention |
| Use case | Patient-level prediction | Spot/cell-level prediction |

## Resources

- **Full Documentation:** See `SCRNA_README.md`
- **Original Paper:** [hist2RNA Paper](https://www.mdpi.com/2072-6694/15/9/2569)
- **Issues:** [GitHub Issues](https://github.com/raktim-mondol/hist2RNA/issues)

## Support

Questions? Check:
1. Full README: `SCRNA_README.md`
2. Test script: `test_scrna_model.py`
3. Training script: `train_hist2scRNA.py`
4. GitHub Issues

**Congratulations!** You're now ready to use hist2scRNA for single-cell RNA-seq prediction! 🎉
