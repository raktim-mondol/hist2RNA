# hist2scRNA: Single-Cell RNA-seq Prediction from Histopathology Images

This folder contains the **hist2scRNA** model implementation - a state-of-the-art deep learning framework for predicting single-cell and spatial transcriptomics data from histopathology images.

## 📁 Folder Structure

```
hist2scRNA/
├── models/                          # Model architectures
│   ├── base_model.py               # Main hist2scRNA model
│   ├── architectures/              # Model components
│   │   ├── vision_transformer.py   # Vision Transformer
│   │   ├── graph_attention.py      # Graph Attention Network
│   │   └── lightweight.py          # Lightweight variant
│   ├── layers/                     # Custom layers
│   │   ├── patch_embedding.py      # Patch embedding layer
│   │   ├── attention.py            # Multi-head attention
│   │   └── transformer_block.py    # Transformer block
│   ├── losses/                     # Loss functions
│   │   └── zinb_loss.py           # ZINB loss for scRNA data
│   └── checkpoints/                # Saved model weights
│
├── src/                            # Source code
│   ├── data/                       # Data handling
│   │   ├── dataset.py             # Dataset classes
│   │   ├── dataloader.py          # Data loading utilities
│   │   ├── transforms.py          # Data augmentation
│   │   └── generators.py          # Dummy data generation
│   ├── training/                   # Training logic
│   │   └── trainer.py             # Training loop
│   ├── evaluation/                 # Evaluation
│   │   ├── evaluator.py           # Evaluation logic
│   │   └── metrics.py             # Metrics computation
│   └── utils/                      # Utilities
│       ├── config.py              # Configuration system
│       ├── visualization.py       # Plotting functions
│       └── helpers.py             # Helper functions
│
├── experiments/                    # Experiment configurations
│   ├── config/                    # Configuration files
│   │   └── default.yaml           # Default config
│   └── logs/                      # Training logs
│
├── tests/                          # Unit tests
│   ├── test_model.py              # Model architecture tests
│   └── test_data.py               # Data handling tests
│
├── docs/                           # Documentation
│   ├── MODULAR_README.md          # Detailed modular guide
│   ├── SCRNA_README.md            # Full documentation
│   ├── QUICKSTART_SCRNA.md        # Quick start guide
│   ├── INPUT_IMAGE_WORKFLOW.md    # Image preprocessing guide
│   ├── ARCHITECTURE_DIAGRAMS.md   # Visual diagrams
│   └── diagrams/                  # Mermaid diagram files
│
├── data/                           # Data storage
│   ├── raw/                       # Original data
│   ├── processed/                 # Preprocessed data
│   └── splits/                    # Train/val/test splits
│
├── notebooks/                      # Jupyter notebooks
│
├── train.py                        # Training entry point
├── evaluate.py                     # Evaluation entry point
├── config.yaml                     # Main configuration
├── requirements.txt                # Python dependencies
└── setup.py                        # Package setup
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 1. Generate Test Data

```bash
cd hist2scRNA
python -c "from src.data import save_dummy_data; save_dummy_data(n_spots=50, n_genes=500, n_cell_types=5, output_dir='./dummy_data/small')"
```

### 2. Test Model Architecture

```bash
# Test model components
python tests/test_model.py

# Test data handling
python tests/test_data.py

# Run all tests
pytest tests/
```

### 3. Train the Model

```bash
# Quick training with defaults
python train.py --data_dir ./dummy_data/small --epochs 20 --batch_size 4

# Training with custom configuration
python train.py --config experiments/config/default.yaml

# Training with specific parameters
python train.py \
    --data_dir ./dummy_data/small \
    --epochs 50 \
    --batch_size 8 \
    --embed_dim 384 \
    --vit_depth 6 \
    --output_dir ./output_scrna
```

### 4. Evaluate the Model

```bash
python evaluate.py \
    --checkpoint ./models/checkpoints/best_model.pt \
    --data_dir ./dummy_data/small \
    --output_dir ./evaluation_results
```

## 📚 Documentation

- **[docs/MODULAR_README.md](docs/MODULAR_README.md)** - Complete guide to modular architecture
- **[docs/SCRNA_README.md](docs/SCRNA_README.md)** - Comprehensive documentation
- **[docs/QUICKSTART_SCRNA.md](docs/QUICKSTART_SCRNA.md)** - 5-minute quick start guide
- **[docs/INPUT_IMAGE_WORKFLOW.md](docs/INPUT_IMAGE_WORKFLOW.md)** - 📸 Image preprocessing guide
- **[docs/ARCHITECTURE_DIAGRAMS.md](docs/ARCHITECTURE_DIAGRAMS.md)** - Visual architecture diagrams

## 🏗️ Architecture Overview

The hist2scRNA model combines:
- **Vision Transformer (ViT)** for feature extraction from H&E images
- **Graph Attention Networks (GAT)** for spatial relationship modeling
- **Zero-Inflated Negative Binomial (ZINB)** loss for single-cell sparsity
- **Multi-task learning** for gene expression + cell type prediction

## 🔑 Key Features

✅ **Modular Architecture** - Clean separation of concerns for easy extensibility
✅ **State-of-the-art Vision Transformer** - Latest ViT architecture for histopathology
✅ **Spatial Graph Attention** - Models tissue microenvironment relationships
✅ **Handles 70-90% Sparsity** - ZINB loss designed for single-cell data
✅ **Multi-task Learning** - Simultaneous gene expression and cell type prediction
✅ **Flexible Configuration** - YAML-based config system for experiments
✅ **Comprehensive Testing** - Unit tests for all components
✅ **Compatible Platforms** - Works with 10X Visium and spatial transcriptomics

## 🎯 Model Variants

| Variant | Parameters | Embed Dim | Depth | Use Case |
|---------|-----------|-----------|-------|----------|
| **Small** | ~10M | 256 | 4 | Development, quick iterations |
| **Medium** | ~50M | 384 | 6 | Research, balanced performance |
| **Large** | ~100M | 512 | 8 | Production, high accuracy |
| **Lightweight** | ~15M | 1024 | - | Edge devices, real-time inference |

Configure via `--embed_dim` and `--vit_depth` parameters or use `--model_type lightweight`.

## 💻 Python API Usage

```python
import torch
from models import hist2scRNA, ZINBLoss
from src.data import SpatialTranscriptomicsDataset, get_default_transforms
from src.training import hist2scRNATrainer
from src.utils import set_seed, get_device

# Set seed for reproducibility
set_seed(42)

# Load data
transform = get_default_transforms(augment=False)
dataset = SpatialTranscriptomicsDataset('./dummy_data/small', transform=transform)

# Create model
model = hist2scRNA(
    img_size=224,
    embed_dim=384,
    vit_depth=6,
    n_genes=dataset.n_genes,
    n_cell_types=dataset.metadata['n_cell_types']
)

# Setup training
device = get_device('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = ZINBLoss()

# See docs/MODULAR_README.md for complete training example
```

## 🔬 Based on Latest Research

- **GHIST** (Nature Methods, 2025) - Single-cell resolution prediction
- **Hist2ST** (2022) - Transformer + GNN for spatial transcriptomics
- **TransformerST** (2024) - ViT for gene expression prediction

## 🔗 Parent Project

This is an extension of the [hist2RNA](../) project for bulk RNA prediction. See the main README in the parent directory for the original bulk RNA prediction model.

## 📝 Citation

If you use hist2scRNA in your research, please cite the original hist2RNA paper:

```bibtex
@Article{cancers15092569,
  AUTHOR = {Mondol, Raktim Kumar and Millar, Ewan K. A. and Graham, Peter H. and Browne, Lois and Sowmya, Arcot and Meijering, Erik},
  TITLE = {hist2RNA: An Efficient Deep Learning Architecture to Predict Gene Expression from Breast Cancer Histopathology Images},
  JOURNAL = {Cancers},
  VOLUME = {15},
  YEAR = {2023},
  NUMBER = {9},
  ARTICLE-NUMBER = {2569},
  DOI = {10.3390/cancers15092569}
}
```

## 📧 Support

For questions and issues:
- Check the [docs/MODULAR_README.md](docs/MODULAR_README.md) for architecture details
- See [docs/QUICKSTART_SCRNA.md](docs/QUICKSTART_SCRNA.md) for common issues
- Review [docs/SCRNA_README.md](docs/SCRNA_README.md) for comprehensive documentation
- Open an issue on the main repository

## 🧪 Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Run specific test
python tests/test_model.py

# Generate dummy data for testing
python -c "from src.data import save_dummy_data; save_dummy_data()"
```

---

**Ready to get started?** Check out the [Quick Start Guide](docs/QUICKSTART_SCRNA.md) or [Modular Architecture Guide](docs/MODULAR_README.md)!
