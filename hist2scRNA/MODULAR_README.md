# hist2scRNA - Modular Implementation

This is the refactored, modular implementation of hist2scRNA for single-cell RNA-seq prediction from histopathology images.

## Directory Structure

```
hist2scRNA/
├── data/                       # Data storage
│   ├── raw/                    # Original, immutable data
│   ├── processed/              # Cleaned, preprocessed data
│   ├── external/               # External data sources
│   └── splits/                 # Train/validation/test splits
│
├── models/                     # Model architectures
│   ├── __init__.py
│   ├── base_model.py           # Main hist2scRNA model
│   ├── architectures/          # Model components
│   │   ├── vision_transformer.py
│   │   ├── graph_attention.py
│   │   └── lightweight.py
│   ├── layers/                 # Custom layers
│   │   ├── patch_embedding.py
│   │   ├── attention.py
│   │   └── transformer_block.py
│   ├── losses/                 # Loss functions
│   │   └── zinb_loss.py
│   └── checkpoints/            # Saved model weights
│
├── src/                        # Source code
│   ├── data/                   # Data handling
│   │   ├── dataset.py          # Dataset classes
│   │   ├── dataloader.py       # Data loading utilities
│   │   ├── transforms.py       # Data augmentation
│   │   └── generators.py       # Dummy data generation
│   ├── training/               # Training logic
│   │   └── trainer.py          # Training loop
│   ├── evaluation/             # Evaluation
│   │   ├── evaluator.py        # Evaluation logic
│   │   └── metrics.py          # Evaluation metrics
│   └── utils/                  # Utilities
│       ├── config.py           # Configuration
│       ├── visualization.py    # Plotting
│       └── helpers.py          # Helper functions
│
├── experiments/                # Experiment configurations
│   ├── config/                 # Configuration files
│   │   └── default.yaml
│   └── logs/                   # Training logs
│
├── notebooks/                  # Jupyter notebooks
│
├── tests/                      # Unit tests
│   ├── test_model.py
│   └── test_data.py
│
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation entry point
├── config.yaml                 # Main configuration
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # Documentation
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Generate Dummy Data

```bash
cd hist2scRNA
python -c "from src.data import save_dummy_data; save_dummy_data(n_spots=50, n_genes=500, n_cell_types=5, output_dir='./dummy_data/small')"
```

### Train a Model

```bash
# Using default configuration
python train.py --data_dir ./dummy_data/small --epochs 20 --batch_size 4

# Using configuration file
python train.py --config experiments/config/default.yaml

# With custom settings
python train.py --data_dir ./dummy_data/small \
                --epochs 50 \
                --batch_size 8 \
                --embed_dim 384 \
                --vit_depth 6
```

### Evaluate a Model

```bash
python evaluate.py --checkpoint ./models/checkpoints/best_model.pt \
                   --data_dir ./dummy_data/small \
                   --output_dir ./evaluation_results
```

### Run Tests

```bash
# Test model architectures
python tests/test_model.py

# Test data handling
python tests/test_data.py

# Run all tests with pytest
pytest tests/
```

## Usage Examples

### Python API

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

# Train (simplified example)
device = get_device('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = ZINBLoss()

# ... (see train.py for complete example)
```

### Configuration-Based Training

Create a custom configuration file:

```yaml
# my_config.yaml
model:
  embed_dim: 512
  vit_depth: 8
  vit_heads: 8

training:
  epochs: 100
  batch_size: 16
  lr: 0.0001

data:
  data_dir: './my_data'
  augment: true
```

Then train:

```bash
python train.py --config my_config.yaml
```

## Key Features

### Modular Design
- **Separate concerns**: Models, data, training, and evaluation are clearly separated
- **Easy to extend**: Add new models, losses, or data loaders easily
- **Reusable components**: Share layers, metrics, and utilities across experiments

### Flexible Configuration
- **YAML-based configs**: Store and version control your experiment settings
- **Command-line override**: Override config values from command line
- **Default configs**: Sensible defaults for quick experimentation

### Comprehensive Evaluation
- **Multiple metrics**: Overall, gene-wise, spot-wise, and zero-inflation metrics
- **Visualization**: Automatic plot generation for analysis
- **Detailed reports**: JSON output with all metrics

### Testing
- **Unit tests**: Test individual components
- **Integration tests**: Test full pipelines
- **Easy to run**: Simple pytest integration

## Model Architecture

The hist2scRNA model combines:

1. **Vision Transformer (ViT)** - Extracts features from histopathology image patches
2. **Graph Attention Network (GAT)** - Models spatial relationships between tissue spots
3. **ZINB Decoder** - Predicts gene expression with zero-inflation modeling

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hist2scrna2024,
  title={hist2scRNA: Single-cell RNA-seq Prediction from Histopathology Images},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
