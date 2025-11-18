# hist2scRNA: State-of-the-Art Single-Cell RNA-seq Prediction from Histopathology Images

## Overview

**hist2scRNA** is a cutting-edge deep learning framework for predicting single-cell RNA sequencing (scRNA-seq) expression profiles directly from histopathology images. This extends the original hist2RNA project from bulk RNA prediction to spatially-resolved single-cell level predictions.

## Key Features

- **Vision Transformer (ViT)** architecture for superior feature extraction from H&E images
- **Graph Neural Networks (GNN)** for modeling spatial relationships between cells/spots
- **Zero-Inflated Negative Binomial (ZINB) loss** to properly handle the sparsity characteristic of single-cell data
- **Multi-task learning** with simultaneous gene expression and cell type prediction
- **Spatial attention mechanisms** for capturing tissue architecture
- **State-of-the-art performance** based on recent methods (GHIST, Hist2ST, TransformerST)

## Architecture

📊 **Visual Diagrams:** See [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) for comprehensive Mermaid diagrams of the architecture, data flow, and training process.

The hist2scRNA model consists of three main components:

### 1. Vision Transformer (ViT) Encoder
- Processes histopathology image patches (default: 224×224 pixels)
- Divides images into smaller patches (default: 16×16 pixels)
- Uses multi-head self-attention to capture long-range dependencies
- Generates rich feature representations for each spatial location

### 2. Spatial Graph Attention Network
- Models spatial relationships between neighboring spots/cells
- Uses Graph Attention Networks (GAT) to aggregate information from nearby locations
- Captures tissue microenvironment effects on gene expression
- Learns which neighbors are most relevant for prediction

### 3. Gene Expression Decoder
- Predicts parameters of Zero-Inflated Negative Binomial distribution:
  - **μ (mu)**: Mean expression level for each gene
  - **θ (theta)**: Dispersion parameter capturing gene-specific variability
  - **π (pi)**: Zero-inflation probability modeling dropout events
- Simultaneously predicts cell type for each spot (multi-task learning)

## Model Variants

### Full Model (`hist2scRNA`)
- Complete Vision Transformer with 6-12 layers
- Full spatial graph attention
- ~50-100M parameters
- Best performance for research applications

### Lightweight Model (`hist2scRNA_Lightweight`)
- Simplified architecture using pre-extracted features
- Efficient spatial attention without full ViT
- ~10-20M parameters
- Faster training and inference for production use

## Installation

```bash
# Clone the repository
git clone https://github.com/raktim-mondol/hist2RNA.git
cd hist2RNA

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (for GNN components)
pip install torch_geometric
```

## Quick Start

### 1. Generate Dummy Data (for testing)

```bash
python generate_dummy_scrna_data.py
```

This will create:
- `dummy_data/small/` - Small dataset (50 spots, 500 genes) for quick testing
- `dummy_data/medium/` - Medium dataset (200 spots, 2000 genes) for realistic testing

Each dataset includes:
- Synthetic H&E histopathology patches
- Simulated single-cell gene expression with realistic sparsity
- Spatial coordinates and graph structure
- Cell type labels
- Visualizations

### 2. Test Model Architecture

```bash
python test_scrna_model.py
```

This verifies that the model architecture is working correctly without requiring training.

### 3. Train the Model

```bash
# Train on small dataset (quick test)
python train_hist2scRNA.py \
    --data_dir ./dummy_data/small \
    --epochs 50 \
    --batch_size 8 \
    --model_type full \
    --output_dir ./output_scrna_small

# Train on medium dataset (realistic)
python train_hist2scRNA.py \
    --data_dir ./dummy_data/medium \
    --epochs 100 \
    --batch_size 4 \
    --model_type full \
    --lr 0.0001 \
    --output_dir ./output_scrna_medium
```

### 4. Evaluate the Model

```bash
# Evaluate with the comprehensive evaluation script
python evaluate_hist2scRNA.py

# Or evaluate a trained model programmatically
from evaluate_hist2scRNA import evaluate_model
import torch

model = torch.load('./output_scrna_medium/best_model.pt', weights_only=False)
results = evaluate_model(
    model=model,
    dataloader=test_dataloader,
    device='cuda',
    save_dir='./evaluation_results'
)
```

This will generate:
- Comprehensive metrics (MSE, MAE, R², Pearson/Spearman correlations)
- Gene-wise and spot-wise performance analysis
- Zero-inflation metrics (critical for scRNA-seq)
- Cell type classification metrics
- Visualization plots saved to `./evaluation_results/plots/`
- Results JSON file with all metrics

### 5. Training with Custom Parameters

```bash
python train_hist2scRNA.py \
    --data_dir /path/to/your/data \
    --model_type full \
    --img_size 224 \
    --patch_size 16 \
    --embed_dim 384 \
    --vit_depth 6 \
    --vit_heads 6 \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --alpha 0.1 \
    --output_dir ./output_scrna \
    --checkpoint_path ./output_scrna/best_model.pt
```

## Data Format

The model expects the following data structure:

```
data_dir/
├── patches/                    # Histopathology image patches
│   ├── spot_0000.png
│   ├── spot_0001.png
│   └── ...
├── gene_expression.csv         # Gene expression matrix (spots × genes)
├── spatial_coordinates.csv     # Spatial coordinates (spot_id, x, y)
├── spatial_edges.csv          # Graph edges (source, target)
├── cell_types.csv             # Cell type labels (spot_id, cell_type)
└── metadata.json              # Dataset metadata
```

### File Formats

#### gene_expression.csv
```csv
spot_id,Gene_0000,Gene_0001,Gene_0002,...
spot_0000,5.2,0.0,12.3,...
spot_0001,3.1,8.7,0.0,...
```

#### spatial_coordinates.csv
```csv
spot_id,x,y
spot_0000,1.5,2.3
spot_0001,1.7,2.5
```

#### spatial_edges.csv
```csv
source,target
0,1
0,2
1,3
```

#### cell_types.csv
```csv
spot_id,cell_type
spot_0000,0
spot_0001,1
```

## Model Parameters

### Architecture Parameters
- `img_size`: Input image size (default: 224)
- `patch_size`: Patch size for ViT (default: 16)
- `embed_dim`: Embedding dimension (default: 384)
- `vit_depth`: Number of transformer layers (default: 6)
- `vit_heads`: Number of attention heads (default: 6)
- `dropout`: Dropout rate (default: 0.1)

### Training Parameters
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Batch size (default: 8)
- `lr`: Learning rate (default: 0.0001)
- `weight_decay`: Weight decay for AdamW (default: 0.01)
- `alpha`: Weight for cell type classification loss (default: 0.1)

## Evaluation Metrics

### Comprehensive Evaluation Script

Use `evaluate_hist2scRNA.py` for comprehensive model evaluation:

```bash
# Run standalone evaluation example with dummy data
python evaluate_hist2scRNA.py

# Or use in your code
from evaluate_hist2scRNA import evaluate_model, scRNAEvaluator
import torch

# Evaluate a trained model
model = torch.load('best_model.pt')
results = evaluate_model(model, test_dataloader, device='cuda', save_dir='./results')
```

### Performance Metrics

The evaluation script computes the following metrics:

#### 1. **Overall Regression Metrics** (across all genes and spots)
   - **MSE**: Mean Squared Error
   - **RMSE**: Root Mean Squared Error
   - **MAE**: Mean Absolute Error
   - **R²**: Coefficient of determination (variance explained)
   - **Pearson correlation**: Linear relationship between true and predicted
   - **Spearman correlation**: Rank-based correlation (robust to outliers)

#### 2. **Gene-wise Metrics** (per-gene correlation across spots)
   - Mean/median Pearson correlation per gene
   - Mean/median Spearman correlation per gene
   - MSE and MAE for each gene
   - Expression statistics (mean, std) for true vs predicted

   *Important: Shows how well the model captures expression patterns for each individual gene*

#### 3. **Spot-wise Metrics** (per-spot correlation across genes)
   - Mean/median Pearson correlation per spot
   - Mean/median Spearman correlation per spot

   *Important: Shows how well the model captures the full expression profile at each spatial location*

#### 4. **Zero-Inflation Metrics** (handling sparsity)
   - **Zero rate**: Proportion of zeros in true vs predicted data
   - **Zero detection accuracy**: Ability to classify zero vs non-zero
   - **Sensitivity**: True positive rate (detecting non-zero values)
   - **Specificity**: True negative rate (detecting zero values)
   - **F1 score**: Harmonic mean of precision and recall
   - **MAE for non-zero values**: Error excluding dropout events

   *Critical for single-cell data: 70-90% of values are zero*

#### 5. **Cell Type Classification**
   - **Accuracy**: Overall classification accuracy
   - **F1 scores**: Macro and weighted F1
   - **Confusion matrix**: Per-class performance
   - **Classification report**: Precision, recall, F1 per class

#### 6. **Expression Level Analysis**
   - Performance stratified by expression level (low/medium/high)
   - Identifies if model performs better/worse on highly vs lowly expressed genes

### Visualization Outputs

The evaluation script automatically generates:
- **Overall scatter plot**: True vs predicted expression (10K random samples)
- **Gene-wise correlation distribution**: Histogram of per-gene correlations
- **Zero-inflation analysis**: Zero vs non-zero comparison
- **Expression levels plot**: Performance across expression ranges
- **Confusion matrix**: Cell type classification heatmap

All plots are saved to `{save_dir}/plots/` and results to `{save_dir}/evaluation_results.json`

### Example Output

```
[Overall Metrics]
  MSE:           2.5706
  RMSE:          1.6033
  MAE:           1.0319
  R²:            0.9353
  Pearson r:     0.9714
  Spearman ρ:    0.8024

[Gene-wise Metrics]
  Mean Pearson r:   0.9705
  Median Pearson r: 0.9715

[Spot-wise Metrics]
  Mean Pearson r:   0.9712
  Median Pearson r: 0.9708

[Zero-Inflation Metrics]
  Zero rate (true): 0.7018
  Zero rate (pred): 0.4248
  Zero detection accuracy: 0.7152
  Sensitivity:      0.9870
  Specificity:      0.5997

[Cell Type Classification]
  Accuracy:       0.7500
  F1 (macro):     0.7401
```

### Training Metrics

During training, the following losses are monitored:

- **ZINB loss**: Zero-Inflated Negative Binomial loss for gene expression
- **Cross-entropy loss**: Cell type classification loss
- **Total combined loss**: Weighted sum of ZINB and CE losses

## Advanced Usage

### Using Pre-extracted Features

If you have pre-extracted features from a foundation model (e.g., UNI, Virchow):

```python
from hist2scRNA_model import hist2scRNA_Lightweight

model = hist2scRNA_Lightweight(
    feature_dim=2048,  # Dimension of pre-extracted features
    n_genes=2000,
    n_cell_types=10
)

# Train with pre-extracted features instead of images
features = torch.randn(batch_size, 2048)
output = model(features)
```

### Custom Loss Functions

You can modify the ZINB loss or add custom losses:

```python
from hist2scRNA_model import ZINBLoss
import torch.nn as nn

# Combine ZINB with additional regularization
criterion_zinb = ZINBLoss()
criterion_ce = nn.CrossEntropyLoss()

# Custom loss with additional spatial smoothness term
def custom_loss(output, target, spatial_graph):
    zinb_loss = criterion_zinb(output['mu'], output['theta'], output['pi'], target)
    ce_loss = criterion_ce(output['cell_type_logits'], cell_types)

    # Add spatial smoothness regularization
    # (expressions of neighboring spots should be similar)
    spatial_loss = compute_spatial_smoothness(output['mu'], spatial_graph)

    return zinb_loss + 0.1 * ce_loss + 0.01 * spatial_loss
```

## Theoretical Background

### Why Zero-Inflated Negative Binomial (ZINB)?

Single-cell RNA-seq data has two key characteristics:

1. **Sparsity (Zero-inflation)**: Many genes show zero expression due to:
   - Biological absence (gene truly not expressed)
   - Technical dropout (RNA molecule not captured)
   - ZINB models both sources with parameter π

2. **Overdispersion**: Expression variance exceeds the mean
   - Negative Binomial captures this with dispersion parameter θ
   - Better than Poisson or Gaussian distributions

### Why Vision Transformers?

Recent research shows ViTs outperform CNNs for histopathology because:
- Self-attention captures long-range tissue architecture
- Better modeling of global context (tumor microenvironment)
- Patch-based processing natural for tissue organization
- Pre-training on large histopathology datasets available

### Why Graph Neural Networks?

Spatial transcriptomics data has inherent graph structure:
- Spots/cells are nodes with spatial coordinates
- Edges connect spatially proximal spots
- Gene expression is influenced by neighboring cells
- GAT learns which neighbors matter most for each spot

## Comparison with Other Methods

| Method | Architecture | Spatial Modeling | Loss Function | Year |
|--------|-------------|------------------|---------------|------|
| ST-Net | CNN | None | MSE | 2020 |
| Hist2ST | CNN + Transformer + GNN | Graph | ZINB | 2022 |
| HisToGene | ViT | Positional encoding | MSE | 2023 |
| TransformerST | ViT | Positional encoding | MSE | 2024 |
| GHIST | ViT + scRNA integration | Attention | Custom | 2025 |
| **hist2scRNA** | **ViT + GAT** | **Graph + Attention** | **ZINB + Multi-task** | **2025** |

## Performance Tips

1. **For faster training:**
   - Use `hist2scRNA_Lightweight` model
   - Reduce `embed_dim` and `vit_depth`
   - Increase `batch_size` if GPU memory allows
   - Use pre-extracted features

2. **For better performance:**
   - Use full `hist2scRNA` model
   - Increase `vit_depth` to 8-12
   - Use larger `embed_dim` (512-768)
   - Train longer (100-200 epochs)
   - Use data augmentation

3. **For limited GPU memory:**
   - Reduce `batch_size`
   - Use gradient accumulation
   - Reduce `embed_dim`
   - Use mixed precision training (fp16)

## Citation

If you use hist2scRNA in your research, please cite:

```bibtex
@Article{cancers15092569,
  AUTHOR = {Mondol, Raktim Kumar and Millar, Ewan K. A. and Graham, Peter H. and Browne, Lois and Sowmya, Arcot and Meijering, Erik},
  TITLE = {hist2RNA: An Efficient Deep Learning Architecture to Predict Gene Expression from Breast Cancer Histopathology Images},
  JOURNAL = {Cancers},
  VOLUME = {15},
  YEAR = {2023},
  NUMBER = {9},
  ARTICLE-NUMBER = {2569},
  URL = {https://www.mdpi.com/2072-6694/15/9/2569},
  DOI = {10.3390/cancers15092569}
}
```

And relevant papers for the techniques used:
- **GHIST**: He et al., "Spatial gene expression at single-cell resolution from histology using deep learning with GHIST", Nature Methods (2025)
- **Hist2ST**: Zeng et al., "Spatial transcriptomics prediction from histology jointly through Transformer and graph neural networks", Briefings in Bioinformatics (2022)
- **Vision Transformers**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR (2021)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This work builds upon:
- Original hist2RNA framework for bulk RNA prediction
- Recent advances in spatial transcriptomics prediction (GHIST, Hist2ST, TransformerST)
- Vision Transformer architectures for computational pathology
- Graph neural networks for spatial modeling

## Support

For questions and issues:
- Open an issue on GitHub
- Check existing documentation
- Contact: [raktim.mondol@example.com](mailto:raktim.mondol@example.com)

## Roadmap

Future enhancements:
- [ ] Integration with foundation models (UNI, Virchow, CONCH)
- [ ] Support for multi-modal data (H&E + IHC + IF)
- [ ] Attention visualization and interpretability
- [ ] Pre-trained models on public datasets
- [ ] Benchmark on 10X Visium and other platforms
- [ ] Super-resolution gene expression prediction
- [ ] Integration with cell segmentation methods
