# hist2scRNA: Single-Cell RNA-seq Prediction from Histopathology Images

This folder contains the **hist2scRNA** model implementation - a state-of-the-art deep learning framework for predicting single-cell and spatial transcriptomics data from histopathology images.

## 📁 Folder Structure

```
hist2scRNA/
├── hist2scRNA_model.py              # Core model implementation (ViT + GAT + ZINB)
├── train_hist2scRNA.py              # Training pipeline
├── test_scrna_model.py              # Model architecture tests
├── generate_dummy_scrna_data.py     # Synthetic data generation
├── SCRNA_README.md                  # Full documentation
├── QUICKSTART_SCRNA.md              # Quick start guide
├── ARCHITECTURE_DIAGRAMS.md         # Diagram documentation
└── diagrams/                        # Mermaid diagram files
    ├── 01_overall_architecture.mmd
    ├── 02_data_structure_flow.mmd
    ├── 03_training_pipeline.mmd
    ├── 04_vision_transformer_block.mmd
    ├── 05_graph_attention_network.mmd
    ├── 06_zinb_distribution.mmd
    ├── 07_inference_pipeline.mmd
    ├── 08_data_flow_sequence.mmd
    ├── 09_multitask_learning.mmd
    ├── 10_bulk_vs_singlecell.mmd
    ├── 11_model_scalability.mmd
    └── README.md
```

## 🚀 Quick Start

### 1. Generate Test Data
```bash
cd hist2scRNA
python generate_dummy_scrna_data.py
```

### 2. Test Model Architecture
```bash
python test_scrna_model.py
```

### 3. Train the Model
```bash
python train_hist2scRNA.py \
    --data_dir ./dummy_data/small \
    --epochs 50 \
    --batch_size 8 \
    --output_dir ./output_scrna
```

## 📚 Documentation

- **[SCRNA_README.md](SCRNA_README.md)** - Comprehensive documentation
- **[QUICKSTART_SCRNA.md](QUICKSTART_SCRNA.md)** - 5-minute quick start guide
- **[ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)** - Visual architecture diagrams

## 🏗️ Architecture Overview

The hist2scRNA model combines:
- **Vision Transformer (ViT)** for feature extraction from H&E images
- **Graph Attention Networks (GAT)** for spatial relationship modeling
- **Zero-Inflated Negative Binomial (ZINB)** loss for single-cell sparsity
- **Multi-task learning** for gene expression + cell type prediction

## 🔑 Key Features

✅ State-of-the-art Vision Transformer architecture
✅ Spatial graph attention for tissue microenvironment
✅ Handles 70-90% sparsity in single-cell data
✅ Simultaneous gene expression and cell type prediction
✅ Compatible with 10X Visium and spatial transcriptomics platforms

## 📊 Model Variants

| Variant | Parameters | Use Case |
|---------|-----------|----------|
| **Small** | ~10M | Development, quick iterations |
| **Medium** | ~50M | Research, balanced performance |
| **Large** | ~300M | Production, maximum accuracy |
| **Lightweight** | ~15M | Edge devices, real-time inference |

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
- Check the [SCRNA_README.md](SCRNA_README.md) documentation
- See [QUICKSTART_SCRNA.md](QUICKSTART_SCRNA.md) for common issues
- Open an issue on the main repository

---

**Ready to get started?** Check out the [Quick Start Guide](QUICKSTART_SCRNA.md)!
