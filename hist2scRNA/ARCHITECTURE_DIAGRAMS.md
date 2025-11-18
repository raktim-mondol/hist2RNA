# hist2scRNA Architecture Diagrams

Comprehensive visual documentation of the hist2scRNA model architecture, data flow, and training process.

## 📁 Diagram Files

All diagrams are available as individual `.mmd` files in the [`diagrams/`](./diagrams/) folder for direct viewing on GitHub.

**Quick Access:** See [diagrams/README.md](./diagrams/README.md) for the complete index.

---

## 📊 Diagram Overview

### 1. Overall Architecture
**File:** [`diagrams/01_overall_architecture.mmd`](./diagrams/01_overall_architecture.mmd)

Shows the complete model flow from input to output:
- **Input Processing**: H&E images (224×224×3) and spatial coordinates
- **Vision Transformer Encoder**: Patch embedding → Multi-head self-attention → Transformer blocks
- **Spatial Graph Attention**: k-NN graph construction → GAT layers
- **Gene Expression Decoder**: Dense layers → ZINB parameter heads (μ, θ, π)
- **Cell Type Prediction**: Multi-task classification head
- **Output**: Predicted gene expression + cell types

---

### 2. Data Structure and Flow
**File:** [`diagrams/02_data_structure_flow.mmd`](./diagrams/02_data_structure_flow.mmd)

Documents the data preprocessing pipeline:
- **Raw Data**: WSI images, 10X Visium spatial transcriptomics
- **Preprocessing**: Color normalization (Macenko), k-NN graph construction, log2(1+x) transform
- **Dataset Structure**: File organization (patches/, spatial_edges.csv, gene_expression.csv, etc.)
- **Model Input**: Tensor formats (batch×3×224×224, edge_index, coordinates)

---

### 3. Training Pipeline
**File:** [`diagrams/03_training_pipeline.mmd`](./diagrams/03_training_pipeline.mmd)

Complete training workflow:
- **Data Splitting**: 70% train, 15% validation, 15% test
- **Training Loop**: Batch processing → Forward pass → Loss computation → Backpropagation
- **Loss Function**: ZINB loss + Cross-Entropy loss (weighted combination)
- **Validation**: Metrics computation (Spearman correlation, cell type accuracy)
- **Early Stopping**: Based on validation loss with patience
- **Checkpoint Saving**: Best model preservation
- **Test Evaluation**: Final performance on held-out test set

---

### 4. Vision Transformer Block
**File:** [`diagrams/04_vision_transformer_block.mmd`](./diagrams/04_vision_transformer_block.mmd)

Internal structure of a single transformer block:
- **Layer Normalization**: Pre-normalization architecture
- **Multi-Head Self-Attention**: Q, K, V projections → Scaled dot-product → Multi-head concat
- **Residual Connections**: Skip connections for gradient flow
- **Feed-Forward Network**: 2-layer MLP with GELU activation (4× expansion)
- **Dropout**: Regularization throughout

---

### 5. Graph Attention Network
**File:** [`diagrams/05_graph_attention_network.mmd`](./diagrams/05_graph_attention_network.mmd)

Spatial relationship modeling:
- **Node Features**: Spot-level representations
- **Edge Index**: Spatial graph structure (2×n_edges)
- **Attention Mechanism**: Learn importance weights for each neighbor
- **Message Passing**: Weighted aggregation of neighbor features
- **Multi-Head**: Multiple attention heads for diverse relationships

---

### 6. ZINB Distribution and Loss
**File:** [`diagrams/06_zinb_distribution.mmd`](./diagrams/06_zinb_distribution.mmd)

Zero-Inflated Negative Binomial modeling:
- **Three Parameter Heads**: μ (mean), θ (dispersion), π (zero-inflation)
- **Zero Case**: Models structural zeros and dropouts
- **Non-Zero Case**: Negative binomial distribution for counts
- **Loss Computation**: Log-likelihood of observed expression

**Why ZINB?** Single-cell data has 70-90% zeros (sparsity), captures both biological zeros and technical dropouts, models overdispersion (variance > mean).

---

### 7. Inference Pipeline
**File:** [`diagrams/07_inference_pipeline.mmd`](./diagrams/07_inference_pipeline.mmd)

Production deployment workflow:
- **Input Preparation**: Image resize, normalization, optional graph construction
- **Feature Extraction**: Vision Transformer forward pass
- **Spatial Context**: Graph attention (if available)
- **Prediction**: ZINB parameters + cell type probabilities
- **Output**: Gene expression (μ), uncertainty (θ), dropout (π), cell type
- **Visualization**: Heatmaps, spatial maps, uncertainty maps

---

### 8. Data Flow Sequence
**File:** [`diagrams/08_data_flow_sequence.mmd`](./diagrams/08_data_flow_sequence.mmd)

Step-by-step sequence diagram showing temporal flow of data through the model, parallel processing in prediction heads, and loop structures in transformer blocks.

---

### 9. Multi-Task Learning
**File:** [`diagrams/09_multitask_learning.mmd`](./diagrams/09_multitask_learning.mmd)

Simultaneous gene expression and cell type prediction:
- **Shared Backbone**: ViT + GAT feature extraction
- **Task 1**: Gene expression (ZINB loss)
- **Task 2**: Cell type classification (Cross-Entropy loss)
- **Combined Loss**: L_total = L_ZINB + α×L_CE

---

### 10. Bulk vs Single-Cell Comparison
**File:** [`diagrams/10_bulk_vs_singlecell.mmd`](./diagrams/10_bulk_vs_singlecell.mmd)

Architecture differences between hist2RNA (bulk) and hist2scRNA (single-cell):
- CNN vs Vision Transformer
- No spatial model vs Graph Attention Network
- MSE loss vs ZINB loss
- Single-task vs Multi-task learning

---

### 11. Model Scalability
**File:** [`diagrams/11_model_scalability.mmd`](./diagrams/11_model_scalability.mmd)

Different model configurations (Small/Medium/Large/Lightweight) with parameter counts, computational costs, and recommended use cases.

---

### 12. Input Preprocessing Workflow
**File:** [`diagrams/12_input_preprocessing_workflow.mmd`](./diagrams/12_input_preprocessing_workflow.mmd)

Complete preprocessing pipeline from raw data to model-ready input:
- **Load Data**: WSI, spot coordinates, gene expression
- **Coordinate Alignment**: Visium space → WSI space transformation
- **Quality Control**: Filter in-tissue spots, check coverage
- **Patch Extraction**: Extract 224×224 patches at spot locations
- **Spatial Graph**: Build k-NN graph connecting neighbors
- **Output**: Organized dataset ready for training

See [INPUT_IMAGE_WORKFLOW.md](./INPUT_IMAGE_WORKFLOW.md) for detailed documentation.

---

### 13. Patch Extraction Process
**File:** [`diagrams/13_patch_extraction_process.mmd`](./diagrams/13_patch_extraction_process.mmd)

Detailed patch extraction from Whole Slide Images:
- Calculate patch boundaries (center ± 112 pixels)
- Extract region using OpenSlide
- Convert to RGB and check quality
- Apply color normalization (Macenko)
- Save as 224×224 PNG files

**Key Point**: Unlike bulk RNA (random patches), patches are **centered at specific spot coordinates**.

---

### 14. Spatial Graph Construction
**File:** [`diagrams/14_spatial_graph_construction.mmd`](./diagrams/14_spatial_graph_construction.mmd)

Building the spatial graph for Graph Attention Networks:
- Use KDTree for efficient neighbor search
- Find k=6 nearest neighbors (Visium hexagonal grid)
- Create edge list connecting spots
- Alternative: distance threshold method
- Graph properties: ~6× edges as nodes, sparse connectivity

**For Visium**: Each spot has 6 neighbors in hexagonal arrangement (100 μm center-to-center spacing).

---

### 15. Coordinate Alignment
**File:** [`diagrams/15_coordinate_alignment.mmd`](./diagrams/15_coordinate_alignment.mmd)

Aligning multiple coordinate systems:
- **Visium array space** → **Visium image space** → **WSI pixel space**
- Apply scale factors from `scalefactors_json.json`
- Handle coordinate transformations (flip, rotate)
- Manual registration using fiducial markers if needed
- Validation: visual inspection of alignment

**Common Issues**: Y-axis flipping, rotation, scale factor mismatches.

---

## 🎨 Design Rationale

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Encoder** | Vision Transformer | Captures long-range tissue architecture better than CNNs |
| **Spatial Model** | Graph Attention Network | Learns importance of different neighbors, handles irregular grids |
| **Loss Function** | Zero-Inflated Negative Binomial | Properly models 70-90% zeros and overdispersion in scRNA-seq |
| **Multi-task** | Gene Expression + Cell Type | Biologically related tasks, shared representations improve both |

---

## 🔗 Resources

- **Main Documentation**: [SCRNA_README.md](./SCRNA_README.md)
- **Quick Start Guide**: [QUICKSTART_SCRNA.md](./QUICKSTART_SCRNA.md)
- **Diagram Index**: [diagrams/README.md](./diagrams/README.md)
- **Mermaid Documentation**: [mermaid.js.org](https://mermaid.js.org/)

---

*All diagrams are rendered natively by GitHub. Click any `.mmd` file to view directly.*
