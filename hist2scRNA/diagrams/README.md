# hist2scRNA Architecture Diagrams

This folder contains Mermaid diagrams that visualize the hist2scRNA model architecture, data flow, and training process.

## 📊 Diagrams Index

### Core Architecture

1. **[Overall Architecture](./01_overall_architecture.mmd)** 🎯
   - Complete model flow: Image → ViT → GAT → Decoder → ZINB outputs
   - All components with color-coded sections
   - Shows Vision Transformer, Graph Attention, and ZINB parameter heads

2. **[Data Structure and Flow](./02_data_structure_flow.mmd)** 📦
   - Raw data preprocessing pipeline
   - Dataset organization (patches, coordinates, edges, expression)
   - Input tensor formats for training

### Training & Inference

3. **[Training Pipeline](./03_training_pipeline.mmd)** 🔄
   - Complete training loop flowchart
   - Validation, early stopping, and checkpointing
   - Loss computation and metric evaluation

7. **[Inference Pipeline](./07_inference_pipeline.mmd)** 🚀
   - Production deployment workflow
   - Feature extraction and prediction steps
   - Visualization generation

8. **[Data Flow Sequence](./08_data_flow_sequence.mmd)** 📊
   - Step-by-step sequence diagram
   - Component communication flow
   - Parallel head processing

### Model Components

4. **[Vision Transformer Block](./04_vision_transformer_block.mmd)** 🔬
   - Multi-head self-attention mechanism internals
   - Feed-forward network details
   - Residual connections and layer normalization

5. **[Graph Attention Network](./05_graph_attention_network.mmd)** 🕸️
   - Message passing mechanism
   - Attention score computation
   - Neighbor aggregation

6. **[ZINB Distribution](./06_zinb_distribution.mmd)** 📐
   - Three-head decoder (μ, θ, π)
   - Zero-inflated and non-zero cases
   - Mathematical loss computation

### Input Preprocessing

12. **[Input Preprocessing Workflow](./12_input_preprocessing_workflow.mmd)** 🔧
    - Complete preprocessing pipeline from WSI to model-ready data
    - Coordinate alignment and quality control
    - Patch extraction and spatial graph construction

13. **[Patch Extraction Process](./13_patch_extraction_process.mmd)** ✂️
    - Detailed patch extraction from Whole Slide Images
    - Boundary calculation and region extraction
    - Quality control and color normalization

14. **[Spatial Graph Construction](./14_spatial_graph_construction.mmd)** 🕸️
    - k-NN neighbor search using KDTree
    - Edge list creation for Graph Attention Networks
    - Visium hexagonal grid structure

15. **[Coordinate Alignment](./15_coordinate_alignment.mmd)** 📍
    - Coordinate system transformations (Visium → WSI)
    - Scale factor application and manual registration
    - Alignment validation and troubleshooting

### Evaluation & Performance Metrics

16. **[Evaluation Metrics Hierarchy](./16_evaluation_metrics_hierarchy.mmd)** 📊
    - Complete 6-category evaluation framework
    - Overall, gene-wise, spot-wise, zero-inflation, cell type, expression level metrics
    - Interpretation guidelines for each metric type
    - Automatic visualization outputs

17. **[Evaluation Process Flow](./17_evaluation_process_flow.mmd)** 🔄
    - Step-by-step evaluation pipeline
    - Inference loop without gradients
    - Sequential metrics computation (6 categories)
    - JSON export and plot generation

18. **[Metrics Interpretation Guide](./18_metrics_interpretation_guide.mmd)** 📖
    - Research questions → Metrics mapping
    - Performance benchmarks (Excellent/Good/Fair/Poor)
    - Interpretation → Actions for improvement
    - Trade-off analysis (e.g., sensitivity vs specificity)

19. **[Metrics to Model Components](./19_metrics_to_model_components.mmd)** 🔍
    - Maps metrics to specific model components for debugging
    - Component diagnostics (ViT, GAT, Decoder, ZINB heads)
    - Diagnostic workflow decision tree
    - Troubleshooting based on metric patterns

### Advanced Concepts

9. **[Multi-Task Learning](./09_multitask_learning.mmd)** 🎯
   - Shared backbone architecture
   - Gene expression + Cell type prediction
   - Benefits and regularization effects

10. **[Bulk vs Single-Cell Comparison](./10_bulk_vs_singlecell.mmd)** ⚖️
    - hist2RNA (bulk) vs hist2scRNA (single-cell)
    - Architecture differences highlighted
    - CNN vs ViT, MSE vs ZINB

11. **[Model Scalability](./11_model_scalability.mmd)** 📈
    - Different model configurations (Small/Medium/Large)
    - Parameter counts and computational costs
    - Use case recommendations

## 🎨 Viewing the Diagrams

### On GitHub
Click on any `.mmd` file above to view it directly on GitHub. GitHub natively renders Mermaid diagrams.

### In Your Editor
If using VS Code, install the "Markdown Preview Mermaid Support" extension to view diagrams in preview mode.

### Online
Copy the content of any `.mmd` file and paste it into the [Mermaid Live Editor](https://mermaid.live/) for interactive viewing and editing.

## 🔑 Color Coding

- 🔵 **Blue** (`#e1f5ff`): Input/Data layers
- 🟡 **Yellow** (`#fff4e1`): Processing/Intermediate layers
- 🟣 **Pink** (`#ffe1f5`): Attention/Graph layers
- 🟢 **Green** (`#e1ffe1`): Output/Results
- 🔴 **Red** (`#ffe1e1`): Loss/Error components

## 📖 Usage in Documentation

These diagrams are referenced in:
- Main [README.md](../README.md)
- [SCRNA_README.md](../SCRNA_README.md) - Full documentation
- [QUICKSTART_SCRNA.md](../QUICKSTART_SCRNA.md) - Quick start guide
- [ARCHITECTURE_DIAGRAMS.md](../ARCHITECTURE_DIAGRAMS.md) - Detailed explanations

## 🛠️ Editing Diagrams

To modify a diagram:
1. Edit the `.mmd` file directly
2. Test in [Mermaid Live Editor](https://mermaid.live/)
3. Commit changes to repository
4. GitHub will automatically render the updated diagram

## 📚 Learn More

- [Mermaid Documentation](https://mermaid.js.org/)
- [Mermaid Flowchart Syntax](https://mermaid.js.org/syntax/flowchart.html)
- [Mermaid Sequence Diagram Syntax](https://mermaid.js.org/syntax/sequenceDiagram.html)
