# hist2scRNA Architecture Diagrams

This document provides visual diagrams of the hist2scRNA model architecture, data flow, and training process.

## Table of Contents
- [Overall Architecture](#overall-architecture)
- [Data Structure and Flow](#data-structure-and-flow)
- [Training Pipeline](#training-pipeline)
- [Model Components](#model-components)
- [Inference Pipeline](#inference-pipeline)

---

## Overall Architecture

```mermaid
graph TB
    subgraph Input
        A[Histopathology Image<br/>224×224×3] --> B[Patch Embedding<br/>16×16 patches]
        C[Spatial Coordinates<br/>x, y] --> D[Spatial Graph<br/>k-NN edges]
    end

    subgraph "Vision Transformer Encoder"
        B --> E[Patch Tokens<br/>196×embed_dim]
        E --> F[Add Positional Embedding]
        F --> G[Multi-Head Self-Attention]
        G --> H[Feed-Forward Network]
        H --> I[Transformer Blocks<br/>×6-12 layers]
        I --> J[Class Token<br/>Global Image Features]
    end

    subgraph "Spatial Graph Attention"
        J --> K[Spot Features<br/>embed_dim]
        D --> L[Graph Edges]
        K --> M[Graph Attention Layer 1<br/>Multi-head GAT]
        L --> M
        M --> N[Graph Attention Layer 2<br/>Single-head GAT]
        L --> N
        N --> O[Spatially-Aware Features<br/>gnn_hidden_dim]
    end

    subgraph "Gene Expression Decoder"
        O --> P[Dense Layer 1<br/>1024 units]
        P --> Q[Dense Layer 2<br/>2048 units]
        Q --> R[ZINB Parameter Heads]

        R --> S1[μ decoder<br/>Mean Expression]
        R --> S2[θ decoder<br/>Dispersion]
        R --> S3[π decoder<br/>Zero-inflation]

        S1 --> T1[μ: n_genes<br/>exp activation]
        S2 --> T2[θ: n_genes<br/>exp activation]
        S3 --> T3[π: n_genes<br/>sigmoid activation]
    end

    subgraph "Cell Type Prediction"
        Q --> U[Cell Type Head<br/>n_cell_types]
        U --> V[Cell Type Logits<br/>softmax]
    end

    subgraph Output
        T1 --> W1[Gene Expression<br/>Mean per gene]
        T2 --> W2[Expression Variance<br/>Dispersion per gene]
        T3 --> W3[Dropout Probability<br/>Zero-inflation per gene]
        V --> W4[Cell Type<br/>Predicted class]
    end

    style A fill:#e1f5ff
    style J fill:#fff4e1
    style O fill:#ffe1f5
    style W1 fill:#e1ffe1
    style W2 fill:#e1ffe1
    style W3 fill:#e1ffe1
    style W4 fill:#ffe1e1
```

---

## Data Structure and Flow

```mermaid
graph LR
    subgraph "Raw Data"
        A1[WSI<br/>Whole Slide Image] --> A2[Image Patches<br/>224×224 pixels]
        A3[Spatial<br/>Transcriptomics<br/>10X Visium] --> A4[Spot Coordinates<br/>x, y positions]
        A3 --> A5[Gene Expression<br/>Count Matrix]
        A3 --> A6[Cell Type Labels<br/>Annotations]
    end

    subgraph "Preprocessing"
        A2 --> B1[Color Normalization<br/>Macenko]
        B1 --> B2[H&E Normalized<br/>Patches]

        A4 --> C1[k-NN Graph<br/>Construction]
        C1 --> C2[Spatial Edges<br/>Adjacency List]

        A5 --> D1[log2 1+x<br/>Transform]
        D1 --> D2[Normalized<br/>Expression]
    end

    subgraph "Dataset Structure"
        B2 --> E1[patches/<br/>spot_0000.png<br/>spot_0001.png<br/>...]
        C2 --> E2[spatial_edges.csv<br/>source, target]
        A4 --> E3[spatial_coordinates.csv<br/>spot_id, x, y]
        D2 --> E4[gene_expression.csv<br/>spots × genes]
        A6 --> E5[cell_types.csv<br/>spot_id, cell_type]
    end

    subgraph "Model Input"
        E1 --> F1[Image Tensor<br/>batch×3×224×224]
        E2 --> F2[Edge Index<br/>2×n_edges]
        E3 --> F3[Coordinates<br/>batch×2]
        E4 --> F4[Target Expression<br/>batch×n_genes]
        E5 --> F5[Target Cell Type<br/>batch]
    end

    style A1 fill:#ffe1e1
    style A3 fill:#ffe1e1
    style E1 fill:#e1f5ff
    style E2 fill:#e1f5ff
    style E3 fill:#e1f5ff
    style E4 fill:#e1f5ff
    style E5 fill:#e1f5ff
```

---

## Training Pipeline

```mermaid
flowchart TD
    Start([Start Training]) --> Init[Initialize Model<br/>ViT + GAT + Decoder]
    Init --> LoadData[Load Dataset<br/>Images + Expression + Graph]

    LoadData --> Split{Split Data}
    Split --> Train[Training Set<br/>70%]
    Split --> Val[Validation Set<br/>15%]
    Split --> Test[Test Set<br/>15%]

    Train --> Epoch{For Each Epoch}

    Epoch --> TrainLoop[Training Loop]

    subgraph "Training Loop"
        TrainLoop --> Batch1[Get Batch<br/>images, targets, edges]
        Batch1 --> Forward1[Forward Pass<br/>ViT → GAT → Decoder]
        Forward1 --> Loss1[Compute Losses]

        subgraph "Loss Computation"
            Loss1 --> ZINB[ZINB Loss<br/>Gene Expression]
            Loss1 --> CE[Cross-Entropy<br/>Cell Type]
            ZINB --> Combined[Combined Loss<br/>L = L_ZINB + α×L_CE]
            CE --> Combined
        end

        Combined --> Backward[Backward Pass<br/>Compute Gradients]
        Backward --> Update[Update Parameters<br/>AdamW Optimizer]
    end

    Update --> ValLoop[Validation Loop]

    subgraph "Validation Loop"
        ValLoop --> Batch2[Get Batch<br/>No Gradient]
        Batch2 --> Forward2[Forward Pass]
        Forward2 --> Metrics[Compute Metrics]

        subgraph "Metrics"
            Metrics --> M1[Val Loss]
            Metrics --> M2[Spearman Corr]
            Metrics --> M3[Cell Type Acc]
        end
    end

    M1 --> CheckImprove{Val Loss<br/>Improved?}
    CheckImprove -->|Yes| Save[Save Best Model<br/>Checkpoint]
    CheckImprove -->|No| Counter[Increment Counter]

    Save --> LR[Update LR<br/>ReduceLROnPlateau]
    Counter --> LR

    LR --> EarlyStop{Early<br/>Stopping?}
    EarlyStop -->|No| Epoch
    EarlyStop -->|Yes| LoadBest[Load Best Model]

    Epoch -->|All Epochs Done| LoadBest

    LoadBest --> TestLoop[Test Evaluation]

    subgraph "Test Evaluation"
        TestLoop --> TestMetrics[Compute Test Metrics]
        TestMetrics --> TM1[MSE, MAE]
        TestMetrics --> TM2[Per-Spot Correlation]
        TestMetrics --> TM3[Per-Gene Correlation]
        TestMetrics --> TM4[Cell Type Accuracy]
    end

    TM4 --> SaveResults[Save Results<br/>results.json]
    SaveResults --> PlotCurves[Plot Training Curves<br/>Loss vs Epoch]
    PlotCurves --> End([Training Complete])

    style Start fill:#e1ffe1
    style ZINB fill:#ffe1e1
    style CE fill:#ffe1e1
    style Save fill:#e1f5ff
    style End fill:#e1ffe1
```

---

## Model Components Deep Dive

### Vision Transformer Block

```mermaid
graph TB
    subgraph "Single Transformer Block"
        A[Input Features<br/>batch×n_tokens×embed_dim] --> B[Layer Norm 1]
        B --> C[Multi-Head Self-Attention]

        subgraph "Multi-Head Self-Attention"
            C --> D[Linear: Q, K, V<br/>embed_dim → 3×embed_dim]
            D --> E[Split into<br/>num_heads heads]
            E --> F[Scaled Dot-Product<br/>Attention per head]
            F --> G[Concat heads]
            G --> H[Linear Projection<br/>embed_dim → embed_dim]
        end

        H --> I[Dropout]
        I --> J[Residual Add]
        A --> J

        J --> K[Layer Norm 2]
        K --> L[Feed-Forward Network]

        subgraph "Feed-Forward Network"
            L --> M[Linear 1<br/>embed_dim → 4×embed_dim]
            M --> N[GELU Activation]
            N --> O[Dropout]
            O --> P[Linear 2<br/>4×embed_dim → embed_dim]
            P --> Q[Dropout]
        end

        Q --> R[Residual Add]
        J --> R
        R --> S[Output Features<br/>batch×n_tokens×embed_dim]
    end

    style C fill:#e1f5ff
    style L fill:#ffe1f5
```

### Graph Attention Network

```mermaid
graph TB
    subgraph "Graph Attention Layer"
        A[Node Features<br/>n_nodes×in_channels] --> B[Linear Transform<br/>W·h]
        C[Edge Index<br/>2×n_edges] --> D[Gather Neighbors]

        B --> E[Compute Attention<br/>for each edge]
        D --> E

        subgraph "Attention Mechanism"
            E --> F[Attention Scores<br/>α_ij = a W·h_i, W·h_j]
            F --> G[LeakyReLU]
            G --> H[Softmax per node<br/>Normalize neighbors]
        end

        H --> I[Weighted Sum<br/>h'_i = Σ α_ij·W·h_j]
        I --> J[Multi-Head Concat<br/>or Average]
        J --> K[ELU Activation]
        K --> L[Output Features<br/>n_nodes×out_channels]
    end

    style E fill:#ffe1f5
    style I fill:#e1f5ff
```

### ZINB Distribution and Loss

```mermaid
graph TB
    subgraph "Zero-Inflated Negative Binomial"
        A[Decoder Output<br/>2048 features] --> B{Three Heads}

        B --> C1[μ Head<br/>Linear → Exp]
        B --> C2[θ Head<br/>Linear → Exp]
        B --> C3[π Head<br/>Linear → Sigmoid]

        C1 --> D1[μ ∈ ℝ+<br/>Mean Expression]
        C2 --> D2[θ ∈ ℝ+<br/>Dispersion]
        C3 --> D3[π ∈ 0,1<br/>Zero Prob]
    end

    subgraph "ZINB Loss Computation"
        D1 --> E[Target Expression y]
        D2 --> E
        D3 --> E

        E --> F{y = 0?}

        F -->|Yes| G[Zero Case<br/>log π + 1-π·NB0]
        F -->|No| H[Non-Zero Case<br/>log 1-π + log NBμ,θy]

        subgraph "Negative Binomial"
            H --> I[log Γθ+y]
            I --> J[- log Γθ]
            J --> K[- log Γy+1]
            K --> L[+ θ·log θ]
            L --> M[- θ+y·log θ+μ]
        end

        G --> N[Mean Loss<br/>across batch & genes]
        M --> N
    end

    style D1 fill:#e1ffe1
    style D2 fill:#e1ffe1
    style D3 fill:#e1ffe1
    style N fill:#ffe1e1
```

---

## Inference Pipeline

```mermaid
flowchart LR
    subgraph "Input Preparation"
        A1[New Histopathology<br/>Image] --> A2[Resize & Normalize<br/>224×224]
        A3[Spatial Locations<br/>Optional] --> A4[Build k-NN Graph<br/>if multiple spots]
    end

    subgraph "Feature Extraction"
        A2 --> B1[Vision Transformer<br/>Extract Features]
        B1 --> B2[Class Token<br/>or Patch Tokens]
    end

    subgraph "Spatial Context"
        B2 --> C1[Graph Attention<br/>if graph available]
        A4 --> C1
        C1 --> C2[Spatially-Aware<br/>Features]

        B2 -.->|No Graph| C2
    end

    subgraph "Prediction"
        C2 --> D1[Gene Expression<br/>Decoder]
        D1 --> D2[ZINB Parameters<br/>μ, θ, π]

        C2 --> E1[Cell Type<br/>Classifier]
        E1 --> E2[Cell Type<br/>Probabilities]
    end

    subgraph "Output"
        D2 --> F1[Predicted Expression<br/>Use μ mean]
        D2 --> F2[Expression Uncertainty<br/>Use θ variance]
        D2 --> F3[Dropout Probability<br/>Use π]
        E2 --> F4[Cell Type<br/>argmax]
    end

    subgraph "Visualization"
        F1 --> G1[Heatmap<br/>Gene Expression]
        F4 --> G2[Spatial Map<br/>Cell Types]
        F2 --> G3[Uncertainty Map<br/>Confidence]
    end

    style A1 fill:#e1f5ff
    style F1 fill:#e1ffe1
    style F4 fill:#ffe1e1
    style G1 fill:#fff4e1
```

---

## Data Flow Through Full Model

```mermaid
sequenceDiagram
    participant I as Input Image
    participant PE as Patch Embedding
    participant ViT as Vision Transformer
    participant GAT as Graph Attention
    participant Dec as Decoder
    participant ZINB as ZINB Heads
    participant CT as Cell Type Head
    participant Out as Output

    I->>PE: 224×224×3 image
    PE->>PE: Split into 196 patches
    PE->>ViT: 196×embed_dim tokens

    loop 6-12 Transformer Blocks
        ViT->>ViT: Multi-Head Self-Attention
        ViT->>ViT: Feed-Forward Network
    end

    ViT->>GAT: Class token (global features)

    Note over GAT: Spatial Graph Processing
    GAT->>GAT: Graph Attention Layer 1<br/>Multi-head aggregation
    GAT->>GAT: Graph Attention Layer 2<br/>Single-head output

    GAT->>Dec: Spatially-aware features
    Dec->>Dec: Dense Layer 1 (1024)
    Dec->>Dec: Dense Layer 2 (2048)

    par Parallel Heads
        Dec->>ZINB: μ decoder (Mean)
        Dec->>ZINB: θ decoder (Dispersion)
        Dec->>ZINB: π decoder (Zero-inflation)
        Dec->>CT: Cell Type classifier
    end

    ZINB->>Out: Gene expression (n_genes)
    CT->>Out: Cell type (n_cell_types)

    Note over Out: Loss Computation<br/>ZINB Loss + CE Loss
```

---

## Multi-Task Learning Architecture

```mermaid
graph TB
    subgraph "Shared Backbone"
        A[H&E Image] --> B[Vision Transformer<br/>Feature Extractor]
        B --> C[Graph Attention<br/>Spatial Context]
        C --> D[Shared Decoder<br/>2048 features]
    end

    subgraph "Task 1: Gene Expression"
        D --> E1[ZINB μ Head<br/>Mean expression]
        D --> E2[ZINB θ Head<br/>Dispersion]
        D --> E3[ZINB π Head<br/>Zero-inflation]

        E1 --> F1[Predicted μ<br/>n_genes]
        E2 --> F2[Predicted θ<br/>n_genes]
        E3 --> F3[Predicted π<br/>n_genes]

        F1 --> G1[ZINB Loss]
        F2 --> G1
        F3 --> G1
    end

    subgraph "Task 2: Cell Type"
        D --> H1[Classification Head<br/>n_cell_types]
        H1 --> H2[Cell Type Logits]
        H2 --> I1[Cross-Entropy Loss]
    end

    subgraph "Combined Loss"
        G1 --> J[L_total = L_ZINB + α×L_CE]
        I1 --> J
        J --> K[Backpropagate<br/>Update Shared Weights]
    end

    K --> L{Benefits}
    L --> M1[✓ Shared representations]
    L --> M2[✓ Regularization effect]
    L --> M3[✓ Better generalization]
    L --> M4[✓ Biological consistency]

    style D fill:#e1f5ff
    style G1 fill:#ffe1e1
    style I1 fill:#ffe1e1
    style J fill:#fff4e1
```

---

## Comparison: Bulk vs Single-Cell Models

```mermaid
graph TB
    subgraph "hist2RNA Bulk"
        A1[Image Patches] --> A2[CNN ResNet50]
        A2 --> A3[Average Pooling<br/>Aggregate patches]
        A3 --> A4[1D Convolutions]
        A4 --> A5[Global Average Pool]
        A5 --> A6[Linear Decoder<br/>n_genes outputs]
        A6 --> A7[MSE Loss]
    end

    subgraph "hist2scRNA Single-Cell"
        B1[Image Patches] --> B2[Vision Transformer<br/>Self-Attention]
        B2 --> B3[Spatial Graph<br/>GAT layers]
        B3 --> B4[Multi-Layer Decoder<br/>1024→2048]
        B4 --> B5[Three ZINB Heads<br/>μ, θ, π]
        B5 --> B6[ZINB Loss<br/>+ Cell Type Loss]
    end

    A1 -.Comparison.-> B1
    A2 -.CNN vs ViT.-> B2
    A3 -.No Graph.-> B3
    A4 -.Simple.-> B4
    A6 -.Single Head.-> B5
    A7 -.MSE vs ZINB.-> B6

    style A1 fill:#ffe1e1
    style A7 fill:#ffe1e1
    style B1 fill:#e1ffe1
    style B6 fill:#e1ffe1
```

---

## Model Scalability and Variants

```mermaid
graph LR
    subgraph "Input Variants"
        I1[Raw Images<br/>224×224] --> M1[Full Model]
        I2[Pre-extracted<br/>Features] --> M2[Lightweight Model]
    end

    subgraph "Model Configurations"
        M1 --> C1[Small<br/>embed_dim=256<br/>depth=4]
        M1 --> C2[Medium<br/>embed_dim=384<br/>depth=6]
        M1 --> C3[Large<br/>embed_dim=768<br/>depth=12]

        M2 --> C4[Lightweight<br/>Feature Input<br/>No ViT]
    end

    subgraph "Computational Cost"
        C1 --> P1[~10M params<br/>Fast training]
        C2 --> P2[~50M params<br/>Balanced]
        C3 --> P3[~300M params<br/>Best accuracy]
        C4 --> P4[~15M params<br/>Fastest]
    end

    subgraph "Use Cases"
        P1 --> U1[Development<br/>Quick iterations]
        P2 --> U2[Research<br/>SOTA performance]
        P3 --> U3[Production<br/>Maximum accuracy]
        P4 --> U4[Edge Devices<br/>Real-time inference]
    end

    style C2 fill:#e1ffe1
    style P2 fill:#e1f5ff
```

---

## Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Encoder** | Vision Transformer | Captures long-range tissue architecture better than CNNs |
| **Spatial Model** | Graph Attention Network | Learns importance of different neighbors, handles irregular grids |
| **Loss Function** | Zero-Inflated Negative Binomial | Properly models sparsity (70-90% zeros) and overdispersion in scRNA-seq |
| **Multi-task** | Gene Expression + Cell Type | Shared representations improve both tasks, biological consistency |
| **Activation** | GELU in ViT, ELU in GAT | Smooth gradients, better than ReLU for transformers |
| **Normalization** | LayerNorm | Works better with attention mechanisms than BatchNorm |
| **Pooling** | Class Token | Learnable global representation, better than average pooling |

---

*These diagrams provide a comprehensive visual guide to understanding the hist2scRNA architecture and workflow.*
