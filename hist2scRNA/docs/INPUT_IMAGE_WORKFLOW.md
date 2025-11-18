# Input Image Workflow for hist2scRNA

This document explains how to prepare histopathology images for the hist2scRNA model, including patch extraction from Whole Slide Images (WSI) and spatial graph construction.

## 📊 Visual Workflow Diagrams

See the diagrams in the [`diagrams/`](./diagrams/) folder:
- **[12_input_preprocessing_workflow.mmd](./diagrams/12_input_preprocessing_workflow.mmd)** - Complete preprocessing pipeline
- **[13_patch_extraction_process.mmd](./diagrams/13_patch_extraction_process.mmd)** - Detailed patch extraction
- **[14_spatial_graph_construction.mmd](./diagrams/14_spatial_graph_construction.mmd)** - Graph building process

---

## Table of Contents
1. [Overview](#overview)
2. [Complete Example: 5-Spot Dataset](#complete-example-5-spot-dataset)
3. [Input Data Requirements](#input-data-requirements)
4. [Workflow Steps](#workflow-steps)
5. [Patch Extraction](#patch-extraction)
6. [Spatial Graph Construction](#spatial-graph-construction)
7. [Coordinate Alignment](#coordinate-alignment)
8. [Tools and Software](#tools-and-software)
9. [Common Issues](#common-issues)

---

## Overview

Unlike bulk RNA prediction (which uses random patches), **hist2scRNA requires spatially-aligned patches** that correspond to specific spots where RNA was sequenced.

### Key Concept

```
Whole Slide Image (WSI)  +  Spatial Transcriptomics Data
        ↓                              ↓
   Extract patches at              Spot coordinates
   spot locations                  & gene expression
        ↓                              ↓
        └──────────────┬───────────────┘
                       ↓
              hist2scRNA Model
                       ↓
        Gene expression per spot
```

---

## Complete Example: 5-Spot Dataset

Let's walk through a **complete concrete example** with 5 spots to understand how all the data files connect.

### Scenario

You have a breast cancer tissue sample analyzed with **10X Visium**:
- **Patient ID**: patient_001
- **Number of spots**: 5 (simplified for clarity; real Visium has 2000-5000 spots)
- **Number of genes**: 10 (simplified; real data has 2000-20000 genes)
- **Cell types**: 3 types (0=Cancer cells, 1=Fibroblasts, 2=Immune cells)

### Visual Spatial Layout

```
Coordinate System: WSI pixel space
Y-axis ↑

3550  │        spot_0004 (Fibroblast)
      │            ●
3500  │            │
      │            │
3450  │    ●───────●───────●
      │ spot_0001  spot_0002  spot_0003
      │ (Cancer)   (Immune)   (Cancer)
3400  │
      │
      └─────────────────────────────────→ X-axis
           1200    1250    1300
```

### Complete Dataset Files

#### 1️⃣ **patches/ folder** (Histopathology Images)

```
patches/
├── spot_0001.png    # 224×224 RGB image centered at (1200, 3450)
├── spot_0002.png    # 224×224 RGB image centered at (1250, 3450)
├── spot_0003.png    # 224×224 RGB image centered at (1300, 3450)
├── spot_0004.png    # 224×224 RGB image centered at (1250, 3500)
└── spot_0005.png    # 224×224 RGB image centered at (1250, 3550)
```

**Each patch:**
- **Size**: 224 × 224 × 3 (RGB)
- **File size**: ~50-200 KB
- **Centered** at the spot coordinate in the WSI
- **Extracted** from larger Whole Slide Image (e.g., 75,000 × 62,000 pixels)

---

#### 2️⃣ **spatial_coordinates.csv** (Where are the spots?)

```csv
spot_id,x,y
spot_0001,1200,3450
spot_0002,1250,3450
spot_0003,1300,3450
spot_0004,1250,3500
spot_0005,1250,3550
```

**Explanation:**
- `x, y`: Pixel coordinates in the Whole Slide Image where patch was extracted
- These coordinates were obtained from 10X Visium `tissue_positions_list.csv`
- The patch for `spot_0001` is extracted from WSI region: `[1200-112:1200+112, 3450-112:3450+112]`

---

#### 3️⃣ **gene_expression.csv** (What genes are expressed?)

```csv
spot_id,BRCA1,TP53,ESR1,ERBB2,MKI67,CD8A,CD4,ACTA2,VIM,COL1A1
spot_0001,12.3,8.5,15.2,0.0,22.1,0.0,0.3,1.2,0.5,0.8
spot_0002,0.5,1.2,0.0,0.0,1.3,45.6,38.2,0.0,2.1,0.3
spot_0003,15.8,9.2,18.5,0.0,25.3,0.0,0.5,0.9,0.0,1.1
spot_0004,2.1,3.5,0.0,0.0,5.2,0.0,0.0,52.3,48.7,65.4
spot_0005,1.8,2.9,0.0,0.0,4.8,0.0,0.0,48.9,51.2,62.8
```

**Explanation:**
- Each row = one spot
- Each column (except spot_id) = one gene
- Values = normalized gene expression (e.g., log2(counts+1))
- **Sparsity**: Notice many zeros (~70-80% in real single-cell data)

**Gene interpretation:**
- `spot_0001` (Cancer): High BRCA1, TP53, ESR1, MKI67 (cancer markers)
- `spot_0002` (Immune): High CD8A, CD4 (T-cell markers)
- `spot_0004` (Fibroblast): High ACTA2, VIM, COL1A1 (stromal markers)

---

#### 4️⃣ **cell_types.csv** (What cell type is each spot?)

```csv
spot_id,cell_type
spot_0001,0
spot_0002,2
spot_0003,0
spot_0004,1
spot_0005,1
```

**Explanation:**
- Cell type labels (integer encoding)
- **0** = Cancer cells (epithelial)
- **1** = Fibroblasts (stromal)
- **2** = Immune cells (lymphocytes)

**This is optional** but helps with:
- Multi-task learning (model predicts both gene expression AND cell type)
- Validation (check if predicted expression matches expected cell type)
- Biological interpretation

---

#### 5️⃣ **spatial_edges.csv** (Which spots are neighbors?)

```csv
source,target
0,1
0,3
1,0
1,2
1,3
2,1
2,3
3,0
3,1
3,2
3,4
4,3
```

**Explanation:**
- **Indices** (not spot_ids): 0=spot_0001, 1=spot_0002, 2=spot_0003, 3=spot_0004, 4=spot_0005
- **Each row** = one edge in the spatial graph
- **Undirected graph**: Both (0,1) and (1,0) included for bidirectional message passing

**Visual representation:**
```
    0 (spot_0001) ←→ 1 (spot_0002) ←→ 2 (spot_0003)
         ↕                ↕                ↕
         └────────→ 3 (spot_0004) ←──────┘
                         ↕
                    4 (spot_0005)
```

**How edges were created (k-NN, k=3 for this example):**

For `spot_0001` (index 0):
```
Distances to other spots:
  - spot_0002 (index 1): 50 pixels   ✓ (1st neighbor)
  - spot_0003 (index 2): 100 pixels  ✗ (4th neighbor, beyond k=3)
  - spot_0004 (index 3): 52 pixels   ✓ (2nd neighbor)
  - spot_0005 (index 4): 102 pixels  ✗ (beyond k=3)

Created edges: (0,1) and (0,3)
```

For `spot_0002` (index 1):
```
Distances:
  - spot_0001: 50 pixels   ✓
  - spot_0003: 50 pixels   ✓
  - spot_0004: 52 pixels   ✓
  - spot_0005: 102 pixels  ✗

Created edges: (1,0), (1,2), (1,3)
```

...and so on for all spots.

---

### How the Model Uses This Data

#### Step 1: Data Loading

```python
# Load spot_0001
image = load_image('patches/spot_0001.png')        # 224×224×3 tensor
coords = get_coordinates('spot_0001')              # [1200, 3450]
expression = get_expression('spot_0001')           # [12.3, 8.5, 15.2, ...]
cell_type = get_cell_type('spot_0001')             # 0 (Cancer)
neighbors = get_neighbors(index=0)                 # [1, 3] from edges
```

#### Step 2: Vision Transformer (ViT)

```python
# Process image through ViT
image_features = ViT(image)
# Output: 768-dimensional feature vector
# Captures: tissue morphology, cell density, nuclear features, etc.
```

#### Step 3: Graph Attention Network (GAT)

```python
# Get neighbor features
neighbor_features = [
    ViT(load_image('patches/spot_0002.png')),  # neighbor 1
    ViT(load_image('patches/spot_0004.png'))   # neighbor 3
]

# Aggregate with attention
attention_weights = compute_attention(image_features, neighbor_features)
# e.g., [0.6, 0.4] - weight neighbor 1 more than neighbor 3

aggregated_features = (
    0.6 * neighbor_features[0] +
    0.4 * neighbor_features[1]
)

# Combine own and neighbor information
final_features = image_features + aggregated_features
```

#### Step 4: Prediction

```python
# Predict ZINB parameters
mu, theta, pi = ZINB_decoder(final_features)
# mu: [12.1, 8.7, 14.9, 0.1, 21.8, ...] - predicted mean expression
# theta: [2.5, 1.8, 3.2, ...] - dispersion
# pi: [0.1, 0.15, 0.05, 0.9, ...] - zero probability

# Predict cell type
cell_type_logits = CellType_head(final_features)
predicted_cell_type = argmax(cell_type_logits)  # 0 (Cancer) ✓
```

#### Step 5: Loss Calculation

```python
# Gene expression loss
zinb_loss = ZINB_loss(
    mu=[12.1, 8.7, 14.9, ...],
    theta=[2.5, 1.8, 3.2, ...],
    pi=[0.1, 0.15, 0.05, ...],
    target=[12.3, 8.5, 15.2, ...]  # ground truth
)

# Cell type loss
ce_loss = CrossEntropy_loss(
    predicted=cell_type_logits,
    target=0  # ground truth: Cancer
)

# Total loss
loss = zinb_loss + 0.1 * ce_loss
```

---

### Complete Directory Structure

```
patient_001_dataset/
│
├── patches/                        # Extracted from WSI
│   ├── spot_0001.png              # 224×224 RGB, ~150 KB
│   ├── spot_0002.png              # 224×224 RGB, ~145 KB
│   ├── spot_0003.png              # 224×224 RGB, ~148 KB
│   ├── spot_0004.png              # 224×224 RGB, ~152 KB
│   └── spot_0005.png              # 224×224 RGB, ~149 KB
│
├── spatial_coordinates.csv         # 5 rows × 3 columns (spot_id, x, y)
├── spatial_edges.csv               # 12 rows × 2 columns (source, target)
├── gene_expression.csv             # 5 rows × 11 columns (spot_id + 10 genes)
├── cell_types.csv                  # 5 rows × 2 columns (spot_id, cell_type)
│
└── metadata.json                   # Optional dataset info
    {
      "patient_id": "patient_001",
      "n_spots": 5,
      "n_genes": 10,
      "n_cell_types": 3,
      "platform": "10X Visium",
      "tissue": "Breast Cancer",
      "resolution": "0.25 um/pixel"
    }
```

---

### Data Relationships

```
┌──────────────────────────────────────────────────────────────┐
│                    ONE-TO-ONE RELATIONSHIPS                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  spot_0001  →  patches/spot_0001.png (image)                 │
│             →  x=1200, y=3450 (location)                     │
│             →  [12.3, 8.5, 15.2, ...] (gene expression)      │
│             →  cell_type=0 (Cancer)                          │
│                                                               │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    MANY-TO-MANY RELATIONSHIPS                 │
│                      (via spatial_edges.csv)                  │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  spot_0001 (index 0)  ←→  spot_0002 (index 1)                │
│                       ←→  spot_0004 (index 3)                │
│                                                               │
│  spot_0002 (index 1)  ←→  spot_0001 (index 0)                │
│                       ←→  spot_0003 (index 2)                │
│                       ←→  spot_0004 (index 3)                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

### Why This Example Matters

**Understanding the connections:**

1. **Each spot is a physical location** in the tissue (x, y coordinates)
2. **Each spot has a histopathology image** (224×224 patch)
3. **Each spot has gene expression** (measured via RNA sequencing)
4. **Each spot has neighbors** (defined by spatial proximity)
5. **The model learns** to predict gene expression from the image AND its spatial context

**Key insight**:
- Spot `spot_0002` (Immune cell) is **surrounded by** cancer cells (spot_0001, spot_0003) and fibroblasts (spot_0004)
- The GAT learns: "This image looks immune-like AND it's near cancer cells → predict high CD8A/CD4 expression"
- This is **spatial transcriptomics**: gene expression depends on tissue microenvironment!

---

### Scaling to Real Data

| Aspect | This Example | Real 10X Visium | Real Slide-seq |
|--------|-------------|-----------------|----------------|
| **Spots** | 5 | 2,000 - 5,000 | 10,000 - 50,000 |
| **Genes** | 10 | 2,000 - 20,000 | 2,000 - 20,000 |
| **Edges** | 12 | ~12,000 - 30,000 | ~60,000 - 300,000 |
| **Cell Types** | 3 | 5 - 20 | 5 - 20 |
| **WSI Size** | 75K × 62K pixels | 50K - 200K pixels | 50K - 200K pixels |
| **Dataset Size** | ~1 MB | ~500 MB - 2 GB | ~1 - 5 GB |

---

## Input Data Requirements

### 1. Whole Slide Image (WSI)

**Format:**
- `.svs`, `.tiff`, `.ndpi`, or other WSI formats
- Typically very large: 50,000 × 50,000 to 200,000 × 200,000 pixels
- H&E stained tissue sections
- High resolution: 0.25-0.5 μm/pixel

**Example:**
```
patient_001.svs
Size: 75,432 × 62,108 pixels
Resolution: 0.25 μm/pixel
File size: 2.3 GB
```

### 2. Spatial Transcriptomics Data

**From 10X Visium:**
```
spatial/
├── tissue_positions_list.csv    # Spot coordinates
├── scalefactors_json.json       # Coordinate scaling factors
└── tissue_hires_image.png       # Downsampled tissue image
```

**From other platforms (e.g., Slide-seq, MERFISH):**
- CSV file with spot/cell coordinates
- Gene expression matrix
- Cell type annotations (optional)

### 3. Gene Expression Matrix

**Format:** CSV or H5AD (AnnData)
```csv
spot_id,Gene_1,Gene_2,Gene_3,...
spot_0001,5.2,0.0,12.3,...
spot_0002,3.1,8.7,0.0,...
```

---

## Workflow Steps

### Complete Pipeline

```
Step 1: Load WSI and Spatial Data
   ↓
Step 2: Align Coordinates (Visium space → WSI space)
   ↓
Step 3: Filter Spots (keep only in-tissue spots)
   ↓
Step 4: Extract 224×224 Patches (centered at each spot)
   ↓
Step 5: Color Normalization (Macenko method)
   ↓
Step 6: Build Spatial Graph (k-NN based on coordinates)
   ↓
Step 7: Save Dataset (patches, coordinates, edges, expression)
   ↓
Ready for hist2scRNA Training!
```

---

## Patch Extraction

### Step-by-Step Process

#### 1. Load Whole Slide Image

```python
# Pseudocode (conceptual, not to run)
import openslide

# Open WSI
slide = openslide.OpenSlide('patient_001.svs')

# Get dimensions
width, height = slide.dimensions
print(f"WSI size: {width} × {height} pixels")

# Get magnification level
magnification = slide.properties.get('aperio.AppMag', '20')
```

#### 2. Load Spot Coordinates

```python
# From 10X Visium
import pandas as pd

spots = pd.read_csv('spatial/tissue_positions_list.csv',
                    header=None,
                    names=['barcode', 'in_tissue', 'array_row', 'array_col',
                           'pxl_col_in_fullres', 'pxl_row_in_fullres'])

# Filter in-tissue spots
spots_in_tissue = spots[spots['in_tissue'] == 1]

print(f"Total spots: {len(spots)}")
print(f"In-tissue spots: {len(spots_in_tissue)}")
```

#### 3. Extract Patch for Each Spot

```python
PATCH_SIZE = 224  # Standard for ViT

for idx, spot in spots_in_tissue.iterrows():
    # Get center coordinates in WSI
    center_x = spot['pxl_col_in_fullres']
    center_y = spot['pxl_row_in_fullres']

    # Calculate patch boundaries
    left = center_x - PATCH_SIZE // 2
    top = center_y - PATCH_SIZE // 2

    # Extract patch
    patch = slide.read_region(
        location=(left, top),
        level=0,  # Highest resolution
        size=(PATCH_SIZE, PATCH_SIZE)
    )

    # Convert to RGB (remove alpha channel)
    patch_rgb = patch.convert('RGB')

    # Save patch
    spot_id = f"spot_{idx:04d}"
    patch_rgb.save(f"patches/{spot_id}.png")
```

### Patch Size Considerations

| Aspect | 224×224 (Standard) | 512×512 (Large) |
|--------|-------------------|----------------|
| **Coverage** | ~55-100 μm tissue | ~125-230 μm tissue |
| **Context** | Local microenvironment | Broader tissue structure |
| **Memory** | Lower GPU memory | Higher GPU memory |
| **Processing** | Faster | Slower |
| **Recommendation** | ✓ Default for ViT | Special cases only |

### Visual Representation

```
┌───────────────────────────────────────┐
│                                       │
│  Whole Slide Image (WSI)              │
│                                       │
│    ● ← Spot (x=1250, y=3400)         │
│   ┌─────────────┐                    │
│   │             │                    │
│   │  224×224    │                    │
│   │   patch     │                    │
│   │      ●      │ ← Center on spot   │
│   │             │                    │
│   └─────────────┘                    │
│                                       │
└───────────────────────────────────────┘

Extracted patch saved as: patches/spot_0001.png
```

---

## Spatial Graph Construction

### Why Spatial Graphs?

Gene expression in a spot is influenced by its neighboring cells. The spatial graph allows the model to:
- Aggregate information from nearby spots
- Learn spatial patterns (e.g., tumor-stroma interface)
- Model cell-cell interactions

### K-Nearest Neighbors (k-NN) Approach

#### Step 1: Compute Pairwise Distances

```python
from scipy.spatial import KDTree
import numpy as np

# Get spot coordinates (x, y)
coordinates = spots_in_tissue[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values

# Build k-d tree for efficient neighbor search
tree = KDTree(coordinates)

# For each spot, find k nearest neighbors
k = 6  # Typical for hexagonal Visium grid
edges = []

for i, coord in enumerate(coordinates):
    # Query k+1 neighbors (includes self)
    distances, indices = tree.query(coord, k=k+1)

    # Skip first index (self) and create edges
    for j in indices[1:]:
        edges.append([i, j])

# Convert to edge list
edge_index = np.array(edges).T  # Shape: (2, num_edges)
```

#### Step 2: Distance Threshold (Alternative)

Instead of k-NN, use distance threshold:

```python
# Connect spots within distance threshold
threshold = 200  # pixels (~100 μm for 0.5 μm/pixel)

edges = []
for i in range(len(coordinates)):
    for j in range(i+1, len(coordinates)):
        dist = np.linalg.norm(coordinates[i] - coordinates[j])
        if dist <= threshold:
            edges.append([i, j])
            edges.append([j, i])  # Undirected graph
```

### Visium Hexagonal Grid

10X Visium has a **hexagonal spot arrangement**:

```
    ●   ●   ●   ●       Spot spacing: 100 μm center-to-center
      ●   ●   ●   ●     Spot diameter: 55 μm
    ●   ●   ●   ●
      ●   ●   ●   ●     Each spot typically has 6 neighbors
```

For Visium, **k=6** neighbors captures the hexagonal structure.

### Graph Statistics

Expected graph properties:
- **Nodes**: Number of in-tissue spots (typically 2,000-5,000 for Visium)
- **Edges**: ~6× nodes for k=6 (hexagonal grid)
- **Density**: Sparse (< 1% of possible edges)
- **Clustering**: High (spots in same tissue region are interconnected)

---

## Coordinate Alignment

### Problem: Multiple Coordinate Systems

1. **Visium Coordinate Space**: Array row/col indices (e.g., row=10, col=5)
2. **Visium Image Space**: Pixels in tissue_hires_image.png
3. **WSI Pixel Space**: Pixels in the actual whole slide image

### Solution: Transformation Matrix

10X Space Ranger provides `scalefactors_json.json`:

```json
{
    "tissue_hires_scalef": 0.08,
    "tissue_lowres_scalef": 0.02,
    "fiducial_diameter_fullres": 144.44,
    "spot_diameter_fullres": 89.43
}
```

#### Coordinate Transformation

```python
import json

# Load scale factors
with open('spatial/scalefactors_json.json', 'r') as f:
    scale_factors = json.load(f)

# Convert Visium pixel coordinates to WSI coordinates
def visium_to_wsi(visium_x, visium_y, scale_factor):
    """
    Convert Visium coordinates to WSI coordinates

    Args:
        visium_x, visium_y: Coordinates in tissue_positions_list.csv
        scale_factor: From scalefactors_json.json

    Returns:
        wsi_x, wsi_y: Coordinates in WSI pixel space
    """
    # These are already in fullres (WSI) space for Visium
    # No additional scaling needed
    return visium_x, visium_y

# For other platforms, you may need:
# wsi_x = visium_x / scale_factor
# wsi_y = visium_y / scale_factor
```

### Manual Alignment (If Needed)

If automatic alignment fails:

1. **Use QuPath or similar tool**
2. **Manually annotate fiducial markers** (corner spots on Visium slide)
3. **Compute affine transformation matrix**
4. **Apply transformation to all spot coordinates**

---

## Tools and Software

### Essential Tools

#### 1. OpenSlide (Python)
**Purpose**: Read WSI files
```bash
pip install openslide-python
```

**Supported formats**: .svs, .tiff, .ndpi, .vms, .vmu, .scn, .mrxs, .bif, .svslide

#### 2. Scanpy (Python)
**Purpose**: Spatial transcriptomics analysis
```bash
pip install scanpy squidpy
```

**Features**:
- Load 10X Visium data
- Quality control
- Preprocessing
- Integration with AnnData

#### 3. QuPath
**Purpose**: Visual inspection and annotation
- **Download**: https://qupath.github.io/
- **Features**:
  - View WSI and spots overlay
  - Manual annotation
  - Quality control
  - Export coordinates

#### 4. 10X Space Ranger
**Purpose**: Official Visium pipeline
- **Download**: https://support.10xgenomics.com/spatial-gene-expression/software
- **Features**:
  - Automatic tissue detection
  - Image alignment
  - Gene expression quantification

### Recommended Workflow

```
Step 1: Space Ranger (10X official pipeline)
   ↓
Step 2: QuPath (visual QC)
   ↓
Step 3: Python script (patch extraction)
   ↓
Step 4: hist2scRNA (training)
```

---

## Common Issues and Solutions

### Issue 1: Coordinate Misalignment

**Symptom**: Patches don't align with tissue morphology

**Causes**:
- Wrong coordinate system
- Tissue rotation/flipping
- Scale factor mismatch

**Solutions**:
```python
# Check if coordinates need flipping
# Try different transformations
wsi_x = visium_x
wsi_y = height - visium_y  # Flip Y axis

# Or rotate coordinates
import numpy as np
angle = np.deg2rad(90)  # Try 90, 180, 270 degrees
rotation_matrix = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])
```

### Issue 2: Patches Outside Tissue

**Symptom**: Many blank/white patches

**Solution**:
```python
# Filter spots based on tissue mask
def is_in_tissue(patch):
    """Check if patch contains enough tissue"""
    # Convert to grayscale
    gray = np.array(patch.convert('L'))

    # Threshold (tissue is darker than background)
    tissue_pixels = (gray < 220).sum()
    total_pixels = gray.size

    # Require at least 50% tissue
    return (tissue_pixels / total_pixels) > 0.5

# Only save patches with sufficient tissue
if is_in_tissue(patch):
    patch.save(f"patches/{spot_id}.png")
```

### Issue 3: Memory Issues with Large WSI

**Symptom**: Out of memory when loading WSI

**Solution**:
```python
# Use pyramid levels (lower resolution)
# Level 0: highest resolution
# Level 1: 2x downsampled
# Level 2: 4x downsampled, etc.

level = 1  # Use lower resolution
downsample = slide.level_downsamples[level]

# Adjust coordinates for lower resolution
adjusted_x = int(center_x / downsample)
adjusted_y = int(center_y / downsample)

# Extract at lower level
patch = slide.read_region(
    location=(adjusted_x, adjusted_y),
    level=level,
    size=(PATCH_SIZE, PATCH_SIZE)
)

# Resize to 224×224 if needed
patch_resized = patch.resize((224, 224))
```

### Issue 4: Color Variation Across Slides

**Symptom**: Different staining intensity between patients

**Solution**: Apply Macenko color normalization
```python
from torchstain import normalizers

# Initialize normalizer with reference image
normalizer = normalizers.MacenkoNormalizer()
normalizer.fit(reference_image)

# Normalize each patch
normalized_patch = normalizer.normalize(patch)
```

---

## Example: Complete Preprocessing Script

### Minimal Working Example

```python
"""
Preprocessing script for hist2scRNA
Extracts patches from WSI based on Visium spot locations
"""

import openslide
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.spatial import KDTree

# Configuration
WSI_PATH = 'patient_001.svs'
VISIUM_DIR = 'spatial/'
OUTPUT_DIR = 'dataset/'
PATCH_SIZE = 224
K_NEIGHBORS = 6

# Create output directories
Path(OUTPUT_DIR + 'patches/').mkdir(parents=True, exist_ok=True)

# Step 1: Load WSI
print("Loading WSI...")
slide = openslide.OpenSlide(WSI_PATH)
print(f"WSI dimensions: {slide.dimensions}")

# Step 2: Load spot coordinates
print("Loading spot coordinates...")
spots = pd.read_csv(
    VISIUM_DIR + 'tissue_positions_list.csv',
    header=None,
    names=['barcode', 'in_tissue', 'array_row', 'array_col',
           'pxl_col_in_fullres', 'pxl_row_in_fullres']
)

# Filter in-tissue spots
spots_in_tissue = spots[spots['in_tissue'] == 1].reset_index(drop=True)
print(f"In-tissue spots: {len(spots_in_tissue)}")

# Step 3: Extract patches
print("Extracting patches...")
coordinates = []

for idx, spot in spots_in_tissue.iterrows():
    spot_id = f"spot_{idx:04d}"

    # Get center coordinates
    center_x = int(spot['pxl_col_in_fullres'])
    center_y = int(spot['pxl_row_in_fullres'])

    # Calculate patch boundaries
    left = center_x - PATCH_SIZE // 2
    top = center_y - PATCH_SIZE // 2

    # Extract patch
    patch = slide.read_region(
        location=(left, top),
        level=0,
        size=(PATCH_SIZE, PATCH_SIZE)
    )

    # Convert and save
    patch_rgb = patch.convert('RGB')
    patch_rgb.save(OUTPUT_DIR + f'patches/{spot_id}.png')

    # Store coordinates
    coordinates.append([center_x, center_y])

    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(spots_in_tissue)} spots")

# Step 4: Build spatial graph
print("Building spatial graph...")
coordinates = np.array(coordinates)
tree = KDTree(coordinates)

edges = []
for i in range(len(coordinates)):
    # Find k nearest neighbors
    distances, indices = tree.query(coordinates[i], k=K_NEIGHBORS+1)

    # Create edges (skip self)
    for j in indices[1:]:
        edges.append([i, j])

# Step 5: Save metadata
print("Saving metadata...")

# Save coordinates
coord_df = pd.DataFrame(coordinates, columns=['x', 'y'])
coord_df['spot_id'] = [f"spot_{i:04d}" for i in range(len(coordinates))]
coord_df.to_csv(OUTPUT_DIR + 'spatial_coordinates.csv', index=False)

# Save edges
edge_df = pd.DataFrame(edges, columns=['source', 'target'])
edge_df.to_csv(OUTPUT_DIR + 'spatial_edges.csv', index=False)

print(f"Preprocessing complete!")
print(f"  Patches: {len(coordinates)}")
print(f"  Edges: {len(edges)}")
print(f"  Output: {OUTPUT_DIR}")
```

---

## Summary

### Key Takeaways

1. **One patch per spot** - Not random sampling
2. **224×224 pixels** - Standard for Vision Transformers
3. **Spatial alignment is critical** - Patches must match spot locations
4. **Build spatial graph** - Connects neighboring spots (k=6 for Visium)
5. **Quality control** - Verify alignment visually

### Dataset Structure

```
dataset/
├── patches/
│   ├── spot_0000.png (224×224 RGB)
│   ├── spot_0001.png
│   └── ...
├── spatial_coordinates.csv (spot_id, x, y)
├── spatial_edges.csv (source, target)
├── gene_expression.csv (spot_id, Gene_1, Gene_2, ...)
└── cell_types.csv (spot_id, cell_type) [optional]
```

### Next Steps

After preprocessing, you're ready to:
1. **Train hist2scRNA model**
2. **Predict gene expression** for new spots
3. **Analyze spatial patterns** in predicted data

See [QUICKSTART_SCRNA.md](./QUICKSTART_SCRNA.md) for training instructions.

---

## Additional Resources

### Papers
- **10X Visium**: https://www.10xgenomics.com/spatial-transcriptomics
- **Slide-seq**: https://science.sciencemag.org/content/363/6434/1463
- **MERFISH**: https://science.sciencemag.org/content/348/6233/aaa6090

### Tutorials
- **Scanpy Spatial Tutorial**: https://scanpy-tutorials.readthedocs.io/en/latest/spatial/basic-analysis.html
- **Squidpy Documentation**: https://squidpy.readthedocs.io/
- **OpenSlide Python API**: https://openslide.org/api/python/

### Datasets
- **10X Visium Public Datasets**: https://www.10xgenomics.com/resources/datasets
- **Spatial Research**: https://www.spatialresearch.org/
