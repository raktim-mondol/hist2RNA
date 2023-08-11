# hist2RNA: Predicting Gene Expression from Histopathology Images [[Paper]](https://www.mdpi.com/2072-6694/15/9/2569)

![hist2RNA banner](https://github.com/raktim-mondol/hist2RNA/blob/main/banner_hist2RNA_updated.png)


## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Data Sources](#data-sources)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

**hist2RNA** is an efficient deep learning-based project that aims to predict gene expression from breast cancer histopathology images. This project employs a efficient architecture to unlock underlying genetic expression in breast cancer.


## Features

- A state-of-the-art deep learning model tailored for breast cancer histopathology images
- Efficient prediction of gene expression from histopathology images which means less training time
- User-friendly command-line interface
- Comprehensive documentation and tutorials

## Data Sources

The following data sources have been used in this project:

- Genetic Data:
  - [BRCA TCGA](http://www.cbioportal.org/study/summary?id=brca_tcga)
  - [BRCA TCGA Pub2015](http://www.cbioportal.org/study/summary?id=brca_tcga_pub2015)
- Diagnostic Slide (DS): [GDC Data Portal](https://portal.gdc.cancer.gov/)
- DS Download Guideline: [Download TCGA Digital Pathology Images (FFPE)](http://www.andrewjanowczyk.com/download-tcga-digital-pathology-images-ffpe/)

## Requirements

- Python 3.9+
- Pytorch 2.0

## Image preprocessing

### Annotation and Patch Creation

- [Qupath Guideline](https://github.com/raktim-mondol/qu-path)

### Image Color Normalization

- Python implementation: [Normalizing H&E Images](https://github.com/bnsreenu/python_for_microscopists/blob/master/122_normalizing_HnE_images.py)

- Actual Matlab implementation: [Staining Normalization](https://github.com/mitkovetta/staining-normalization/blob/master/normalizeStaining.m)

- Reference: [Macenko et al. (2009) - A method for normalizing histology slides for quantitative analysis](http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf)

## Installation

1. Clone the repository:
```git clone https://github.com/raktim-mondol/hist2RNA.git``` 

2. Change directory to the cloned repository:
```cd hist2RNA```

3. Install the required packages:
```pip install -r requirements.txt```
  
  
1. Train the model:
```python
python train.py --train_data_path ./data/train --epochs 50 --batch_size 32
```

2. Evaluate the model:
```python
python evaluate.py --test_data_path ./data/test --model_path ./models/hist2RNA_model.h5
```

3. Predict gene expression from a single image:
```python
python predict.py --image_path ./data/sample.jpg --model_path ./models/hist2RNA_model.h5
```


For detailed usage instructions, please refer to the [documentation](./DOCUMENTATION.md).
## Notable Results using hist2RNA
![Spearman Correlation Coefficient](https://github.com/raktim-mondol/hist2RNA/assets/28592095/7f4aa4e1-4048-4cf7-9bff-1f20ea711dba)

## Contributing

We welcome contributions to improve and expand the capabilities of **hist2RNA**! Please follow the [contributing guidelines](./CONTRIBUTING.md) to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Cite Us: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")
If you find this code useful in your research, please consider citing:
```
@Article{cancers15092569,
AUTHOR = {Mondol, Raktim Kumar and Millar, Ewan K. A. and Graham, Peter H. and Browne, Lois and Sowmya, Arcot and Meijering, Erik},
TITLE = {hist2RNA: An Efficient Deep Learning Architecture to Predict Gene Expression from Breast Cancer Histopathology Images},
JOURNAL = {Cancers},
VOLUME = {15},
YEAR = {2023},
NUMBER = {9},
ARTICLE-NUMBER = {2569},
URL = {https://www.mdpi.com/2072-6694/15/9/2569},
ISSN = {2072-6694},
DOI = {10.3390/cancers15092569}
}
```


