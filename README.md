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
python training_main.py --slides_dir ./data/slides/ --epochs 50 --batch_size 12 --lr 0.001
```

2. Test the model:
```python
python test_main.py --test_patient_id ./patient_details/test_patient_id.txt --checkpoint_file ./models/hist2RNA_model.pth
```

**For most efficient way use following code:**
```python
python feature_extraction_step_1.py
```
**Then,**
```python
python model_training_step_2.py
```

For detailed usage instructions, please refer to the [documentation](./DOCUMENTATION.md).
## Peak results utilizing the hist2RNA methodology:

The following results show predictions for the PAM50 genes from histopathology test datatest images:

### Spearman Correlation Coefficient **[Updated]**
![Spearman Correlation Coefficient](https://github.com/raktim-mondol/hist2RNA/assets/28592095/7f4aa4e1-4048-4cf7-9bff-1f20ea711dba)

### AUC-RCH (A performance metric we've developed)
![Reverse_cumulative_histogram](https://github.com/raktim-mondol/hist2RNA/assets/28592095/c35a99ea-429e-4bb5-a244-84a313a0a0a3)


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


