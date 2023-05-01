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

- Python 3.7+
- TensorFlow 2.0+
- NumPy
- pandas
- scikit-learn
- OpenCV
- Matplotlib

## Installation

1. Clone the repository:
```git clone https://github.com/raktim-mondol/hist2RNA.git``` 

2. Change directory to the cloned repository:
```cd hist2RNA```

3. Install the required packages:
```pip install -r requirements.txt```


## Usage

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
ABSTRACT = {Gene expression can be used to subtype breast cancer with improved prediction of risk of recurrence and treatment responsiveness over that obtained using routine immunohistochemistry (IHC). However, in the clinic, molecular profiling is primarily used for ER+ cancer, is costly, tissue destructive, requires specialised platforms, and takes several weeks to obtain a result. Deep learning algorithms can effectively extract morphological patterns in digital histopathology images to predict molecular phenotypes quickly and cost-effectively. We propose a new, computationally efficient approach called hist2RNA inspired by bulk RNA sequencing techniques to predict the expression of 138 genes (incorporated from 6 commercially available molecular profiling tests), including luminal PAM50 subtype, from hematoxylin and eosin (H&amp;E)-stained whole slide images (WSIs). The training phase involves the aggregation of extracted features for each patient from a pretrained model to predict gene expression at the patient level using annotated H&amp;E images from The Cancer Genome Atlas (TCGA, n = 335). We demonstrate successful gene prediction on a held-out test set (n = 160, corr = 0.82 across patients, corr = 0.29 across genes) and perform exploratory analysis on an external tissue microarray (TMA) dataset (n = 498) with known IHC and survival information. Our model is able to predict gene expression and luminal PAM50 subtype (Luminal A versus Luminal B) on the TMA dataset with prognostic significance for overall survival in univariate analysis (c-index = 0.56, hazard ratio = 2.16 (95% CI 1.12&ndash;3.06), p &lt; 5 &times; 10&minus;3), and independent significance in multivariate analysis incorporating standard clinicopathological variables (c-index = 0.65, hazard ratio = 1.87 (95% CI 1.30&ndash;2.68), p &lt; 5 &times; 10&minus;3). The proposed strategy achieves superior performance while requiring less training time, resulting in less energy consumption and computational cost compared to patch-based models. Additionally, hist2RNA predicts gene expression that has potential to determine luminal molecular subtypes which correlates with overall survival, without the need for expensive molecular testing.},
DOI = {10.3390/cancers15092569}
}
```


