# hist2RNA: Predicting Gene Expression from Histopathology Images

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

git clone https://github.com/raktim-mondol/hist2RNA.git

2. Change directory to the cloned repository:
- cd hist2RNA

3. Install the required packages:
- pip install -r requirements.txt


## Usage

1. Train the model:
- python train.py --train_data_path ./data/train --epochs 50 --batch_size 32

2. Evaluate the model:
- python evaluate.py --test_data_path ./data/test --model_path ./models/hist2RNA_model.h5

3. Predict gene expression from a single image:
- python predict.py --image_path ./data/sample.jpg --model_path ./models/hist2RNA_model.h5


For detailed usage instructions, please refer to the [documentation](./docs/).

## Contributing

We welcome contributions to improve and expand the capabilities of **hist2RNA**! Please follow the [contributing guidelines](./CONTRIBUTING.md) to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

