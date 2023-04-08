# hist2RNA Documentation

Welcome to the hist2RNA documentation! This guide will walk you through the process of using hist2RNA to predict gene expression from breast cancer histopathology images. Please follow the instructions below to get started.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Preparing the Data](#preparing-the-data)
4. [Training the Model](#training-the-model)
5. [Evaluating the Model](#evaluating-the-model)
6. [Predicting Gene Expression](#predicting-gene-expression)
7. [Troubleshooting](#troubleshooting)

## Requirements

Before you begin, ensure that your system meets the following requirements:

- Python 3.7+
- TensorFlow 2.0+
- NumPy
- pandas
- scikit-learn
- OpenCV
- Matplotlib

## Installation

To install hist2RNA, follow these steps:

1. Clone the repository:
```git clone https://github.com/yourusername/hist2RNA.git```

2. Change directory to the cloned repository:
```cd hist2RNA```

3. Install the required packages:
```pip install -r requirements.txt```


## Preparing the Data

Before training the model, you'll need to prepare your dataset. Ensure that your data is organized into separate folders for training and testing. Each folder should contain subfolders for each class, with the corresponding images inside.

## Training the Model

To train the hist2RNA model, use the `train.py` script as follows:

```bash
python train.py --train_data_path ./data/train --epochs 50 --batch_size 32
```

This command will train the model using the training data in the ./data/train folder, with 50 epochs and a batch size of 32.

## Evaluating the Model
To evaluate the performance of the trained model, use the evaluate.py script:
