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

```bash 
python evaluate.py --test_data_path ./data/test --model_path ./models/hist2RNA_model.h5
```
This command will evaluate the model using the test data in the ./data/test folder and the trained model saved in ./models/hist2RNA_model.h5.

## Predicting Gene Expression
To predict gene expression from a single histopathology image, use the predict.py script:

```bash 
python predict.py --image_path ./data/sample.jpg --model_path ./models/hist2RNA_model.h5
```
This command will use the trained model saved in ./models/hist2RNA_model.h5 to predict gene expression for the image located at ./data/sample.jpg.

## Troubleshooting
If you encounter any issues while using hist2RNA, please refer to the README.md file, check the existing issues, or create a new issue with a detailed description of the problem.


8. [Advanced Usage](#advanced-usage)

## Advanced Usage

In this section, we will explore some advanced usage scenarios and options for the hist2RNA project.

### Customizing the Model

If you want to customize the hist2RNA model architecture, you can modify the `model.py` file. This file contains the model definition and allows you to experiment with different layers, activation functions, and other hyperparameters.

### Data Augmentation

To improve the performance of the model, you can apply data augmentation techniques. To do this, modify the `train.py` script to include data augmentation options when loading the training data. You can use the `ImageDataGenerator` class from Keras to easily apply various data augmentation techniques, such as rotation, zooming, and flipping. For example:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Apply data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Load the training data with the applied augmentation
train_data = datagen.flow_from_directory(train_data_path, ...)

### Transfer Learning
To leverage the power of pre-trained models, you can use transfer learning. This approach involves using the weights from a pre-trained model as a starting point for training your model. Transfer learning can improve the performance of your model, especially when dealing with limited datasets. To implement transfer learning, modify the model.py file to include a pre-trained model (e.g., VGG16, ResNet50, etc.) as the base of your model architecture.


### Monitoring Training Progress
To monitor the training progress, you can use TensorBoard, a visualization tool provided by TensorFlow. To enable TensorBoard, add the following lines to the train.py script:

```python 
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Set up the TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Add the TensorBoard callback when fitting the model
model.fit(..., callbacks=[tensorboard_callback])
```

To visualize the training progress, run TensorBoard in your terminal:

```python
tensorboard --logdir logs/fit```

