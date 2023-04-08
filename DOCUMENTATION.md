# hist2RNA Documentation

Welcome to the hist2RNA documentation! This guide will walk you through the process of using hist2RNA to predict gene expression from breast cancer histopathology images. Please follow the instructions below to get started.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Preparing the Data](#preparing-the-data)
4. [Training the Model](#training-the-model)
5. [Evaluating the Model](#evaluating-the-model)
6. [Predicting Gene Expression](#predicting-gene-expression)
7. [Advanced Usage](#usage)
8. [Troubleshooting](#troubleshooting)
9. [Frequently Asked Questions (FAQs)](#frequently-asked-questions-faqs)
10. 10. [References](#references)

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

### Data Structure of Images

The hist2RNA project expects the dataset to be organized in a specific structure to ensure proper retrieval of images for training and validation. The dataset should be organized into separate folders for each patient, with each patient folder containing 1000+ patches of histopathology images.

Here is an example of the expected directory structure:
```bash
dataset/
│
├── patient_01/
│ ├── patch_0001.png
│ ├── patch_0002.png
│ ├── ...
│ └── patch_1000.png
│
├── patient_02/
│ ├── patch_0001.png
│ ├── patch_0002.png
│ ├── ...
│ └── patch_1000.png
│
└── ...
```

Make sure to organize your dataset according to this structure before running the training script. The training script will process the data accordingly and retrieve the images based on this organization.
### Data Structure of Gene Expression
The hist2RNA project requires gene expression data to be provided as labels for each patient during training. Each patient should have 138 gene expression values corresponding to their histopathology images.

The gene expression data should be organized in a CSV (Comma Separated Values) file with the following structure:
```bash
patient_id,gene_1,gene_2,gene_3,...,gene_138
patient_01,0.23,0.56,0.78,...,1.32
patient_02,0.34,0.67,0.82,...,1.45
patient_03,0.28,0.54,0.75,...,1.28
...
```
The first row of the CSV file should contain the column names, with the first column being the `patient_id` and the subsequent columns being the gene expression values for each gene (`gene_1`, `gene_2`, ..., `gene_138`).

Each subsequent row should contain the patient ID and the gene expression values for each of the 138 genes, separated by commas.

Ensure that your gene expression data is formatted according to this structure before running the training script. The training script will read the gene expression data and associate it with the corresponding patient's histopathology images during training.



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

```
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
tensorboard --logdir logs/fit
```
Then, open your browser and navigate to the URL displayed in the terminal (usually http://localhost:6006/).

### Fine-Tuning Hyperparameters
To fine-tune the hyperparameters of your model, such as the learning rate, batch size, or number of epochs, you can modify the relevant arguments in the train.py script. Experimenting with different hyperparameters can help you optimize the performance of your model.

Add this section to the `DOCUMENTATION.md` file in the `./docs` folder. As before, remember to replace `yourusername` with your actual GitHub username and update the repository URL accordingly. Adjust file paths or other details depending on your project's structure as needed.

9. [Frequently Asked Questions (FAQs)](#frequently-asked-questions-faqs)

## Troubleshooting
If you encounter any issues while using hist2RNA, please refer to the README.md file, check the existing issues, or create a new issue with a detailed description of the problem.

## Frequently Asked Questions (FAQs)

In this section, we provide answers to some frequently asked questions about the hist2RNA project.

### Q: How can I improve the model's performance?

A: To improve the model's performance, you can try the following approaches:

- Increase the size or diversity of the training dataset.
- Apply data augmentation techniques (see the [Advanced Usage](#advanced-usage) section).
- Use transfer learning with pre-trained models (see the [Advanced Usage](#advanced-usage) section).
- Fine-tune the model's hyperparameters, such as the learning rate, batch size, or number of epochs (see the [Advanced Usage](#advanced-usage) section).
- Modify the model architecture to include additional or different layers (see the [Advanced Usage](#advanced-usage) section).

### Q: Can I use hist2RNA with other types of cancer?

A: Yes, you can adapt hist2RNA to work with other types of cancer by changing the dataset and adjusting the model architecture as needed. However, the current implementation is tailored specifically for breast cancer histopathology images, and additional modifications might be necessary for optimal performance with other types of cancer.

### Q: Can I use hist2RNA with other imaging modalities, like MRI or CT scans?

A: While hist2RNA is designed for histopathology images, it is possible to adapt the model for other imaging modalities. You would need to preprocess the data to ensure compatibility with the model and make any necessary adjustments to the model architecture.

## References

Below are some key references and resources for the hist2RNA project:

1. Original paper: Not Published Yet(#)
2. TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
3. Keras: [https://keras.io/](https://keras.io/)
4. TCGA data portal: [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/)
5. cBioPortal: [http://www.cbioportal.org/](http://www.cbioportal.org/)
6. Andrew Janowczyk's guide to downloading TCGA digital pathology images: [http://www.andrewjanowczyk.com/download-tcga-digital-pathology-images-ffpe/](http://www.andrewjanowczyk.com/download-tcga-digital-pathology-images-ffpe/)



