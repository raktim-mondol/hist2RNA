### IMPORT LIBRARIES ###
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import csv
import pandas as pd

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from color_normalizer import MacenkoColorNormalization
from data_load import PatientDataset 
from torchvision.models import (resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, 
                                vit_l_32, ViT_L_32_Weights, vit_l_16, ViT_L_16_Weights, 
                                maxvit_t, MaxVit_T_Weights, swin_b, Swin_B_Weights,
                                efficientnet_v2_m, EfficientNet_V2_M_Weights)
from scipy.stats import spearmanr
import numpy as np
from tqdm import tqdm
import csv

def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#for reproducibility 
set_seeds()

# Hyperparameters

LR = 0.0001
WEIGHT_DECAY = 0.01
EPOCHS = 500
BATCH_SIZE = 12
NUM_OF_PATIENCE = 20
NUM_OF_GENES = 50
# choose the base model 
BASE_MODEL_NAME = "swin_b"  

FEATURES_DIR = "./saved_features/resnet50/"

# file_name and path 
CHECKPOINT_PATH = './saved_model/best_model.pt'

#slides folder should contain folder of all patients (folder name: patient id) and each folder should contain multiple patches e.g. 1000 patches 
# Path to extracted features



gene_expression_file = './data/gene_expression/pam50_gene_expression.csv'

train_data = pd.read_csv('.data/patient_details/train_patient_list.txt', index_col='patient_id')
train_patient_ids = train_data.index.astype(str).tolist()

test_data = pd.read_csv('.data/patient_details/test_patient_list.txt', index_col='patient_id')
test_patient_ids = test_data.index.astype(str).tolist()



# Your hist2RNA model, EarlyStopping class and other definitions

# Function to load features for a patient
class PatientDatasetWithFeatures(Dataset):
    def __init__(self, features_dir, patient_ids, gene_expression_file):
        """
        Args:
            features_dir (str): Path to the directory containing the extracted features.
            patient_ids (list): List of patient IDs.
            gene_expression_file (str): Path to the gene expression CSV file.
        """
        self.features_dir = features_dir
        self.gene_expression_data = pd.read_csv(gene_expression_file, index_col='PATIENT_ID')
        
        # Apply the log2(1+x) transformation to the entire gene_expression_data DataFrame
        self.gene_expression_data = np.log2(1 + self.gene_expression_data)
        self.patient_ids = patient_ids

    def load_features(self, patient_id):
        feature_path = os.path.join(self.features_dir, f"{patient_id}.pt")
        return torch.load(feature_path)


    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        
        # Use the load_features function
        patient_features = self.load_features(patient_id)
        
        # Load the gene expression data for this patient
        gene_expression = self.gene_expression_data.loc[patient_id]
        gene_expression = torch.tensor(gene_expression.values, dtype=torch.float32)

        return patient_features, gene_expression
        

# Example usage

def get_base_model_and_transforms(model_name):
    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Identity()
        
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Identity()
        
    elif model_name == "efficientnet_v2_m":
        weights = EfficientNet_V2_M_Weights.DEFAULT
        model = efficientnet_v2_m(weights=weights)  
        num_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
        
    elif model_name == "vit_l_32":
        weights = ViT_L_32_Weights.DEFAULT
        model = vit_l_32(weights=weights)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Identity()
        
    elif model_name == "vit_l_16":
        weights = ViT_L_16_Weights.DEFAULT
        model = vit_l_16(weights=weights)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Identity()
        
    elif model_name == "maxvit_t":
        weights = MaxVit_T_Weights.DEFAULT
        model = maxvit_t(weights=weights)
        num_features = model.classifier[3].in_features
        model.classifier = nn.Sequential(
            model.classifier[0],  # AdaptiveAvgPool2d(output_size=1)
            model.classifier[1]  # Flatten(start_dim=1, end_dim=-1)
            )
    elif model_name == "swin_b":
        weights = Swin_B_Weights.DEFAULT
        model = swin_b(weights=weights)
        num_features = model.head.in_features
        model.head = nn.Identity()
        
    else:
        raise ValueError(f"Unknown model name {model_name}")

    # Make all the layer of the base model trainable 
    for param in model.parameters():
        param.requires_grad = False
    
    preprocess = weights.transforms()
    
    # model name; number of output features, and model specific preprocess
    return model, num_features, preprocess
    
    
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
class hist2RNA(nn.Module):
    def __init__(self, feature_out):
        super(hist2RNA, self).__init__()

        # 1D Convolution blocks
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=feature_out, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1),
            nn.ReLU()
        )
        
        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU()
        )
        
        # GAP
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
      
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(in_features=512, out_features=NUM_OF_GENES)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # convert to [batch_size, features, 1]
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.output(x)
        return x
        
        
        
# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# this is required to know the number of output features of the base model
base_model, num_features, preprocess = get_base_model_and_transforms(BASE_MODEL_NAME)


dataset = PatientDatasetWithFeatures(FEATURES_DIR, train_patient_ids, gene_expression_file)


dataset_length = len(dataset)
train_length = int(0.9 * dataset_length)
valid_length = dataset_length - train_length

train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Initialize model, criterion and optimizer

model = hist2RNA(feature_out=num_features).to(device)
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# Define the early stopping mechanism
early_stopping = EarlyStopping(patience=NUM_OF_PATIENCE, verbose=True, path=CHECKPOINT_PATH)


# Initialize lists to store the training and validation losses
train_losses = []
val_losses = []


# Training loop

for epoch in range(EPOCHS):
    # Training Phase
    model.train()  # set the model to training mode
    running_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        images, gene_expression = data
        x, y = images.to(device), gene_expression.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # Append the average training loss for this epoch
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Training Loss: {avg_train_loss}")

    # Validation Phase
    model.eval()  # set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            images, gene_expression = data
            x, y = images.to(device), gene_expression.to(device)
                
            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_loader)
    val_losses.append(avg_val_loss)  # Append the average validation loss for this epoch
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Validation Loss: {avg_val_loss}")

    # Early stopping check
    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

print("Finished Training!")

# Plot training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('./save_figure/training_validation_loss_curve.png')



# Load the saved model weights


checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint)



test_dataset = PatientDatasetWithFeatures(FEATURES_DIR, test_patient_ids, gene_expression_file)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Ensure model is in evaluation mode and on the correct device
model.eval()
model.to(device)

all_predictions = []
all_true_values = []


# Make predictions on the test dataset
with torch.no_grad():
    for images, gene_expression in test_dataloader:
        #images = images.to(device)
        x, y = images.to(device), gene_expression.to(device)
        outputs = model(x)
        all_predictions.append(outputs.cpu().numpy())
        all_true_values.append(y.cpu().numpy())

all_predictions = np.vstack(all_predictions)
all_true_values = np.vstack(all_true_values)

# Get the gene names from the gene_expression_file for column headers
df_gene_expression = pd.read_csv(gene_expression_file)
gene_names = df_gene_expression.columns.tolist()
gene_names = gene_names[1:]

# Save predicted and true gene expressions into a CSV file using pandas
df_predicted = pd.DataFrame(all_predictions, columns=gene_names, index=test_patient_ids)
df_true = pd.DataFrame(all_true_values, columns=gene_names, index=test_patient_ids)

# Saving to CSV
df_predicted.to_csv('./save_result/predicted_gene_expressions.csv')
df_true.to_csv('./save_result/true_gene_expressions.csv')

# Note: The shape of all_predictions and all_true_values is (num_patients, 138)

spearman_coeffs_per_patient = []
p_values_per_patient = []
# Compute the Spearman correlation coefficient for the entire gene set of each patient
for i in range(all_predictions.shape[0]):
    coefficient, p_value = spearmanr(all_predictions[i], all_true_values[i])
    spearman_coeffs_per_patient.append(coefficient)
    p_values_per_patient.append(p_value)

#print("Spearman correlation coefficients for each patient:", spearman_coeffs_per_patient)
#print("P-values for each patient:", p_values_per_patient)

spearman_coeffs_per_gene = []
p_values_per_gene = []

# Compute the Spearman correlation coefficient for each gene across all patients
for j in range(all_predictions.shape[1]):  # iterate through each gene
    coefficient, p_value = spearmanr(all_predictions[:, j], all_true_values[:, j])
    spearman_coeffs_per_gene.append(coefficient)
    p_values_per_gene.append(p_value)

#print("Spearman correlation coefficients for each gene:", spearman_coeffs_per_gene)
#print("P-values for each gene:", p_values_per_gene)


# Training parameters and other details
details = {
    "LR": LR,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "BATCH_SIZE": BATCH_SIZE,
    "Base Model Name": BASE_MODEL_NAME
}

# Filename definitions
FILENAME_PATIENT = "./save_result/test_result_across_patient.csv"
FILENAME_GENE = "./save_result/test_result_across_gene.csv"

def save_to_csv(filename, headers, data1, data2, details):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Writing training details
        for key, value in details.items():
            writer.writerow([key, value])
        
        # Add an empty line
        writer.writerow([])
        
        # Headers
        writer.writerow(headers)
        
        # Get the maximum number of rows to iterate over
        max_rows = max(len(data1), len(data2))

        for i in range(max_rows):
            row = [
                data1[i] if i < len(data1) else '',
                data2[i] if i < len(data2) else ''
            ]
            writer.writerow(row)

details = {
    "LR": LR,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "BATCH_SIZE": BATCH_SIZE,
    "Base Model Name": BASE_MODEL_NAME
}

# Save patient data
save_to_csv(FILENAME_PATIENT, 
            ["Spearman correlation coef across each patient", "P-values across each patient"],
            spearman_coeffs_per_patient, 
            p_values_per_patient,
            details)

# Save gene data
save_to_csv(FILENAME_GENE, 
            ["Spearman correlation coef across each gene", "P-values across each gene"],
            spearman_coeffs_per_gene, 
            p_values_per_gene,
            details)

