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
                                maxvit_t, MaxVit_T_Weights, swin_b, Swin_B_Weights)
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
LR = 0.001
WEIGHT_DECAY = 0.01
EPOCHS = 100
BATCH_SIZE = 12

# choose the base model 
BASE_MODEL_NAME = "resnet50"  

# file_name and path 
CHECKPOINT_PATH = 'best_model.pt'
FILENAME = "training_details_and_results.csv"

#slides folder should contain folder of all patients (folder name: patient id) and each folder should contain multiple patches e.g. 1000 patches 

slides_dir = 'F:/main_folder/slides/'
gene_expression_file = 'F:/main_folder/gene_file/gene_expression_.csv'
train_data = pd.read_csv('F:/main_folder/list_of_patient.txt', index_col='patient_id')
train_patient_ids = train_data.index.astype(str).tolist()

test_data = pd.read_csv('F:/main_folder/list_of_patient.txt', index_col='patient_id')
test_patient_ids = train_data.index.astype(str).tolist()


def get_base_model_and_transforms(model_name):
    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = nn.Identity()
        num_features = model.fc.in_features
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
        num_features = model.fc.in_features
    elif model_name == "vit_l_32":
        weights = ViT_L_32_Weights.DEFAULT
        model = vit_l_32(weights=weights)
        model.heads.head = nn.Identity()
        num_features = model.heads.head.in_features
    elif model_name == "vit_l_16":
        weights = ViT_L_16_Weights.DEFAULT
        model = vit_l_16(weights=weights)
        model.heads.head = nn.Identity()
        num_features = model.heads.head.in_features
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
        model.head = nn.Identity()
        num_features = model.head.in_features
        
    else:
        raise ValueError(f"Unknown model name {model_name}")

    # Make all the layer of the base model trainable 
    for param in model.parameters():
        param.requires_grad = True
    
    preprocess = weights.transforms()
    
    # model name; number of output features, and model specific preprocess
    return model, num_features, preprocess


def collate_fn(batch):
    images, patient_id = zip(*batch)
    clipped_images = []
    for img in images:
        if len(img) > 500:
            img = img[:500]  # If more than 1000 patches, clip it to 1000
        clipped_images.append(img)
    return clipped_images, patient_id


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
    def __init__(self, base_model, input_features, batch_size):
        super(hist2RNA, self).__init__()
        
        # Use the provided base model
        self.base_model = base_model
        
        self.batch_size = batch_size
        
        # 1D Convolution blocks
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=256, kernel_size=5, padding=2),  
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
            nn.Linear(in_features=512, out_features=50)
        )
        
    def forward(self, x):
        
        batch_size = x[0].size(0)
        #print(batch_size)
      
        aggregated_features_list = []
        
        for patient_images in x:
            patient_features = self.base_model(patient_images)
            aggregated_feature = patient_features.mean(dim=0)
            aggregated_features_list.append(aggregated_feature)
        
        aggregated_features = torch.stack(aggregated_features_list, dim=0)
        print(aggregated_features.shape)
        print(aggregated_features.unsqueeze(2).shape)
        #[12, 512, 1]
        x = self.c1(aggregated_features.unsqueeze(2))
        x = self.c2(x)
        x = self.c3(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.output(x)
        return x

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model, num_features, preprocess = get_base_model_and_transforms(BASE_MODEL_NAME)


color_normalization = MacenkoColorNormalization()
transform = preprocess


dataset = PatientDataset(slides_dir, train_patient_ids, gene_expression_file, transform=transform, color_normalization=color_normalization)
#dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)

dataset_length = len(dataset)
train_length = int(0.9 * dataset_length)
valid_length = dataset_length - train_length

train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=1)



# Initialize model, criterion and optimizer
model = hist2RNA(base_model, input_features = num_features, batch_size=BATCH_SIZE).to(device)
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# Define the early stopping mechanism
early_stopping = EarlyStopping(patience=5, verbose=True, path=CHECKPOINT_PATH)


# Initialize lists to store the training and validation losses
train_losses = []
val_losses = []

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    # Training phase with progress bar
    for images, gene_expression in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=False):
        images = tuple(batch.to(device) for batch in images)
        gene_expression = tuple(tensor.to(device) for tensor in gene_expression)
        gene_expression = torch.stack(gene_expression, dim=0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, gene_expression)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(images)

        # Delete tensors to free up memory
        del images, gene_expression, outputs

    torch.cuda.empty_cache()

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}")

    # Validation phase
    model.eval()
    running_val_loss = 0.0

    # Validation phase with progress bar
    for images, gene_expression in tqdm(valid_loader, desc=f"Validating Epoch {epoch+1}/{EPOCHS}", leave=False):
        images = tuple(batch.to(device) for batch in images)
        gene_expression = tuple(tensor.to(device) for tensor in gene_expression)
        gene_expression = torch.stack(gene_expression, dim=0)
        
        outputs = model(images)
        loss = criterion(outputs, gene_expression)
        running_val_loss += loss.item() * len(images)

        # Delete tensors to free up memory
        del images, gene_expression, outputs

    torch.cuda.empty_cache()

    val_loss = running_val_loss / len(valid_loader.dataset)
    val_losses.append(val_loss)
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Validation Loss: {val_loss:.4f}")

    # Early stopping check
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

print("Training complete.")




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
plt.savefig('training_validation_loss_curve.png')
#plt.show()


# Load the saved model weights
checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint)



test_dataset = PatientDataset(slides_dir, test_patient_ids, gene_expression_file, transform=transform, color_normalization=color_normalization)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)


# Ensure model is in evaluation mode and on the correct device
model.eval()
model.to(device)

all_predictions = []
all_true_values = []

# Make predictions on the test dataset
with torch.no_grad():
    for images, gene_expression in test_dataloader:
        #images = images.to(device)
        images = tuple(batch.to(device) for batch in images)
        gene_expression = tuple(tensor.to(device) for tensor in gene_expression)
        gene_expression = torch.stack(gene_expression, dim=0)
        outputs = model(images)
        all_predictions.append(outputs.cpu().numpy())
        all_true_values.append(gene_expression.cpu().numpy())

all_predictions = np.vstack(all_predictions)
all_true_values = np.vstack(all_true_values)

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
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "Base Model Name": BASE_MODEL_NAME
}



# Filename definitions
FILENAME_PATIENT = "/scratch/nk53/rm8989/gene_prediction/code/hist2RNA/save_result/test_result_across_patient.csv"
FILENAME_GENE = "/scratch/nk53/rm8989/gene_prediction/code/hist2RNA/save_result/test_result_across_gene.csv"

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
    "EPOCHS": EPOCHS,
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
