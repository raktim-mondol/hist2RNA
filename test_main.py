### IMPORT LIBRARIES ###

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

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
from tqdm import tqdm
def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#for reproducibility 
set_seeds()


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


def collate_fn(batch):
    images, patient_id = zip(*batch)
    clipped_images = []
    for img in images:
        if len(img) > 1000:
            img = img[:1000]  # If more than 1000 patches, clip it to 1000
        clipped_images.append(img)
    return clipped_images, patient_id
    
    
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
        
        #when tuple is used
        batch_size = x[0].size(0)
        #print(batch_size)
        
        aggregated_features_list = []
        
        #for i in range(batch_size):
        for patient_images in x:
            #print(patient_images.shape)
            patient_features = self.base_model(patient_images)
            aggregated_feature = patient_features.mean(dim=0)
            aggregated_features_list.append(aggregated_feature)
        
        aggregated_features = torch.stack(aggregated_features_list, dim=0)
        #print(aggregated_features.unsqueeze(2).shape)
        #[12, 512, 1]
        x = self.c1(aggregated_features.unsqueeze(2))
        x = self.c2(x)
        x = self.c3(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.output(x)
        return x


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
            
            
# Setup device

def main(args):
    # Your existing code goes here, but replace hardcoded hyperparameters 
    # and other arguments with the values from `args`
    
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    BASE_MODEL_NAME = args.base_model_name
    CHECKPOINT_PATH = args.checkpoint_file
    RESULTS = args.results_dir
    slides_dir = args.slides_dir
    gene_expression_file = args.gene_expression_file
    
    test_data = pd.read_csv(args.test_patient_id, index_col='patient_id')
    test_patient_ids = test_data.index.astype(str).tolist()
    
   
    FILENAME_ACROSS_PATIENT = RESULTS + "test_result_across_patient.csv"
    FILENAME_ACROSS_GENE = RESULTS + "test_result_across_gene.csv"
   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    base_model, num_features, preprocess = get_base_model_and_transforms(BASE_MODEL_NAME)
    
    
    color_normalization = MacenkoColorNormalization()
    transform = preprocess
    
    
    test_dataset = PatientDataset(slides_dir, test_patient_ids, gene_expression_file, transform=transform, color_normalization=color_normalization)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    
    # Initialize model, criterion and optimizer
    model = hist2RNA(base_model, input_features = num_features, batch_size=BATCH_SIZE).to(device)
    
    
    # Load the saved model weights
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint, strict=True)
    
    print("Model Loaded Successfully")
#    if torch.cuda.device_count() > 1:
#        print("Multiple GPU Detected")
#        print(f"Using {torch.cuda.device_count()} GPUs")
#        model = nn.DataParallel(model)
    
    # Ensure model is in evaluation mode and on the correct device
    
    model.eval()
    
    all_predictions = []
    all_true_values = []
    
    # Make predictions on the test dataset
    with torch.no_grad():
        for images, gene_expression in tqdm(test_dataloader, desc="Predicting"):
            # images = images.to(device)
            images = tuple(batch.to(device) for batch in images)
            gene_expression = tuple(tensor.to(device) for tensor in gene_expression)
            gene_expression = torch.stack(gene_expression, dim=0)
            outputs = model(images)
            all_predictions.append(outputs.cpu().numpy())
            all_true_values.append(gene_expression.cpu().numpy())

    
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
    df_predicted.to_csv(RESULTS+'predicted_gene_expressions.csv')
    df_true.to_csv(RESULTS+'true_gene_expressions.csv')


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
    
    # Save patient data
    save_to_csv(FILENAME_ACROSS_PATIENT, 
                ["Spearman correlation coef across each patient", "P-values across each patient"],
                spearman_coeffs_per_patient, 
                p_values_per_patient,
                details)
    
    # Save gene data
    save_to_csv(FILENAME_ACROSS_GENE, 
                ["Spearman correlation coef across each gene", "P-values across each gene"],
                spearman_coeffs_per_gene, 
                p_values_per_gene,
                details)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training Script')
    
    
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight Decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of Epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch Size')
    parser.add_argument('--base_model_name', type=str, default="resnet50", help='Base Model Name')   
    parser.add_argument('--checkpoint_file', default='./saved_model/best_model.pth', help='Path to save the model checkpoint in .pth')
    parser.add_argument('--results_dir', default="./saved_result/", help='Location for saving details and results')
    parser.add_argument('--slides_dir', default="./data/raw_wsi_tcga_images/", help="Directory for slides")
    parser.add_argument('--gene_expression_file', default="./data/gene_expression_file/pam50_gene_expression.csv", help="Path to the gene expression file in .csv")
    parser.add_argument('--test_patient_id', default="./data/patient_details/test_patient_list.txt", help="Path to the patient id file in .txt")

    
    args = parser.parse_args()
    
    main(args)
    
