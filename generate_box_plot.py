
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import numpy as np
import seaborn as sns
import numpy as np
from sklearn.metrics import auc

# Need to load the saved result [generated after training]
FILENAME_ACROSS_GENE = './saved_results/test_result_across_gene.csv'
FILENAME_ACROSS_PATIENT = './saved_results/test_result_across_patient.csv'

save_directory = "./saved_results/figures/"

# Load the initial rows of the CSV to extract the model parameters
parameters_df = pd.read_csv(FILENAME_ACROSS_GENE, nrows=4, header=None, names=['Parameter', 'Value'])

# Convert the dataframe to a dictionary for easy access
parameters_dict = parameters_df.set_index('Parameter').to_dict()['Value']

# Define a function to generate the title with model parameters
def generate_title(base_title, parameters):
    params_str = ", ".join([f"{key}: {value}" for key, value in parameters.items()])
    return f"{base_title} ({params_str})"

def load_and_preprocess_data(mode):
    """Load and preprocess the data based on the mode."""
    if mode == "gene":
        FILENAME = FILENAME_ACROSS_GENE
        p_value_column = 'P-values across each gene'
    elif mode == "patient":
        FILENAME = FILENAME_ACROSS_PATIENT
        p_value_column = 'P-values across each patient'
    else:
        raise ValueError("Invalid mode specified. Choose either 'across_gene' or 'across_patient'.")
    
    data_df = pd.read_csv(FILENAME, skiprows=5)
    adjusted_p_values = multipletests(data_df[p_value_column], method='fdr_bh')[1]
    transformed_p_values = -np.log10(adjusted_p_values)
    data_df['Adjusted P-values'] = adjusted_p_values
    data_df['Transformed P-values'] = transformed_p_values
    data_df = data_df.dropna(subset=['Adjusted P-values', 'Transformed P-values'])
    
    return data_df

def generate_boxplot(data_df, mode, save_directory):
    """Generate and save box plots based on the provided mode (either 'gene' or 'patient')."""
    fig = plt.figure(figsize=(14, 10))
    
    # Calculate statistics for 'Spearman correlation coefficients'
    mean_value = data_df[f'Spearman correlation coef across each {mode}'].mean()
    std_value = data_df[f'Spearman correlation coef across each {mode}'].std()
    median_value = data_df[f'Spearman correlation coef across each {mode}'].median()

    # First subplot for 'Spearman correlation coefficients'
    ax1 = plt.subplot(2, 1, 1)
    sns.boxplot(data=data_df[f'Spearman correlation coef across each {mode}'], 
                orient='h', palette=["#2ecc71"], ax=ax1)
    ax1.set_title(generate_title(f'Box plot for Spearman correlation coefficients (Across {mode.capitalize()})', parameters_dict))
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Spearman Coefficients')

    # Annotate with vertical lines
    ax1.axvline(x=mean_value, color='#34495e', linestyle='--', label=f"Mean: {mean_value:.3f} \u00B1 {std_value:.3f}")
    #ax1.axvline(x=mean_value, color='#34495e', linestyle='--', label=f"Mean: {mean_value:.3f} ± {std_value:.3f}")
    ax1.axvline(x=median_value, color='#3498db', linestyle='-', label=f"Median: {median_value:.3f}")
    ax1.legend()
    

    # Second subplot for 'P-values'
    ax2 = plt.subplot(2, 1, 2)
    sns.boxplot(data=data_df['Transformed P-values'], 
                orient='h', palette=["#9b59b6"], ax=ax2)
    ax2.set_title(generate_title(f'Box plot for -log(P-values) (Across {mode.capitalize()})', parameters_dict))
    ax2.set_xlabel('Value')
    ax2.set_ylabel('-log(P-values)')

    # Saving the figure
    file_path = save_directory + f"combined_box_plot_across_{mode}.png"
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()




# Example usage:
data_df_gene = load_and_preprocess_data("gene")
#print(data_df_gene)
generate_boxplot(data_df_gene, "gene", save_directory)

# For patient data:
data_df_patient = load_and_preprocess_data("patient")
#print(data_df_patient)
generate_boxplot(data_df_patient, "patient", save_directory)



def compute_auc_of_reverse_cumulative_histogram(data_df, mode):
    spearman_coeffs = data_df[f'Spearman correlation coef across each {mode}']
    
    # Generate reverse cumulative histogram
    histogram, bin_edges = np.histogram(spearman_coeffs, bins=np.linspace(-1, 1, 1001), density=True)
    reverse_cumulative = np.cumsum(histogram[::-1])[::-1]
    reverse_cumulative /= reverse_cumulative[0]  # Normalize
    
    # Compute AUC
    bin_width = bin_edges[1] - bin_edges[0]
    auc = np.sum(reverse_cumulative) * bin_width
    
    # Normalize AUC to [0, 1]
    normalized_auc = auc / 2
    
    return normalized_auc, bin_edges[:-1], reverse_cumulative


def plot_combined_reverse_cumulative_histogram(data_df_gene, data_df_patient, save_directory):
    auc_value_gene, bin_centers_gene, reverse_cumulative_hist_gene = compute_auc_of_reverse_cumulative_histogram(data_df_gene, "gene")
    auc_value_patient, bin_centers_patient, reverse_cumulative_hist_patient = compute_auc_of_reverse_cumulative_histogram(data_df_patient, "patient")

    plt.figure(figsize=(12, 8))

    # Use Seaborn's color palette
    #palettes = ["deep", "muted", "bright", "pastel", "dark", "colorblind"]
    colors = sns.color_palette("dark")
    
    # Plotting data for 'gene'
    plt.plot(bin_centers_gene, reverse_cumulative_hist_gene, color=colors[0], linewidth=2.5, label=f"Across Gene (AUC: {auc_value_gene:.4f})")
    
    # Plotting data for 'patient'
    plt.plot(bin_centers_patient, reverse_cumulative_hist_patient, color=colors[1], linewidth=2.5, label=f"Across Patient (AUC: {auc_value_patient:.4f})")

    # Title including the model name with increased font size
    model_name = parameters_dict.get("Base Model Name", "Model Name")  # Assuming the model name is stored with the key "model" in parameters_dict
    title_str = f"{model_name} - Reverse Cumulative Histogram"
    plt.title(title_str, fontsize=16, pad=20)

    # Set grid, increase font size for axis labels, and set consistent color
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.xlabel('Spearman Correlation Coefficient', fontsize=14, color=colors[2])
    plt.ylabel('Reverse Cumulative Frequency', fontsize=14, color=colors[2])
    
    # Customize ticks' color and size
    plt.tick_params(colors=colors[2], which='both', labelsize=12)
    
    # Legend with AUC values for both 'gene' and 'patient'
    plt.legend(loc='upper right', edgecolor=colors[2])

    plt.tight_layout()

    file_path = save_directory + "combined_reverse_cumulative_histogram.png"
    plt.savefig(file_path, dpi=150)
    plt.close()

# Apply Seaborn style
sns.set_style("whitegrid")

# Example usage:
data_df_gene = load_and_preprocess_data("gene")
data_df_patient = load_and_preprocess_data("patient")

generate_boxplot(data_df_gene, "gene", save_directory)
generate_boxplot(data_df_patient, "patient", save_directory)

# Plot combined reverse cumulative histogram
plot_combined_reverse_cumulative_histogram(data_df_gene, data_df_patient, save_directory)


