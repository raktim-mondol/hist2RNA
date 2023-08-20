
import pandas as pd
import numpy as np



main_gene_file = pd.read_csv('./data/gene_expression/main_gene_expression.csv')
# Transpose to make sample as row and features as column
main_gene_file =pd.DataFrame.transpose(main_gene_file)
# Column as Gene Name
main_gene_file = main_gene_file.rename(columns=main_gene_file.iloc[0,:])
# Remove Un-necessary two column (eg. entrez id and gene symbol)
main_gene_file =main_gene_file.drop(["Entrez_Gene_Id", "Hugo_Symbol"],axis=0)
# Remove genes that have zero value among all patients
main_gene_file = main_gene_file.loc[:, (main_gene_file != 0).any(axis=0)]



#all_genes_from_nanostring = pd.read_csv('./gene_expression_file/gene_combined_list.txt', index_col = 'gene_combined')
#selected_gene = pd.DataFrame(main_gene_file[all_genes_from_nanostring.index])


pam50_gene = pd.read_csv('./data/gene_expression/pam50_gene_list.txt', index_col = 'pam_50_gene')
selected_gene = pd.DataFrame(main_gene_file[pam50_gene.index])


selected_gene.to_csv('./data/gene_expression/pam50_gene_expression.csv', index=True)
