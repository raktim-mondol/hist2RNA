import pandas as pd
import numpy as np


main_gene_file = pd.read_csv('gene_expression.csv')
main_gene_file =pd.DataFrame.transpose(main_gene_file)
main_gene_file = main_gene_file.rename(columns=main_gene_file.iloc[0,:])
main_gene_file=main_gene_file.drop(["Entrez_Gene_Id", "Hugo_Symbol"],axis=0)

selected_patient_list = pd.read_csv('list_of_patient.txt', index_col = 'patient_list')


selected_index=selected_patient_list.index

selected_gene_file =main_gene_file.loc[selected_index,:]

selected_gene_file_transposed =pd.DataFrame.transpose(selected_gene_file)
selected_gene_file_transposed.to_csv("output.csv")



seed=42
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE 
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.preprocessing import  Normalizer, MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, label_binarize, QuantileTransformer

qt = QuantileTransformer(n_quantiles=10, random_state=seed)
qt.fit(selected_gene_file)
selected_gene_file_normalized=qt.transform(selected_gene_file)

#y=np.ones(150)
#X_train, X_test, y_train, y_test = train_test_split(selected_gene_file_normalized, y, test_size=0.33, random_state=42)
selected_gene_file_normalized = pd.DataFrame(data=selected_gene_file_normalized, index=selected_index, columns=main_gene_file.columns)
selected_gene_file_normalized_transposed =pd.DataFrame.transpose(selected_gene_file_normalized)
selected_gene_file_normalized_transposed.to_csv("output_normalized.csv")


gene_file = pd.read_csv('output_normalized.csv', index_col = 'name')
data=pd.DataFrame.transpose(gene_file)

train_data=data.iloc[0:100,:]
test_data=data.iloc[100:150,:]
#go=train_data.loc['TCGA-A1-A0SK-01']

#img_path = glob.glob(each_path + '/*.jpg')













