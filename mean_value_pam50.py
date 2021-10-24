#!/usr/bin/env python3
__author__ ="Raktim Kumar Mondol"

####################################################################################################
###################################### Import Required Lirary ######################################
####################################################################################################
seed=42
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
TMPDIR=os.environ["PBS_JOBFS"]

import gc
#import keras 
import tensorflow as tf
import tensorflow

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation, Multiply, Add, Lambda
from tensorflow.keras import initializers
from rnn_model import *

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Concatenate, Dropout
from tensorflow.keras.layers import BatchNormalization
#from tensorflow.compat.v1 import set_random_seed 
from tensorflow.keras.layers import Lambda, Input
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply
from scipy.stats import kendalltau
from attention_layers import Mil_Attention, Last_Sigmoid
from tensorflow.keras.regularizers import l2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
np.random.seed(seed)
#import seaborn as sns   0.9.0
from scipy import interp
from itertools import cycle
#from xgboost import XGBClassifier
from scipy import spatial
from numpy import dot
import math
from scipy import spatial
from numpy.linalg import norm
from collections import Counter
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve, matthews_corrcoef, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, roc_auc_score, auc, cohen_kappa_score, precision_recall_curve, log_loss, roc_curve, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, cross_val_predict, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn import model_selection
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, MiniBatchSparsePCA, NMF, TruncatedSVD, FastICA, FactorAnalysis, LatentDirichletAllocation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import  Normalizer, MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, label_binarize, QuantileTransformer
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
#from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
#from variational_autoencoder import *
#from variational_autoencoder import vae_model
#from variational_autoencoder import sampling
#from aae_architechture_proposed import *
#from deep_autoencoder import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
#from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE 
#from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.preprocessing import  Normalizer, MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, label_binarize, QuantileTransformer

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.applications import DenseNet201

from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications import NASNetMobile

from tensorflow.keras.applications import Xception

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2

import tensorflow
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Reshape, Permute

from tensorflow.keras import regularizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import DenseNet201
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import hickle as hkl
print("I am using Keras version: ", tf.keras.__version__)


################################  Highway Layer  ##################################################

def highway_layer(value, activation="tanh", transform_gate_bias=-1.0):
    dim = K.int_shape(value)[-1]
    transform_gate_bias_initializer = tensorflow.keras.initializers.Constant(transform_gate_bias)
    transform_gate = Dense(units=dim, bias_initializer=transform_gate_bias_initializer)(value)
    transform_gate = Activation("sigmoid")(transform_gate)
    carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(transform_gate)
    transformed_data = Dense(units=dim)(value)
    transformed_data = Activation(activation)(transformed_data)
    transformed_gated = Multiply()([transform_gate, transformed_data])
    identity_gated = Multiply()([carry_gate, value])
    value = Add()([transformed_gated, identity_gated])
    return value

#####################################################################################################
################################# Import Data and pre-processing  ###################################
#####################################################################################################
#Read the file
main_gene_file = pd.read_csv('/g/data/nk53/rm8989/data/tcga-gene/gene_expression.csv')
# Transpose to make sample as row and features as column
main_gene_file =pd.DataFrame.transpose(main_gene_file)
# Column as Gene Name
main_gene_file = main_gene_file.rename(columns=main_gene_file.iloc[0,:])
# Remove Un-necessary two column (eg. entrez id and gene symbol)
main_gene_file =main_gene_file.drop(["Entrez_Gene_Id", "Hugo_Symbol"],axis=0)
# Remove genes that have zero value among all patients
main_gene_file = main_gene_file.loc[:, (main_gene_file != 0).any(axis=0)]

# load patient list
test_patient_list = pd.read_csv('/g/data/nk53/rm8989/data/tcga-gene/test_patient_list.txt', index_col = 'test_patient_list')
train_patient_list = pd.read_csv('/g/data/nk53/rm8989/data/tcga-gene/train_patient_list.txt', index_col = 'train_patient_list')
# make it as index form
test_patient_index = test_patient_list.index
train_patient_index = train_patient_list.index


pam50_gene = pd.read_csv('/g/data/nk53/rm8989/data/tcga-gene/pam50_gene.txt', index_col = 'pam_50_gene')
# "['ORC6L', 'CDCA1', 'KNTC2'] not in index"
#following lines selevted only specified columns
selected_pam50_gene = main_gene_file[pam50_gene.index]
#this could be used to save pam50 gene file only once not required for this code
#selected_pam50_gene.to_csv('/g/data/nk53/data/rm8989/tcga-gene/pam50_gene_expression.csv')

# select 50 Test 
x_train = selected_pam50_gene.loc[train_patient_index,:]
x_test = selected_pam50_gene.loc[test_patient_index,:]

###############################################################################
###################### Normalizing Data using Log(1+x)#########################
###############################################################################
transformer = FunctionTransformer(np.log1p, validate=True)

x_train_log = transformer.transform(x_train).astype('float32')
x_test_log = transformer.transform(x_test).astype('float32')


x_train_df = pd.DataFrame(data=x_train_log, index=x_train.index, columns=x_train.columns)
x_test_df = pd.DataFrame(data=x_test_log, index=x_test.index, columns=x_test.columns)

###############################################################################
# Normalizing Data using Quantile Transformer/MinMax Scaler
###############################################################################

#qt = QuantileTransformer(n_quantiles=1050, random_state=seed)
#qt.fit(main_data)
#main_data_normalized = qt.transform(main_data)

#minmax_scaler=MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
#minmax_scaler.fit(main_data)
#main_data_normalized = minmax_scaler.transform(main_data)


#convert main_data_normalized to datafram
#main_data_normalized_dataframe = pd.DataFrame(data=main_data_normalized, index=main_data.index, columns=main_data.columns)
#
## contain only 100 patients
#train_gene_file = main_data.loc[train_patient_index,:]
#x_train = main_data_normalized_dataframe.loc[train_patient_index,:]
#
## only contain 50 patients for test
#test_gene_file = main_gene_file.loc[test_patient_index,:]
#x_test = minmax_scaler.transform(test_gene_file)

#######################################################################################################
########################################    Feature Extractor     #####################################
#######################################################################################################
SIZE = 224  #Resize images

#Capture training data and labels into respective lists
train_image = []
train_label = [] 
train_gene_label=[]

test_image = []
test_label=[]
test_gene_label=[]


#TMPDIR='/g/data/nk53/rm8989'
#################################   Read Train Image  ####################################
for directory_path in glob.glob(TMPDIR+"/tcga-gc/*"):
    label = directory_path.split("/")[-1]
    #print(label)
    #label = label.split("_")[-1]
    label=label.split('-')[0]+'-'+label.split('-')[1]+'-'+label.split('-')[2]+'-'+'01'
    #print(label)
    label_flag=False
    for i in range(0,100):
        if(label==train_patient_list.index[i]):
            label_flag=True
        else:
            pass
    if(label_flag==True):
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            #print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_pam50_gene = x_train_df.loc[label]
            train_gene_label.append(train_pam50_gene)
            train_image.append(img)
            train_label.append(label)

print("Finished image loading")
#Convert lists to arrays        

#print("Memory size of a NumPy array:", train_image.nbytes)
#hkl.dump(train_image, '/srv/scratch/z5342745/Data/train_image_gzip.hkl', mode='w', compression='gzip')
######################
#del train_image_temp 
#n = gc.collect()
#print("Number of unreachable objects collected by GC:", n)
#print("Uncollectable garbage:", gc.garbage)
######################
#train_image = hkl.load('/srv/scratch/z5342745/Data/train_image_gzip.hkl')
#print('compressed:   %i bytes' % os.path.getsize('/srv/scratch/z5342745/Data/train_image_gzip.hkl'))

train_image = np.array(train_image, dtype='int8')
train_image = (train_image/255.).astype(np.float32)
print('Training image shape is: ', train_image.shape)

#Encode labels from text to integers.

#print(train_label)
#hkl.dump(train_label, '/srv/scratch/z5342745/Data/train_label.hkl', mode='w')
#train_label = hkl.load('/srv/scratch/z5342745/Data/train_label.hkl')
train_label = np.array(train_label)
print(train_label)
print("Everything loaded")
#########################################   Read Test Image   ################################

for directory_path in glob.glob(TMPDIR+"/tcga-gc/*"):
    
    label = directory_path.split("/")[-1]
    #print(label)
    #label = label.split("_")[-1]
    label=label.split('-')[0]+'-'+label.split('-')[1]+'-'+label.split('-')[2]+'-'+'01'
    #print(label)
    label_flag=False
    #### Check if patient from the test patient list #####
    for i in range(0,50):
        if(label==test_patient_list.index[i]):
            label_flag=True
        else:
            pass
    if(label_flag==True):
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            test_pam50_gene = x_test_df.loc[label]
            test_gene_label.append(test_pam50_gene)
            test_image.append(img)
            test_label.append(label)

#Convert lists to arrays        
print("Test image loaded")
#print("Memory size of a NumPy array:", test_image.nbytes)
#hkl.dump(test_image, '/srv/scratch/z5342745/Data/test_image_gzip.hkl', mode='w', compression='gzip')
##########################################################
#del test_image_temp 
#n = gc.collect()
#print("Number of unreachable objects collected by GC:", n)
#print("Uncollectable garbage:", gc.garbage)
##########################################################
#print('compressed:   %i bytes' % os.path.getsize('/srv/scratch/z5342745/Data/test_image_gzip.hkl'))
#test_image = hkl.load('/srv/scratch/z5342745/Data/test_image_gzip.hkl')
test_image = np.array(test_image, dtype='int8')
test_image = (test_image/255.).astype(np.float32)

print('Testing image shape is: ', test_image.shape)

########## for label  ############

#hkl.dump(test_label, '/srv/scratch/z5342745/Data/test_label.hkl', mode='w')
#test_label = hkl.load('/srv/scratch/z5342745/Data/test_label.hkl')
test_label = np.array(test_label)
print(test_label)

#################### Label Encoder and One-Hot Encoding #########################
#################### Label Encoder and One-Hot Encoding #########################
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_label)
test_label_encoded = le.transform(test_label)
print(test_label_encoded)

from keras.utils import to_categorical
test_label_one_hot = to_categorical(test_label_encoded)

#################################   Regression Label ###################################

train_gene_regn_label = np.array(train_gene_label)
test_gene_regn_label = np.array(test_gene_label)


##############################    Feature Extractor using Pretrained Model (CNN)  #########################

def outer_product(x):
	"""
	x list of 2 tensors, assuming each of which has shape = (size_minibatch, total_pixels, size_filter)
	"""
	return tensorflow.keras.backend.batch_dot(x[0], x[1], axes=[1,1]) / x[0].get_shape().as_list()[1] 
	#return tf.einsum('bom,bpm->bmop', x[0], x[1])
    


#Load model wothout classifier/fully connected layers
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
ResNet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
#Xception_model = Xception(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
#Xception_model = Xception(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
#DenseNet201= DenseNet201(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
#base_model = ResNet_model
#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
#for layer in VGG_model.layers:
#    layer.trainable = False
##Trainable parameters will be 0
#for layer in base_model.layers[:5]:
#    layer.trainable = False
for layer in ResNet_model.layers:
    layer.trainable = False
#ResNet_model.summary()

#for layer in Xception_model.layers:
#    layer.trainable = False
#Xception_model.summary()

for layer in VGG_model.layers:
    layer.trainable = False
    
#VGG_model.summary()
#
feature_1= VGG_model.output
shape_feature_1 = VGG_model.output_shape
x_detector_1 = Reshape((shape_feature_1[1]*shape_feature_1[2], shape_feature_1[3]))(feature_1)



#feature_1= Xception_model.output
#shape_feature_1 = Xception_model.output_shape
#x_detector_1 = Reshape((shape_feature_1[1]*shape_feature_1[2], shape_feature_1[3]))(feature_1)
##Xception_model.summary()


feature_2= ResNet_model.output
shape_feature_2 = ResNet_model.output_shape
x_detector_2 = Reshape((shape_feature_2[1]*shape_feature_2[2], shape_feature_2[3]))(feature_2)
#ResNet_model.summary()


bi_mul = tensorflow.keras.layers.Lambda(outer_product)([x_detector_1,x_detector_2])

#concat=tensorflow.keras.layers.Concatenate(axis=-1)([feature_1, feature_2])


#feature_extractor_1 = ResNet_model.predict(train_image)


#hkl.dump(feature_extractor_1, '/g/data/nk53/rm8989/results/gene_prediction/data/feature_extractor_1_gzip.hkl', mode='w', compression='gzip')
#####################
#del feature_extractor_1
#n = gc.collect()
#print("Number of unreachable objects collected by GC:", n)
#print("Uncollectable garbage:", gc.garbage)
#####################
#feature_extractor_1 = hkl.load('/g/data/nk53/rm8989/results/gene_prediction/data/feature_extractor_1_gzip.hkl')
#print('compressed:   %i bytes' % os.path.getsize('/g/data/nk53/rm8989/results/gene_prediction/data/feature_extractor_1_gzip.hkl'))



#feature_extractor_2 = Xception_model.predict(train_image)

#hkl.dump(feature_extractor_2, '/g/data/nk53/rm8989/results/gene_prediction/data/feature_extractor_2_gzip.hkl', mode='w', compression='gzip')
#####################
#del feature_extractor_2
#n = gc.collect()
#print("Number of unreachable objects collected by GC:", n)
#print("Uncollectable garbage:", gc.garbage)
#####################
#feature_extractor_2 = hkl.load('/g/data/nk53/rm8989/results/gene_prediction/data/feature_extractor_2_gzip.hkl')
#print('compressed:   %i bytes' % os.path.getsize('/g/data/nk53/rm8989/results/gene_prediction/data/feature_extractor_2_gzip.hkl'))

#features1 = feature_extractor_1.reshape(feature_extractor_1.shape[0], -1)
#features2 = feature_extractor_2.reshape(feature_extractor_2.shape[0], -1)

#print("Done with reshaping feature extractor")
#
#combine_feature = (features1,features2)
#combine_feature = np.hstack(combine_feature)
#for layer in VGG_model.layers[:11]:
#last_layer=VGG_model.get_layer('block3_pool')

#base_model= Model(input= base_model.input, output= last_layer.output)
#base_model.summary()  
#base_model = Xception_model
#base_model = ResNet_model
######################## Highway Network######################
#out=base_model.output
#
#hwy = highway_layer(out)
#hwy = highway_layer(hwy)
#hwy = highway_layer(hwy)
#hwy = highway_layer(hwy)
#hwy = highway_layer(hwy)


#base_with_highway=Model(inputs=base_model.input, outputs=hwy)
#base_with_highway.summary()

############################   fully connected layer ############################

### first layer must prvide input shape

#shape = tuple(bi_mul.shape.as_list()[1:])
x=Flatten()(bi_mul)
x = BatchNormalization(axis=-1)(x)
#because output of highway and base_model output are same
x=Dense(512, activation='relu')(x)
#kernel_regularizer=regularizers.l2(0.001)
#l2 squared value of the parameters smaller weights
#l1 absolute value of the parameter zero weights
x=Dropout(0.02)(x)
x=Dense(256, activation='relu')(x)
fc=Dropout(0.01)(x)


###################### Attention Layer #######################
#alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(1e-4), name='alpha', use_gated=False)(fc)
#x_mul = multiply([alpha, fc])

##################### Additional Layer ########################
#add_layer = Dense(128, activation='relu')(x_mul)
#add_layer = Dense(128, activation='relu')(add_layer)

################  Last regression Layer #######################
regn_out = Dense(47, activation='linear')(fc)

#model = Model(inputs=joint_model.input, outputs= regn_out)
hybrid_model = Model(inputs=[VGG_model.input, ResNet_model.input], outputs=regn_out)

#opt=Adam(learning_rate=0.0001)
hybrid_model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['mae','mse'])
#metrics=['mean_squared_error']
#metrics=['mean_squared_error']
#mean_absolute_error
#opt=Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999)
#optimizer='adam',
#RMSprop(learning_rate=0.001, rho=0.9)

################ Callbacks ######################

model_path='/g/data/nk53/rm8989/results/gene_prediction/concat_xception_resnet50_pam50.h5'

checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='loss', verbose=0, save_best_only=True,save_weights_only=True)
earlystop = tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss')

hybrid_model.summary()
print('I am ifit')

#THIS IS FIT 
#history = hybrid_model.fit([train_image, train_image], train_gene_regn_label, batch_size=28, epochs=300, callbacks=[checkpoint, earlystop])

################# Graph for Training ############
#plt.plot(history.history['loss'])
##plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train'], loc='upper left')
##plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('/g/data/nk53/rm8989/results/gene_prediction/figures/training_plot.png', dpi=150)


#### this is not
#model.save('/home/z5342745/result/gene_prediction/pam50_vgg_model_attention.h5')




#### This is working 
#model.save_weights('/g/data/nk53/rm8989/results/gene_prediction/xeption_highway_pam50.h5')




#g=VGG_model.get_layer('block5_conv3')
#g.trainable=True

#for layer in ResNet_model.layers:
#    layer.trainable = False
##Trainable parameters will be 0 
#ResNet_model.summary()  


#model = load_model('/home/z5342745/result/gene_prediction/pam50_vgg_model_attention.h5', compile=False)

hybrid_model.load_weights('/g/data/nk53/rm8989/results/gene_prediction/concat_vgg16_resnet50_pam50.h5')

######################## Make Test data similar to Train Data#############
#feature_extractor_1 = ResNet_model.predict(test_image)
#
#feature_extractor_2 = Xception_model.predict(test_image)
#
#features1 = feature_extractor_1.reshape(feature_extractor_1.shape[0], -1)
#features2 = feature_extractor_2.reshape(feature_extractor_2.shape[0], -1)
#
#combine_feature = (features1,features2)
#combine_feature = np.hstack(combine_feature)

###################################################################


test_gene_predicted = hybrid_model.predict([test_image,test_image])

#test_gene_predicted_back  = np.expm1(test_gene_predicted)
test_gene_predicted_df = pd.DataFrame(test_gene_predicted)
test_gene_predicted_df.to_csv("/g/data/nk53/rm8989/results/gene_prediction/test_gene_predicted.csv")


test_gene_regn_label_df = pd.DataFrame(test_gene_regn_label)
#test_gene_regn_label_df_back  = np.expm1(test_gene_regn_label_df)
test_gene_regn_label_df.to_csv("/g/data/nk53/rm8989/results/gene_prediction/test_gene_actual.csv")


#############################################################################
############################################################################
#################  Mean value of all gene values for all patch prediction  
#############################################################################
#############################################################################

accumulate=[]
#shape = p.shape[0]
start_index=0 
end_index=0
# #10 total number of samples
for i in range(0,50):
    label_index =test_label_encoded[test_label_encoded==i]
    label_index =label_index.shape[0]
    #number of patch for each label
    #print(end_index)
    
    end_index=label_index + end_index
    mean=test_gene_predicted_df.iloc[start_index:end_index,:].mean(axis=0)
    #axis=0 means accross column which is gene NOT samples
    #end_index = end_index + end_index
    start_index=end_index
    #print(start_index)
    accumulate.append(mean)
    
predicted_mean_test_data=np.array(accumulate)
#ccc=train_gene_regn_label_df.iloc[0:165,:].sum(axis=0)
test_gene_predicted = predicted_mean_test_data

# true test data for 50 samples
true_test_data=np.array(x_test_df).astype(np.float32)
test_gene_regn_label =true_test_data
#MSE = mean_squared_error(test_gene_regn_label, test_gene_predicted)
#RMSE = mean_squared_error(test_gene_regn_label, test_gene_predicted, squared=False)
MAE =  mean_absolute_error(test_gene_regn_label, test_gene_predicted)

#print("Mean Squared Error (MSE) is:", MSE)
print("Mean Absolute Error (MAE) is:", MAE)
#print("Root Mean Squared Error (RMSE) is:", RMSE)


#coefficient of determination
R2_score=r2_score(test_gene_regn_label, test_gene_predicted)
multioutput='variance_weighted'
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
print("R2(coefficient of determination)_Regression_score is:", R2_score )


###########################################################################################
################################## calculate spearman's correlation ######################
############################################################################################
result=[]
r2score=[]
count=0
pvalue=[]
for i in range(0,47):

    R2 = r2_score(test_gene_regn_label[:,i], test_gene_predicted[:,i])
    print('R2(coefficient of determination) Regression_score is ', R2)
    r2score.append(R2)
    ####################################################################
    coef, p = spearmanr(test_gene_regn_label[:,i],test_gene_predicted[:,i])
    print('Spearmans correlation coefficient: %.3f' % coef)
    print('Spearmans p-value: %.3f' % p)
    result.append(coef)
    
    pvalue.append(p)
    
    # interpret the significance
    
    alpha = 0.05
    if p > alpha:
      count=count+1
      print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
     
    else:
      print('Samples are correlated (reject H0) p=%.3f' % p)
#        
df = pd.DataFrame(result)
df.to_csv("/g/data/nk53/rm8989/results/gene_prediction/spearman_correlation_result_pam50_accross_gene.csv")

df2 = pd.DataFrame(r2score)
df2.to_csv("/g/data/nk53/rm8989/results/gene_prediction/r2_score_result_pam50_accross_gene.csv")

df3 = pd.DataFrame(pvalue)
df3.to_csv("/g/data/nk53/rm8989/results/gene_prediction/pvalue_spearman_pam50_accross_gene.csv")

print("Total number of uncorrelated sample among 16235 sample is: ", count)
print('Actual Test label shape', test_gene_regn_label.shape)
print('Predicted label shape', test_gene_predicted.shape)
print('#######################################################################')
###########################################################################################
############################## calculate Pearson's correlation  ############################
############################################################################################
#result=[]
#cnt=0
##16235
#for i in range(0,47):
#    
#    coef, p = pearsonr(test_gene_regn_label[:,i],test_gene_predicted[:,i])
#    print('Pearsons correlation coefficient: %.3f' % coef)
#    result.append(coef)
#    # interpret the significance
#    
#    alpha = 0.05
#    if p > alpha:
#      cnt = cnt +1
#      print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
#    else:
#    	print('Samples are correlated (reject H0) p=%.3f' % p)
#        
#df = pd.DataFrame(result)
#df.to_csv("/home/z5342745/result/gene_prediction/pearsons_correlation_result_pam50.csv")
#print("Pearson number of uncorrelated sample among 16235 sample is: ", cnt)
#print("Spearmann number of uncorrelated sample among 16235 sample is: ", count)
###########################################################################################
###########################         Cosine Similary       ##################################
############################################################################################

#cos_sim = 1 - spatial.distance.cosine(test_gene_regn_label, test_gene_predicted)
#print("Cosine Similarity:", cos_sim)
#
#
#############################################################################################
#########################  Angular Similarity ###############################################
############################################################################################
#
#angular_similarity = 1 - math.acos(cos_sim) / math.pi
#print("Angular Similarity:", angular_similarity)
############################################################################################
#Total number of uncorrelated sample among 16235 sample is:  1749 then 980 (attention0/  2264 when gated
#2237 when attention without gated epoch 120 batch 84




#######################    TESTING WITH DECISION TREE   #########################
#fc_layer=model.get_layer('sequential_1')
#
#test_model = Model(input= model.input, output= fc_layer.get_output_at(-1))
#
##################################    X TRAIN  ##############################
#feature_extractor = test_model.predict(train_image)
##feature_extractor_2 = ResNet_model.predict(train_image)
#
#features_1 = feature_extractor.reshape(feature_extractor.shape[0], -1)
##features_2 = feature_extractor_2.reshape(feature_extractor_2.shape[0], -1)
#
##combine_feature = (features_1,features_2)
##combined_feature = np.hstack(combine_feature)
#combined_feature = features_1
##Now, let us use features from convolutional network for RF
#train_image_for_DT = combined_feature #This is our X input to DT
#
#
##################################   X TEST ################################
#feature_extractor_test = test_model.predict(test_image)
##feature_extractor_test_2 = ResNet_model.predict(test_image)
#
#features_test_1 = feature_extractor_test.reshape(feature_extractor_test.shape[0], -1)
##features_test_2 = feature_extractor_test_2.reshape(feature_extractor_test_2.shape[0], -1)
#
##combine_test_feature = (features_test_1, features_test_2)
##combined_test_feature = np.hstack(combine_test_feature)
#combined_test_feature = features_test_1
##Now, let us use features from convolutional network for RF
#test_image_for_DT = combined_test_feature #This is our X input to DT
#
##################################   Regression Part  ###################################
##train_gene_regn_label = np.array(train_gene_label)
##test_gene_regn_label = np.array(test_gene_label)
#
#model = DecisionTreeRegressor()
##model = RandomForestRegressor(n_estimators=1000, max_depth=2, random_state=seed)
#model.fit(train_image_for_DT, train_gene_regn_label)
#
##for single prediction
##x = X_for_RF[125]
##x = x.reshape(1,-1)
##yhat = model.predict(x)
##y=regression_label[125]
##y=y.reshape(1,-1)
##MSE = mean_squared_error(y, yhat)
#test_gene_predicted = model.predict(test_image_for_DT)
#
#MSE = mean_squared_error(test_gene_regn_label, test_gene_predicted)
#R2_score=r2_score(test_gene_regn_label, test_gene_predicted)
##multioutput='variance_weighted'
#print("Mean squared error is:", MSE)
#print("R2_score is:", R2_score )
#
#temp = pd.DataFrame(test_gene_predicted)
#test_gene_predicted_encoded = temp.set_index(test_label)
#test_gene_predicted_encoded.to_csv("/home/z5342745/result/gene_prediction/test_gene_predicted_encoded.csv")  
#
#
############################################################################################
################################### calculate spearman's correlation ######################
#############################################################################################
#result=[]
#count=0
#for i in range(0,16235):
#    
#    coef, p = spearmanr(test_gene_regn_label[i],test_gene_predicted[i])
#    print('Spearmans correlation coefficient: %.3f' % coef)
#    result.append(coef)
#    # interpret the significance
#    
#    alpha = 0.05
#    if p > alpha:
#      count=count+1
#      print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
#     
#    else:
#      print('Samples are correlated (reject H0) p=%.3f' % p)
##        
#df = pd.DataFrame(result)
#df.to_csv("/home/z5342745/result/gene_prediction/spearman_correlation_result_pam50.csv")
#print("Total number of uncorrelated sample among 16235 sample is: ", count)
#
#print('#######################################################################')
#print('#######################################################################')
#print('#######################################################################')
############################################################################################
############################### calculate Pearson's correlation  ############################
#############################################################################################
#result=[]
#cnt=0
##16235
#for i in range(0,16235):
#    
#    coef, p = pearsonr(test_gene_regn_label[i],test_gene_predicted[i])
#    print('Pearsons correlation coefficient: %.3f' % coef)
#    result.append(coef)
#    # interpret the significance
#    
#    alpha = 0.05
#    if p > alpha:
#      cnt = cnt +1
#      print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
#    else:
#    	print('Samples are correlated (reject H0) p=%.3f' % p)
#        
#df = pd.DataFrame(result)
#df.to_csv("/home/z5342745/result/gene_prediction/pearsons_correlation_result_pam50.csv")
#print("Pearson number of uncorrelated sample among 16235 sample is: ", cnt)
#print("Spearmann number of uncorrelated sample among 16235 sample is: ", count)


