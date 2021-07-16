seed=75
import numpy as np
from tensorflow import set_random_seed 
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interp
from itertools import cycle
from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import average_precision_score, precision_recall_curve, matthews_corrcoef, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, roc_auc_score, auc, cohen_kappa_score, precision_recall_curve, log_loss, roc_curve, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, cross_val_predict, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics.classification import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn import model_selection
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, MiniBatchSparsePCA, NMF, TruncatedSVD, FastICA, FactorAnalysis, LatentDirichletAllocation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import  Normalizer, MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, label_binarize, QuantileTransformer
from keras.initializers import RandomNormal
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from variational_autoencoder import *
matplotlib.use('Agg')
np.random.seed(seed)


vae_model_single('./feature_extraction/VAE/'+aaenum+'/',x_train.shape[1],
       					x_train,x_test,intermediate_dim=1000,batch_size=20,latent_dim=50,epochs=100)
####### don't use the following lines when autoencoder requires fine tuning
model = load_model('./feature_extraction/VAE/'+aaenum+'/vae_encoder'+'.h5')
x_train = model.predict(x_train)
print('X_Train Shape after VAE :', x_train.shape)
x_test = model.predict(x_test)








       print('X_Test Shape after VAE :', x_test.shape)
