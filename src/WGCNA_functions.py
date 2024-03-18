"""
WGCNA Implementation Using Numpy. 
All methods are commented callable functions, for an easier use.

"""

################################################################################################
#                                        IMPORTS
################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import gc

from statsmodels.formula.api import ols

import scipy.cluster.hierarchy as scipy_hierarchy
from scipy.spatial.distance import squareform
from scipy import stats

from sklearn.decomposition import PCA

################################################################################################





################################################################################################
#                                       CONFIGURATIONS
################################################################################################

# Settings for printing dataframes
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 1000)

# Colors for the terminal outputs
ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

OKBLUE = "\033[94m"
OKGREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"

################################################################################################





################################################################################################
#                                       PREPROCESSING
################################################################################################

def simple_preprocess():
    # Step 1: Filter out genes with low expression across all samples
    # Threshold set so that a gene needs to be expressed TPM > 1 in at least 10% of samples## Remove genes with no variation across samples (0 vectors) 
    cleaned_dataset = raw_data.loc[:, (raw_data != 0).any(axis=0)] # Actually droping columns with 0 variation

    # Also print how many genes have been removed in this steo
    num_genes_removed = raw_data.shape[1] - cleaned_dataset.shape[1]
    print(f"{BOLD}{WARNING}{num_genes_removed} genes were removed due to having 0 variation across samples...{ENDC}")
    
    return cleaned_dataset


## Prepare and clean data
def preprocess_TPM(raw_data):
    '''
    
    
    '''
    # Step 1: Filter out genes with low expression across all samples
    # Threshold set so that a gene needs to be expressed TPM > 1 in at least 10% of samples
    expression_th = 1
    cleaned_dataset = raw_data.loc[:, (raw_data > expression_th).any(axis=0)].copy()
    
    # Step 2: Log transformation
    cleaned_dataset.iloc[:, 1:] = np.log2(cleaned_dataset.iloc[:, 1:] + 1)
    
    return cleaned_dataset


## Prepare and clean data
def preprocess_TPM_outlier_deletion(raw_data):
    '''
    
    
    '''
    # Step 1: Filter out genes with low expression across all samples
    # Threshold set so that a gene needs to be expressed TPM > 1 in at least 10% of samples
    expression_th = 1
    cleaned_dataset = raw_data.loc[:, (raw_data > expression_th).any(axis=0)].copy()
    
    # Step 2: Log transformation
    cleaned_dataset.iloc[:, 1:] = np.log2(cleaned_dataset.iloc[:, 1:] + 1)
    
    # Step 3: Outlier detection and removal based on PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(cleaned_dataset.iloc[:, 1:].T)  # Transpose to have samples as rows for PCA
    z_scores = np.abs(stats.zscore(pca_result, axis=0))
    good_samples = (z_scores < 3).all(axis=1)  # Keeping samples within 3 standard deviations
    cleaned_dataset = cleaned_dataset.iloc[:, [True] + good_samples.tolist()]  # True for gene identifiers column
    
    # Step 7: Data Standardization (Z-score normalization)
    cleaned_dataset.iloc[:, 1:] = cleaned_dataset.iloc[:, 1:].apply(stats.zscore, axis=1)

    
    return cleaned_dataset

    
# Plotting PCA function for visualization
def plot_pca(dataframe, title='PCA before and after preprocessing', ax=None):
    """
    Performs PCA on the provided dataframe and plots the first two principal components.
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(dataframe.iloc[:, 1:].T)
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
