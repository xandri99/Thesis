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

def simple_preprocessing():
    # Step 1: Filter out genes with low expression across all samples
    # Threshold set so that a gene needs to be expressed TPM > 1 in at least 10% of samples## Remove genes with no variation across samples (0 vectors) 
    cleaned_dataset = raw_data.loc[:, (raw_data != 0).any(axis=0)] # Actually droping columns with 0 variation

    # Also print how many genes have been removed in this steo
    num_genes_removed = raw_data.shape[1] - cleaned_dataset.shape[1]
    print(f"{BOLD}{WARNING}{num_genes_removed} genes were removed due to having 0 variation across samples...{ENDC}")
    
    return cleaned_dataset



