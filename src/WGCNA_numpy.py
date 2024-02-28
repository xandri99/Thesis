"""
WGCNA Implementation Using Numpy. All methods are commented callable functions, for an easier 
implementation.

"""


def preprocessing(number: float) -> float:
    """
    Step 1: Data Preprocessing (Normalization)

    Preprocessing: removing obvious outlier on genes and samples
    
    """
    print(f"{BOLD}{OKBLUE}Pre-processing...{ENDC}")



    ## Prepare and clean data
    # Remove genes expressed under this cutoff number along samples - use clustering of samples



    ## Remove genes with no variation across samples (0 vectors) 
    transcriptomics_dataset_filtered = transcriptomics_dataset.loc[:, (transcriptomics_dataset != 0).any(axis=0)] # Actually droping columns with 0 variation

    # Also print how many genes have been removed in this steo
    num_genes_removed = transcriptomics_dataset.shape[1] - transcriptomics_dataset_filtered.shape[1]
    print(f"{BOLD}{WARNING}{num_genes_removed} genes were removed due to having 0 variation across samples...{ENDC}")



    # NOTES: Maybe no onlyt genes with no variation, but small variation.




    print(f"{BOLD}{OKBLUE}Done...{ENDC}")
