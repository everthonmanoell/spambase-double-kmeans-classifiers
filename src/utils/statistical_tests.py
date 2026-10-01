import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

def perform_friedman_nemenyi(results_dict, metric_name, alpha=0.05):
    """
    Performs the Friedman test and the Nemenyi post-hoc test for the classifiers.

    Args:
        results_dict (dict): Dictionary where keys are classifier names and values 
                             are lists/arrays containing the metric results 
                             (e.g., 300 results from the 30x10-folds).
        metric_name (str): Name of the evaluated metric (used for logging).
        alpha (float): Significance level. Defaults to 0.05.
    
    Returns:
        pd.DataFrame: A matrix of p-values from the Nemenyi test (or None if the 
                      Friedman test is not significant).
    """
    classifiers = list(results_dict.keys())
    
    # Validate that all classifiers have the same number of evaluations (folds)
    n_folds = len(results_dict[classifiers[0]])
    for clf in classifiers:
        if len(results_dict[clf]) != n_folds:
            raise ValueError(f"Classifier {clf} has {len(results_dict[clf])} results, expected {n_folds}.")

    # Extract data in the same order as the keys
    data = [results_dict[clf] for clf in classifiers]

    print(f"\n{'='*40}")
    print(f"STATISTICAL ANALYSIS: {metric_name.upper()}")
    print(f"{'='*40}")

    # 1. Friedman Test
    stat, p_value = friedmanchisquare(*data)
    
    print(f"Friedman Test: Statistic = {stat:.4f}, p-value = {p_value:.4e}")

    if p_value < alpha:
        print(f"[RESULT] Null hypothesis rejected (p < {alpha}). Significant difference found between models.")
        
        # 2. Nemenyi Post-hoc Test
        print("\nExecuting Nemenyi Post-hoc Test...")
        
        # scikit-posthocs expects a DataFrame in a specific format or a transposed matrix
        data_matrix = np.array(data).T 
        df_results = pd.DataFrame(data_matrix, columns=classifiers)
        
        # Calculate Nemenyi p-values
        nemenyi_p_values = sp.posthoc_nemenyi_friedman(df_results)
        
        print("\np-values Matrix (Nemenyi):")
        print(nemenyi_p_values.map(lambda x: f"{x:.4f}"))
        
        return nemenyi_p_values
    else:
        print(f"[RESULT] Null hypothesis not rejected (p >= {alpha}). No significant difference found.")
        return None