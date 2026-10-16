import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

def perform_friedman_nemenyi(df_results, metric_name, alpha=0.05):
    """
    Performs the Friedman test and the Nemenyi post-hoc test for the classifiers.

    Args:
        df_results (pd.DataFrame): DataFrame where columns are classifier names and 
                                   rows are the metric results for each block (fold).
                                   The index ensures fold alignment across classifiers.
        metric_name (str): Name of the evaluated metric (used for logging).
        alpha (float): Significance level. Defaults to 0.05.
    
    Returns:
        pd.DataFrame: A matrix of p-values from the Nemenyi test (or None if the 
                      Friedman test is not significant).
    """
    # Validação de entrada: Garante o uso de DataFrame para travamento de alinhamento
    if not isinstance(df_results, pd.DataFrame):
        raise TypeError("The results must be provided as a pandas DataFrame to ensure strict fold alignment.")
    
    # Verifica proativamente se há NaNs para evitar resultados silenciosamente falsos
    if df_results.isna().any().any():
        raise ValueError("Input data contains NaNs. A model likely failed during cross-validation (error_score=np.nan).")

    classifiers = df_results.columns.tolist()
    n_blocks = len(df_results)
    
    if n_blocks == 0:
        raise ValueError("The provided DataFrame is empty.")

    # Extrai as colunas como arrays para o Scipy
    data = [df_results[clf].values for clf in classifiers]

    print(f"\n{'='*40}")
    print(f"STATISTICAL ANALYSIS: {metric_name.upper()}")
    print(f"{'='*40}")

    # 1. Friedman Test com nan_policy='raise'
    stat, p_value = friedmanchisquare(*data, nan_policy='raise')
    
    print(f"Friedman Test: Statistic = {stat:.4f}, p-value = {p_value:.4e}")

    if p_value < alpha:
        print(f"[RESULT] Null hypothesis rejected (p < {alpha}). Significant difference found between models.")
        
        # 2. Nemenyi Post-hoc Test
        print("\nExecuting Nemenyi Post-hoc Test...")
        
        nemenyi_p_values = sp.posthoc_nemenyi_friedman(df_results)
        
        print("\np-values Matrix (Nemenyi):")
        print(nemenyi_p_values.map(lambda x: f"{x:.4f}"))
        
        return nemenyi_p_values
    else:
        print(f"[RESULT] Null hypothesis not rejected (p >= {alpha}). No significant difference found.")
        return None