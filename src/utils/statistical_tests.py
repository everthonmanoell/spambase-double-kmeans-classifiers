"""Statistical tests for comparing classifiers across aligned CV folds."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare


@dataclass(frozen=True)
class FriedmanNemenyiResult:
    """Results of the Friedman test and its conditional Nemenyi post-hoc test.

    Args:
        statistic (float): Friedman chi-square test statistic.
        p_value (float): Two-sided p-value associated with the Friedman test.
        nemenyi_p_values (pd.DataFrame or None): Pairwise Nemenyi p-value matrix
            when ``p_value < alpha``, or ``None`` otherwise.
    """

    statistic: float
    p_value: float
    nemenyi_p_values: pd.DataFrame | None


def perform_friedman_nemenyi(
    df_results: pd.DataFrame,
    metric_name: str,
    alpha: float = 0.05,
) -> FriedmanNemenyiResult:
    """Compare classifiers on aligned blocks, applying Nemenyi when warranted.

    Rows must represent the same folds for every classifier; columns represent
    classifiers. The post-hoc test is run only when the omnibus test is
    significant at ``alpha``.

    Args:
        df_results (pd.DataFrame): DataFrame of shape (n_blocks, n_classifiers)
            containing aligned fold scores for each classifier.
        metric_name (str): Display name of the evaluated metric.
        alpha (float, optional): Significance threshold in (0, 1). Defaults to
            0.05.

    Returns:
        FriedmanNemenyiResult: Dataclass containing the Friedman statistic,
            p-value, and optional Nemenyi pairwise p-value DataFrame.

    Raises:
        TypeError: If ``df_results`` is not a DataFrame or contains non-numeric
            entries.
        ValueError: If ``df_results`` is empty, has fewer than 2 blocks or 3
            classifiers, has duplicate columns, contains NaN/inf values, has
            identical values across all classifiers, or if ``alpha`` is outside
            (0, 1).
    """
    if not isinstance(df_results, pd.DataFrame):
        raise TypeError("Results must be a pandas DataFrame with aligned folds.")
    if df_results.empty:
        raise ValueError("The results DataFrame is empty.")
    if len(df_results) < 2:
        raise ValueError("At least two aligned blocks are required.")
    if len(df_results.columns) < 3:
        raise ValueError("The Friedman test requires at least three classifiers.")
    if df_results.columns.has_duplicates:
        raise ValueError("Classifier names must be unique.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if df_results.isna().any().any():
        raise ValueError("Results contain missing values.")
    if not df_results.map(lambda value: pd.api.types.is_number(value)).all().all():
        raise TypeError("All classifier results must be numeric.")
    if not df_results.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all().all():
        raise ValueError("Results must contain only finite values.")
    first_classifier = df_results.iloc[:, 0].to_numpy()
    if all(
        np.array_equal(first_classifier, df_results[classifier].to_numpy())
        for classifier in df_results.columns[1:]
    ):
        raise ValueError("The Friedman test is undefined when all results are identical.")

    statistic, p_value = friedmanchisquare(
        *(df_results[classifier].to_numpy() for classifier in df_results.columns),
        nan_policy="raise",
    )
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        raise ValueError("The Friedman test is undefined for these classifier results.")
    nemenyi_p_values = (
        sp.posthoc_nemenyi_friedman(df_results)
        if p_value < alpha
        else None
    )

    print(f"\n{metric_name.upper()}: Friedman chi-square={statistic:.4f}, p={p_value:.4e}")
    if nemenyi_p_values is None:
        print(f"  No significant global difference at alpha={alpha:.3f}; Nemenyi not run.")
    else:
        print(f"  Significant global difference at alpha={alpha:.3f}; Nemenyi p-values:")
        print(nemenyi_p_values.to_string(float_format=lambda value: f"{value:.4f}"))

    return FriedmanNemenyiResult(
        statistic=float(statistic),
        p_value=float(p_value),
        nemenyi_p_values=nemenyi_p_values,
    )
