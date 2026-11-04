import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, StratifiedKFold

def load_spambase_original(filepath="dataset/spambase.data"):
    """Load the original Spambase dataset with its two a priori classes.

    Args:
        filepath (str or Path, optional): Path to the Spambase data file.
            Defaults to "dataset/spambase.data".

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple ``(X, y)`` containing the feature
            matrix of shape ``(n_samples, 57)`` and the original binary class
            labels (1 = spam, 0 = non-spam).
    """
    # The data file has no header, so read it without column names.
    df = pd.read_csv(filepath, header=None)

    # The first 57 columns contain the features.
    X = df.iloc[:, :-1].values
    # The last column contains the target labels.
    y = df.iloc[:, -1].values

    return X, y

def load_spambase_kstar_mock(filepath="dataset/spambase.data", k_star=3, random_state=42):
    """Load the Spambase features with mock labels for ``K*`` classes.

    This is a temporary mock implementation until the Double k-means
    integration is complete.

    Args:
        filepath (str or Path, optional): Path to the Spambase data file.
            Defaults to "dataset/spambase.data".
        k_star (int, optional): Number of classes to generate. Defaults to 3.
        random_state (int, optional): Seed used to generate reproducible mock
            labels. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple ``(X, y_mock)`` containing the
            feature matrix and randomly generated integer labels.
    """
    X, _ = load_spambase_original(filepath)

    # Generate random labels to simulate Double k-means output.
    rng = np.random.default_rng(random_state)
    y_mock = rng.integers(0, k_star, size=X.shape[0])

    return X, y_mock

def load_spambase_kstar_real(filepath="dataset/spambase.data",
                             q1_labels_path="results/q1_object_labels.csv",
                             q1_selection_path="results/q1_selection.json"):
    """Load Spambase features with K*-clustering labels from Q1 analysis.

    This function loads the real K*-clustering partition discovered by Double k-means
    in Q1, instead of using mock random labels. The labels are read from the CSV file
    saved during Q1 evaluation, ensuring exact reproducibility.

    Args:
        filepath (str or Path, optional): Path to the Spambase data file.
            Defaults to "dataset/spambase.data".
        q1_labels_path (str or Path, optional): Path to ``q1_object_labels.csv``
            containing ``row_index``, ``original_class``, and ``cluster``.
            Defaults to "results/q1_object_labels.csv".
        q1_selection_path (str or Path, optional): Path to ``q1_selection.json``
            containing metadata including ``K*`` and ``H*``. Defaults to
            "results/q1_selection.json".

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple ``(X, y_kstar)`` where ``X`` is
            the feature matrix of shape ``(n_samples, 57)`` and ``y_kstar`` is
            the vector of cluster labels from the winning Q1 partition.

    Raises:
        FileNotFoundError: If ``q1_labels_path`` or ``q1_selection_path`` does
            not exist.
        ValueError: If labels cannot be loaded or have an inconsistent shape or
            cluster count.
    """
    # Load features from original dataset.
    X, _ = load_spambase_original(filepath)

    # Verify that q1_labels and q1_selection files exist.
    q1_labels_file = Path(q1_labels_path)
    q1_selection_file = Path(q1_selection_path)

    if not q1_labels_file.exists():
        raise FileNotFoundError(
            f"Q1 labels file not found: {q1_labels_path}\n"
            f"Run Q1 analysis first: python -m src.analysis.clustering_evaluation"
        )

    if not q1_selection_file.exists():
        raise FileNotFoundError(
            f"Q1 selection metadata not found: {q1_selection_path}\n"
            f"Run Q1 analysis first: python -m src.analysis.clustering_evaluation"
        )

    # Load Q1 selection metadata to verify K* and H*.
    with open(q1_selection_file, 'r') as f:
        q1_metadata = json.load(f)

    k_star = q1_metadata["K"]
    h_star = q1_metadata["H"]

    # Load cluster labels from Q1 CSV (preserves row order).
    df_labels = pd.read_csv(q1_labels_file)

    if df_labels.shape[0] != X.shape[0]:
        raise ValueError(
            f"Label count mismatch: CSV has {df_labels.shape[0]} rows, "
            f"but dataset has {X.shape[0]} rows. Check data integrity."
        )

    # Extract cluster column and verify it contains only valid cluster indices.
    y_kstar = df_labels["cluster"].values
    n_unique_clusters = len(np.unique(y_kstar))

    if n_unique_clusters != k_star:
        raise ValueError(
            f"Cluster count mismatch: expected {k_star} clusters (from Q1 metadata), "
            f"but found {n_unique_clusters} in the labels file."
        )

    return X, y_kstar

def get_nested_cv_splitters():
    """Create splitters for the nested cross-validation procedure.

    Returns:
        tuple[RepeatedStratifiedKFold, StratifiedKFold]: A tuple
            ``(outer_cv, inner_cv)`` where ``outer_cv`` is a
            ``RepeatedStratifiedKFold`` (10 folds, 30 repetitions) for outer
            evaluation and ``inner_cv`` is a ``StratifiedKFold`` (5 folds) for
            inner hyperparameter tuning.
    """
    outer_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return outer_cv, inner_cv

def get_learning_curve_splits(X, y, n_repeats=10, random_state=42):
    """Generate stratified train/test splits for a learning curve.

    Args:
        X (np.ndarray): Feature matrix used to generate the splits.
        y (np.ndarray): Target labels used for stratification.
        n_repeats (int, optional): Number of randomized splits per training
            size to allow variance calculation. Defaults to 10.
        random_state (int, optional): Seed used to generate reproducible random
            splits. Defaults to 42.

    Returns:
        dict[float, list[tuple[np.ndarray, np.ndarray]]]: Dictionary mapping
            each training-set proportion from 0.05 to 0.95 to a list of
            ``(train_idx, test_idx)`` index tuples.
    """
    splits = {}

    # Vary the training-set proportion from 5% to 95% in 5% increments.
    train_sizes = np.arange(0.05, 0.96, 0.05)

    for train_size in train_sizes:
        train_size_rounded = round(train_size, 2)

        # StratifiedShuffleSplit preserves the class distribution in each split.
        # n_splits is set to n_repeats to generate multiple independent samples.
        splitter = StratifiedShuffleSplit(
            n_splits=n_repeats, 
            train_size=train_size_rounded, 
            random_state=random_state
        )

        # Store all generated splits for this specific training size as a list
        splits[train_size_rounded] = list(splitter.split(X, y))

    return splits