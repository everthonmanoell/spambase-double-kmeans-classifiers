import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, StratifiedKFold

def load_spambase_original(filepath="dataset/spambase.data"):
    """Load the original Spambase dataset with its two classes.

    Args:
        filepath: Path to the Spambase data file.

    Returns:
        A tuple containing the feature matrix and the original class labels.
        The last column contains the class labels, where 1 represents spam
        and 0 represents non-spam.
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
        filepath: Path to the Spambase data file.
        k_star: Number of classes to generate.
        random_state: Seed used to generate reproducible mock labels.

    Returns:
        A tuple containing the feature matrix and randomly generated labels.
    """
    X, _ = load_spambase_original(filepath)

    # Generate random labels to simulate Double k-means output.
    rng = np.random.default_rng(random_state)
    y_mock = rng.integers(0, k_star, size=X.shape[0])

    return X, y_mock

def get_nested_cv_splitters():
    """Create splitters for the nested cross-validation procedure.

    Returns:
        A tuple (outer_cv, inner_cv) where:
        - outer_cv is a ``RepeatedStratifiedKFold`` (10 folds, 30 repetitions) for evaluation.
        - inner_cv is a ``StratifiedKFold`` (5 folds) for inner hyperparameter tuning.
    """
    outer_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=30, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return outer_cv, inner_cv

def get_learning_curve_splits(X, y, n_repeats=10, random_state=42):
    """Generate stratified train/test splits for a learning curve.

    Args:
        X: Feature matrix used to generate the splits.
        y: Target labels used for stratification.
        n_repeats: Number of randomized splits per training size to allow variance calculation.
        random_state: Seed used to generate reproducible random splits.

    Returns:
        A dictionary mapping each training-set proportion from 0.05 to 0.95
        to a list of tuples containing (train_idx, test_idx).
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