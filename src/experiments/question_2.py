"""Question 2: 30×10 nested cross-validation with 5 classifiers on 2 datasets.

This module orchestrates the nested cross-validation pipeline:
- Outer loop: 30 repetitions × 10 folds = 300 test sets
- Inner loop: 5-fold stratified CV for hyperparameter tuning
- Classifiers: Bayesian, Bayesian k-NN, Parzen, Logistic Regression, Majority Voting
- Datasets: Original (spam/non-spam) and K*-clustering (from Q1)
- Metrics: Error, Precision (macro), Recall (macro), F1-measure (macro)

Output: CSV with 3,000 rows (300 outer folds × 2 datasets × 5 classifiers).
Majority voting uses the individually tuned base estimators from the inner CV.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.dataloader import (
    load_spambase_original,
    load_spambase_kstar_real,
    get_nested_cv_splitters,
)
from src.classification.bayes_gaussian import GaussianBayesClassifier
from src.classification.bayesian_knn import BayesianKNNClassifier
from src.classification.parzen_window import ParzenWindowClassifier
from src.classification.custom_logistic_regression import CustomLogisticRegression
from src.classification.majority_voting import MajorityVotingClassifier


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _fit_tuned_majority_voting(X_train, y_train, inner_cv, n_jobs):
    """Tune each voting member on inner CV, then fit the ensemble on all training data.

    Args:
        X_train (np.ndarray): Training feature matrix of shape
            (n_train, n_features).
        y_train (np.ndarray): Training target labels of shape (n_train,).
        inner_cv (StratifiedKFold): Inner cross-validation splitter.
        n_jobs (int): Number of parallel workers for ``GridSearchCV``.

    Returns:
        tuple[MajorityVotingClassifier, dict[str, dict]]: The fitted majority
            voting classifier and a dictionary mapping each base classifier
            name to its selected hyperparameter dictionary.

    Raises:
        ValueError: If no valid hyperparameter candidates remain for a base
            classifier.
    """
    base_classifiers = [
        ("GaussianBayes", GaussianBayesClassifier()),
        ("BayesianKNN", BayesianKNNClassifier()),
        ("ParzenWindow", ParzenWindowClassifier()),
        ("LogisticRegression", CustomLogisticRegression()),
    ]
    cv_splits = list(inner_cv.split(X_train, y_train))
    minimum_inner_train_size = min(len(train_indices) for train_indices, _ in cv_splits)
    selected_estimators = []
    selected_params = {}

    for classifier_name, classifier in base_classifiers:
        param_grid = classifier.get_param_grid()
        if classifier_name == "BayesianKNN":
            param_grid["n_neighbors"] = [
                neighbors
                for neighbors in param_grid["n_neighbors"]
                if neighbors <= minimum_inner_train_size
            ]
        if any(not values for values in param_grid.values()):
            raise ValueError(
                f"No valid hyperparameter candidates remain for {classifier_name}."
            )

        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            cv=cv_splits,
            scoring="f1_macro",
            n_jobs=n_jobs,
            error_score="raise",
        )
        grid_search.fit(X_train, y_train)
        selected_estimators.append(
            (classifier_name, clone(grid_search.best_estimator_))
        )
        selected_params[classifier_name] = grid_search.best_params_

    majority_voting = MajorityVotingClassifier(estimators=selected_estimators)
    majority_voting.fit(X_train, y_train)
    return majority_voting, selected_params


def _evaluate_outer_fold(
    dataset_name,
    fold_idx,
    train_idx,
    test_idx,
    X,
    y,
    inner_cv,
    classifiers_config,
    base_classifiers_config,
):
    """Evaluate all 5 classifiers on a single outer fold.

    Tunes the four base classifiers once via inner cross-validation and reuses
    their best estimators to assemble the ``MajorityVotingClassifier``.

    Args:
        dataset_name (str): Identifier of the evaluated dataset.
        fold_idx (int): Sequential outer fold index in [0, 299].
        train_idx (np.ndarray): Training row indices for this outer fold.
        test_idx (np.ndarray): Test row indices for this outer fold.
        X (np.ndarray): Full dataset feature matrix.
        y (np.ndarray): Full dataset target vector.
        inner_cv (StratifiedKFold): Inner 5-fold stratified splitter.
        classifiers_config (list[tuple[str, BaseEstimator]]): Full list of 5
            ``(name, estimator)`` pairs.
        base_classifiers_config (list[tuple[str, BaseEstimator]]): List of the
            first 4 base ``(name, estimator)`` pairs.

    Returns:
        list[dict]: List of 5 result dictionaries (one per classifier) for the
            given outer fold.
    """
    X_train_outer = X[train_idx]
    y_train_outer = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    repetition = fold_idx // 10
    fold_in_rep = fold_idx % 10

    cv_splits = list(inner_cv.split(X_train_outer, y_train_outer))
    minimum_inner_train_size = min(
        len(train_indices) for train_indices, _ in cv_splits
    )
    tuned_estimators = []
    tuned_params = {}
    fold_results = []

    for clf_name, clf in classifiers_config:
        if clf_name == "MajorityVoting":
            if len(tuned_estimators) == len(base_classifiers_config):
                best_clf = MajorityVotingClassifier(
                    estimators=[
                        (name, clone(est)) for name, est in tuned_estimators
                    ]
                )
                best_clf.fit(X_train_outer, y_train_outer)
                best_params = dict(tuned_params)
            else:
                best_clf, best_params = _fit_tuned_majority_voting(
                    X_train_outer,
                    y_train_outer,
                    inner_cv,
                    n_jobs=1,
                )
            best_f1_inner = np.nan
        else:
            param_grid = clf.get_param_grid()
            if clf_name == "BayesianKNN":
                param_grid["n_neighbors"] = [
                    neighbors
                    for neighbors in param_grid["n_neighbors"]
                    if neighbors <= minimum_inner_train_size
                ]
            grid_search = GridSearchCV(
                estimator=clf,
                param_grid=param_grid,
                cv=cv_splits,
                scoring="f1_macro",
                n_jobs=1,
                verbose=0,
                error_score="raise",
            )
            grid_search.fit(X_train_outer, y_train_outer)
            best_params = grid_search.best_params_
            best_clf = grid_search.best_estimator_
            best_f1_inner = grid_search.best_score_
            tuned_estimators.append((clf_name, clone(best_clf)))
            tuned_params[clf_name] = best_params

        y_pred = best_clf.predict(X_test)

        error = 1.0 - accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        fold_results.append(
            {
                "fold_id": fold_idx,
                "repetition": repetition + 1,
                "fold_in_rep": fold_in_rep + 1,
                "dataset": dataset_name,
                "classifier": clf_name,
                "error": error,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "best_params": str(best_params),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "best_f1_inner": best_f1_inner,
            }
        )

    return fold_results


def run_nested_cv_experiment(
    output_dir="results",
    n_jobs=1,
    verbose=0,
):
    """Run 30×10 nested cross-validation on both datasets with 5 classifiers.

    Args:
        output_dir (str or Path, optional): Directory where results will be
            saved. Defaults to "results".
        n_jobs (int, optional): Number of parallel jobs across outer folds.
            Defaults to 1.
        verbose (int, optional): Verbosity level (0=silent, 1=progress,
            2=detailed). Defaults to 0.

    Returns:
        pd.DataFrame: DataFrame with 3,000 rows (300 folds × 2 datasets × 5
            classifiers) and columns ``fold_id``, ``repetition``,
            ``fold_in_rep``, ``dataset``, ``classifier``, ``error``,
            ``precision``, ``recall``, ``f1``, ``best_params``, ``n_train``,
            ``n_test``, and ``best_f1_inner``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("QUESTION 2: 30×10 Nested Cross-Validation Experiment")
    logger.info("=" * 70)

    # Load datasets
    logger.info("Loading datasets...")
    X_original, y_original = load_spambase_original()
    X_kstar, y_kstar = load_spambase_kstar_real()

    logger.info(f"  Original dataset: X={X_original.shape}, classes={np.unique(y_original)}")
    logger.info(f"    Class distribution: {np.bincount(y_original)}")
    logger.info(f"  K*-clustering dataset: X={X_kstar.shape}, clusters={np.unique(y_kstar)}")
    logger.info(f"    Cluster distribution: {np.bincount(y_kstar)}")

    # Define datasets
    datasets = {
        "original_2_classes": (X_original, y_original),
        "kstar_classes": (X_kstar, y_kstar),
    }

    # Define classifiers with their param grids
    base_classifiers_config = [
        ("GaussianBayes", GaussianBayesClassifier()),
        ("BayesianKNN", BayesianKNNClassifier()),
        ("ParzenWindow", ParzenWindowClassifier()),
        ("LogisticRegression", CustomLogisticRegression()),
    ]

    # Create MajorityVoting ensemble from the first 4 classifiers
    voting_estimators = [
        (name, clf.__class__()) for name, clf in base_classifiers_config
    ]
    majority_voting = MajorityVotingClassifier(estimators=voting_estimators)

    classifiers_config = base_classifiers_config + [("MajorityVoting", majority_voting)]

    # Get CV splitters
    outer_cv, inner_cv = get_nested_cv_splitters()

    # Collect results
    results = []
    total_iterations = 0
    completed_iterations = 0

    # Pre-calculate total for progress tracking
    n_folds = sum(1 for _ in outer_cv.split(X_original, y_original))
    n_datasets = len(datasets)
    n_classifiers = len(classifiers_config)
    total_expected = n_folds * n_datasets * n_classifiers

    logger.info(f"Total iterations expected: {total_expected}")
    logger.info(f"  ({n_folds} outer folds × {n_datasets} datasets × {n_classifiers} classifiers)")

    # Iterate over datasets
    for dataset_name, (X, y) in datasets.items():
        logger.info(f"\nProcessing dataset: {dataset_name} (n_jobs={n_jobs})")
        outer_splits = list(outer_cv.split(X, y))

        if n_jobs == 1:
            for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
                repetition = fold_idx // 10
                fold_in_rep = fold_idx % 10
                if verbose >= 1 and fold_idx % 10 == 0:
                    logger.info(
                        f"  Repetition {repetition + 1}/30, fold {fold_in_rep + 1}/10..."
                    )
                fold_rows = _evaluate_outer_fold(
                    dataset_name,
                    fold_idx,
                    train_idx,
                    test_idx,
                    X,
                    y,
                    inner_cv,
                    classifiers_config,
                    base_classifiers_config,
                )
                results.extend(fold_rows)
                completed_iterations += len(fold_rows)
        else:
            fold_batches = Parallel(n_jobs=n_jobs, verbose=10 if verbose >= 1 else 0)(
                delayed(_evaluate_outer_fold)(
                    dataset_name,
                    fold_idx,
                    train_idx,
                    test_idx,
                    X,
                    y,
                    inner_cv,
                    classifiers_config,
                    base_classifiers_config,
                )
                for fold_idx, (train_idx, test_idx) in enumerate(outer_splits)
            )
            for fold_rows in fold_batches:
                results.extend(fold_rows)
                completed_iterations += len(fold_rows)

    logger.info(f"\nCompleted: {completed_iterations}/{total_expected} iterations")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Validate shape
    assert len(df_results) == total_expected, \
        f"Expected {total_expected} results, got {len(df_results)}"

    logger.info(f"Results shape: {df_results.shape}")

    # Save to CSV
    output_file = output_dir / "cv_30x10_results.csv"
    df_results.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")

    # Print summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY STATISTICS (by dataset and classifier)")
    logger.info("=" * 70)

    for dataset_name in datasets.keys():
        logger.info(f"\nDataset: {dataset_name}")
        df_subset = df_results[df_results["dataset"] == dataset_name]

        for clf_name in [name for name, _ in classifiers_config]:
            df_clf = df_subset[df_subset["classifier"] == clf_name]
            mean_f1 = df_clf["f1"].mean()
            std_f1 = df_clf["f1"].std()
            mean_error = df_clf["error"].mean()
            std_error = df_clf["error"].std()

            logger.info(
                f"  {clf_name:20s} | "
                f"F1={mean_f1:.4f}±{std_f1:.4f} | "
                f"Error={mean_error:.4f}±{std_error:.4f}"
            )

    logger.info("\n" + "=" * 70)
    logger.info("Experiment completed successfully!")
    logger.info("=" * 70)

    return df_results


def main():
    """Parse CLI arguments and run the Question 2 nested CV experiment.

    Returns:
        pd.DataFrame: Results DataFrame returned by ``run_nested_cv_experiment``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Question 2: 30×10 nested cross-validation experiment"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/q2",
        help="Directory where results will be saved (default: results/q2)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for GridSearchCV (default: -1, all CPU cores)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level: 0=silent, 1=progress, 2=detailed (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available (not yet implemented)",
    )

    args = parser.parse_args()

    df_results = run_nested_cv_experiment(
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )

    return df_results


if __name__ == "__main__":
    main()
