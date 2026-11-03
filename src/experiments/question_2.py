"""Question 2: 30×10 nested cross-validation with 5 classifiers on 2 datasets.

This module orchestrates the nested cross-validation pipeline:
- Outer loop: 30 repetitions × 10 folds = 300 test sets
- Inner loop: 5-fold stratified CV for hyperparameter tuning
- Classifiers: Bayesian, Bayesian k-NN, Parzen, Logistic Regression, Majority Voting
- Datasets: Original (spam/non-spam) and K*-clustering (from Q1)
- Metrics: Error, Precision (macro), Recall (macro), F1-measure (macro)

Output: CSV with 3,000 rows (300 outer folds × 2 datasets × 5 classifiers)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
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


def run_nested_cv_experiment(
    output_dir="results",
    n_jobs=1,
    verbose=0,
):
    """Run 30×10 nested cross-validation on both datasets with 5 classifiers.

    Args:
        output_dir: Directory where results will be saved.
        n_jobs: Number of parallel jobs for GridSearchCV.
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).

    Returns:
        A pandas DataFrame with 3,000 rows (300 folds × 2 datasets × 5 classifiers).
        Columns: fold_id, repetition, fold, dataset, classifier, error, precision,
                 recall, f1, best_params, n_train, n_test
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
        logger.info(f"\nProcessing dataset: {dataset_name}")

        # Iterate over outer CV folds
        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y)
        ):
            X_train_outer = X[train_idx]
            y_train_outer = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            # Calculate repetition (30×10 structure)
            repetition = fold_idx // 10
            fold_in_rep = fold_idx % 10

            if verbose >= 1 and fold_idx % 10 == 0:
                logger.info(f"  Repetition {repetition + 1}/30, fold {fold_in_rep + 1}/10...")

            # Iterate over classifiers
            for clf_name, clf in classifiers_config:
                try:
                    # Get param grid for this classifier
                    param_grid = clf.get_param_grid()

                    # Setup GridSearchCV for hyperparameter tuning (inner CV)
                    grid_search = GridSearchCV(
                        estimator=clf,
                        param_grid=param_grid,
                        cv=inner_cv,
                        scoring="f1_macro",
                        n_jobs=n_jobs,
                        verbose=0,
                    )

                    # Fit on training data (inner CV)
                    grid_search.fit(X_train_outer, y_train_outer)
                    best_params = grid_search.best_params_

                    # Get best estimator
                    best_clf = grid_search.best_estimator_

                    # Predict on test set
                    y_pred = best_clf.predict(X_test)

                    # Calculate metrics
                    error = 1.0 - accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
                    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

                    # Store result
                    result = {
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
                        "best_f1_inner": grid_search.best_score_,
                    }
                    results.append(result)
                    completed_iterations += 1

                    if verbose >= 2:
                        logger.info(
                            f"    {clf_name:20s} | "
                            f"error={error:.4f} | "
                            f"f1={f1:.4f}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error in {dataset_name} fold {fold_idx} "
                        f"classifier {clf_name}: {e}"
                    )
                    raise

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
    """Entry point for CLI execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Question 2: 30×10 nested cross-validation experiment"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where results will be saved (default: results/)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for GridSearchCV (default: 1)",
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
