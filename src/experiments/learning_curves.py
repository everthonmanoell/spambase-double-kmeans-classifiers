"""Generate 5%-to-95% learning curves for the two Question 2 datasets."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.classification.bayes_gaussian import GaussianBayesClassifier
from src.classification.bayesian_knn import BayesianKNNClassifier
from src.classification.custom_logistic_regression import CustomLogisticRegression
from src.classification.majority_voting import MajorityVotingClassifier
from src.classification.parzen_window import ParzenWindowClassifier
from src.utils.dataloader import (
    get_learning_curve_splits,
    load_spambase_kstar_real,
    load_spambase_original,
)


LOGGER = logging.getLogger(__name__)
DATASET_NAMES = ("original_2_classes", "kstar_classes")
CLASSIFIER_NAMES = (
    "GaussianBayes",
    "BayesianKNN",
    "ParzenWindow",
    "LogisticRegression",
    "MajorityVoting",
)
DEFAULT_TRAIN_SIZES = tuple(np.arange(0.05, 0.96, 0.05).round(2))


def _make_classifiers():
    """Create fresh base estimators before tuning each learning-curve split.

    Returns:
        list[tuple[str, BaseEstimator]]: List of the four base
            ``(name, estimator)`` pairs.
    """
    base_estimators = [
        ("GaussianBayes", GaussianBayesClassifier()),
        ("BayesianKNN", BayesianKNNClassifier()),
        ("ParzenWindow", ParzenWindowClassifier()),
        ("LogisticRegression", CustomLogisticRegression()),
    ]
    return base_estimators


def _tune_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    n_jobs: int = 1,
) -> tuple[list[tuple[str, BaseEstimator]], dict[str, dict[str, object]]]:
    """Tune base classifiers with stratified CV using training data only.

    Args:
        X_train (np.ndarray): Training feature matrix for the current split.
        y_train (np.ndarray): Training target labels for the current split.
        random_state (int): Seed for the inner ``StratifiedKFold`` splitter.
        n_jobs (int, optional): Number of parallel jobs for ``GridSearchCV``.
            Defaults to 1.

    Returns:
        tuple[list[tuple[str, BaseEstimator]], dict[str, dict[str, object]]]:
            List of fitted ``(name, best_estimator)`` tuples and a dictionary of
            selected hyperparameters per classifier.

    Raises:
        ValueError: If the rarest class has fewer than 2 training examples or
            if no valid hyperparameter candidates remain for a classifier.
    """
    class_counts = np.unique(y_train, return_counts=True)[1]
    minimum_class_count = int(class_counts.min())
    if minimum_class_count < 2:
        raise ValueError(
            "At least two training examples per class are needed for stratified "
            "cross-validation; increase the learning-curve training proportion."
        )

    n_splits = min(5, minimum_class_count)
    inner_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    inner_splits = list(inner_cv.split(X_train, y_train))
    minimum_inner_train_size = min(len(train_indices) for train_indices, _ in inner_splits)
    tuned_estimators = []
    selected_params = {}

    for classifier_name, classifier in _make_classifiers():
        parameter_grid = classifier.get_param_grid()
        if classifier_name == "BayesianKNN":
            parameter_grid["n_neighbors"] = [
                neighbors
                for neighbors in parameter_grid["n_neighbors"]
                if neighbors <= minimum_inner_train_size
            ]
        if any(not values for values in parameter_grid.values()):
            raise ValueError(
                f"No valid hyperparameter candidates remain for {classifier_name} "
                f"with the current training subset."
            )

        search = GridSearchCV(
            estimator=classifier,
            param_grid=parameter_grid,
            scoring="f1_macro",
            cv=inner_splits,
            refit=True,
            error_score="raise",
            n_jobs=n_jobs,
        )
        search.fit(X_train, y_train)
        tuned_estimators.append((classifier_name, search.best_estimator_))
        selected_params[classifier_name] = search.best_params_

    return tuned_estimators, selected_params


def _evaluate_learning_curve_split(
    dataset_name: str,
    train_size: float,
    repeat_index: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> list[dict[str, object]]:
    """Evaluate all 5 classifiers on a single (train_size, repeat) split.

    Args:
        dataset_name (str): Name of the dataset being evaluated.
        train_size (float): Proportion of data used for training in [0.05, 0.95].
        repeat_index (int): 1-based index of the stratified split repetition.
        train_indices (np.ndarray): Row indices of the training set.
        test_indices (np.ndarray): Row indices of the test set.
        X (np.ndarray): Full feature matrix.
        y (np.ndarray): Full target vector.
        random_state (int): Base random seed for inner CV splitting.

    Returns:
        list[dict[str, object]]: List of 5 evaluation records (one per
            classifier) with train and test macro-F1 scores.
    """
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    tuned_estimators, selected_params = _tune_classifiers(
        X_train,
        y_train,
        random_state=random_state + repeat_index,
        n_jobs=1,
    )
    voting_estimators = [
        (name, clone(estimator)) for name, estimator in tuned_estimators
    ]
    classifiers = tuned_estimators + [
        (
            "MajorityVoting",
            MajorityVotingClassifier(estimators=voting_estimators),
        )
    ]
    selected_params["MajorityVoting"] = {
        name: params for name, params in selected_params.items()
    }
    inner_cv_splits = min(
        5, int(np.unique(y_train, return_counts=True)[1].min())
    )

    split_rows = []
    for classifier_name, classifier in classifiers:
        if classifier_name == "MajorityVoting":
            classifier.fit(X_train, y_train)
        train_prediction = classifier.predict(X_train)
        test_prediction = classifier.predict(X_test)
        split_rows.append(
            {
                "dataset": dataset_name,
                "classifier": classifier_name,
                "train_size": train_size,
                "repeat": repeat_index,
                "n_train": len(train_indices),
                "n_test": len(test_indices),
                "inner_cv_splits": inner_cv_splits,
                "best_params": json.dumps(
                    selected_params[classifier_name],
                    sort_keys=True,
                ),
                "train_f1": f1_score(
                    y_train,
                    train_prediction,
                    average="macro",
                    zero_division=0,
                ),
                "test_f1": f1_score(
                    y_test,
                    test_prediction,
                    average="macro",
                    zero_division=0,
                ),
            }
        )
    return split_rows


def _summarize_learning_curve_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split-level scores into means and standard deviations.

    Args:
        raw_results (pd.DataFrame): Split-level DataFrame produced by
            ``run_learning_curve_experiment``.

    Returns:
        pd.DataFrame: Summary DataFrame grouped by ``dataset``, ``classifier``,
            and ``train_size``.
    """
    return (
        raw_results.groupby(
            ["dataset", "classifier", "train_size"], as_index=False, sort=True
        )
        .agg(
            n_train=("n_train", "mean"),
            n_test=("n_test", "mean"),
            n_repeats=("repeat", "nunique"),
            train_f1_mean=("train_f1", "mean"),
            train_f1_std=("train_f1", "std"),
            test_f1_mean=("test_f1", "mean"),
            test_f1_std=("test_f1", "std"),
        )
        .fillna({"train_f1_std": 0.0, "test_f1_std": 0.0})
    )


def _plot_dataset_learning_curves(
    summary: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:
    """Plot train/test macro-F1 with ±1 SD for every classifier.

    Args:
        summary (pd.DataFrame): Aggregated learning-curve statistics.
        dataset_name (str): Name of the dataset to plot.
        output_path (Path): Destination PNG file path.
    """
    figure, axes = plt.subplots(1, len(CLASSIFIER_NAMES), figsize=(22, 5), sharey=True)

    for axis, classifier_name in zip(axes, CLASSIFIER_NAMES):
        classifier = summary[
            (summary["dataset"] == dataset_name)
            & (summary["classifier"] == classifier_name)
        ].sort_values("train_size")
        x_values = classifier["train_size"].to_numpy() * 100

        for score_name, color, label in (
            ("train", "#2878B5", "Treino"),
            ("test", "#D9534F", "Teste"),
        ):
            means = classifier[f"{score_name}_f1_mean"].to_numpy()
            stds = classifier[f"{score_name}_f1_std"].to_numpy()
            axis.plot(x_values, means, marker="o", markersize=3.5, color=color, label=label)
            axis.fill_between(
                x_values,
                np.clip(means - stds, 0, 1),
                np.clip(means + stds, 0, 1),
                color=color,
                alpha=0.16,
                linewidth=0,
            )

        axis.set_title(classifier_name, fontsize=10)
        axis.set_xlabel("Proporção de treino (%)")
        axis.set_xticks(np.arange(5, 100, 10))
        axis.set_xlim(2, 98)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25, linestyle="--")
        axis.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("F1 macro")
    figure.suptitle(
        f"Curvas de aprendizagem — {dataset_name.replace('_', ' ')}\n"
        "Média ± 1 desvio-padrão em 10 divisões estratificadas",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89), w_pad=1.0)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_learning_curve_experiment(
    results_dir: str | Path = "results",
    plots_dir: str | Path = "plots",
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run and save learning curves for original and Q1-cluster datasets.

    Each base classifier is tuned by stratified CV using only the current
    training subset. The majority-voting model uses those selected parameters.

    Args:
        results_dir (str or Path, optional): Directory where detailed and
            summary CSVs are saved. Defaults to "results".
        plots_dir (str or Path, optional): Directory where learning-curve PNG
            plots are saved. Defaults to "plots".
        n_repeats (int, optional): Number of stratified splits per training
            proportion. Must be at least 2. Defaults to 10.
        random_state (int, optional): Base random seed for reproducible splits.
            Defaults to 42.
        n_jobs (int, optional): Number of parallel jobs across splits. Defaults
            to 1.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple ``(raw_results, summary)``
            containing split-level scores and aggregated mean/SD statistics.

    Raises:
        ValueError: If ``n_repeats < 2``.
        RuntimeError: If the number of collected rows does not match the
            expected total.
    """
    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2 to estimate error bars.")

    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    original_X, original_y = load_spambase_original()
    kstar_X, kstar_y = load_spambase_kstar_real()
    datasets = {
        "original_2_classes": (original_X, original_y),
        "kstar_classes": (kstar_X, kstar_y),
    }

    raw_rows = []
    for dataset_name, (X, y) in datasets.items():
        LOGGER.info(
            "Learning curves for %s: %d samples, %d features, class counts=%s",
            dataset_name,
            X.shape[0],
            X.shape[1],
            np.bincount(y).tolist(),
        )
        splits_by_size = get_learning_curve_splits(
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
        )

        for train_size in DEFAULT_TRAIN_SIZES:
            splits = splits_by_size[train_size]
            LOGGER.info(
                "  %s: train size %.0f%% (%d/%d splits)",
                dataset_name,
                train_size * 100,
                len(splits),
                n_repeats,
            )
            if n_jobs == 1:
                for repeat_index, (train_indices, test_indices) in enumerate(
                    splits, start=1
                ):
                    raw_rows.extend(
                        _evaluate_learning_curve_split(
                            dataset_name,
                            train_size,
                            repeat_index,
                            train_indices,
                            test_indices,
                            X,
                            y,
                            random_state,
                        )
                    )
            else:
                split_batches = Parallel(n_jobs=n_jobs)(
                    delayed(_evaluate_learning_curve_split)(
                        dataset_name,
                        train_size,
                        repeat_index,
                        train_indices,
                        test_indices,
                        X,
                        y,
                        random_state,
                    )
                    for repeat_index, (train_indices, test_indices) in enumerate(
                        splits, start=1
                    )
                )
                for split_rows in split_batches:
                    raw_rows.extend(split_rows)

    raw_results = pd.DataFrame(raw_rows)
    expected_rows = (
        len(DATASET_NAMES)
        * len(CLASSIFIER_NAMES)
        * len(DEFAULT_TRAIN_SIZES)
        * n_repeats
    )
    if len(raw_results) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} learning-curve rows, got {len(raw_results)}."
        )

    summary = _summarize_learning_curve_results(raw_results)
    raw_path = results_dir / "learning_curves_data.csv"
    summary_path = results_dir / "learning_curves_summary.csv"
    raw_results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Split-level scores saved to %s", raw_path)
    LOGGER.info("Mean and standard-deviation summary saved to %s", summary_path)

    for dataset_name, plot_name in (
        ("original_2_classes", "learning_curves_original.png"),
        ("kstar_classes", "learning_curves_kstar.png"),
    ):
        plot_path = plots_dir / plot_name
        _plot_dataset_learning_curves(summary, dataset_name, plot_path)
        LOGGER.info("Learning-curve plot saved to %s", plot_path)

    return raw_results, summary


def main() -> None:
    """Parse CLI arguments and run the Question 2 learning-curve experiment."""
    parser = argparse.ArgumentParser(
        description="Generate learning curves for the two Question 2 datasets."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for GridSearchCV (default: -1, all CPU cores)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    raw_results, _ = run_learning_curve_experiment(
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    LOGGER.info(
        "Learning-curve experiment completed: %d classifier evaluations.",
        len(raw_results),
    )


if __name__ == "__main__":
    main()
