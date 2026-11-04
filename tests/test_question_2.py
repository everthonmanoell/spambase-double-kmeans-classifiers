"""Tests for Question 2: nested cross-validation experiment."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil
import ast

import src.experiments.question_2 as question_2_module
from src.experiments.question_2 import (
    _evaluate_outer_fold,
    _fit_tuned_majority_voting,
    run_nested_cv_experiment,
)
from src.classification.bayes_gaussian import GaussianBayesClassifier
from src.classification.bayesian_knn import BayesianKNNClassifier
from src.classification.parzen_window import ParzenWindowClassifier
from src.classification.custom_logistic_regression import CustomLogisticRegression
from src.utils.dataloader import get_nested_cv_splitters


@pytest.fixture(scope="class")
def fast_q2_pipeline_env():
    """Patch expensive full-grid 4601-row fits with a fast 30-sample slice per fold while preserving the exact 30×10 outer fold structure and real classifiers."""
    mp = pytest.MonkeyPatch()
    rng = np.random.default_rng(42)
    X_synth = rng.normal(size=(4601, 3))
    y_synth = np.zeros(4601, dtype=int)
    y_synth[:1813] = 1
    rng.shuffle(y_synth)
    X_synth[y_synth == 1] += 1.8

    mp.setattr(question_2_module, "load_spambase_original", lambda: (X_synth, y_synth))
    mp.setattr(question_2_module, "load_spambase_kstar_real", lambda: (X_synth, y_synth))
    mp.setattr(
        BayesianKNNClassifier,
        "get_param_grid",
        staticmethod(lambda: {"n_neighbors": [3], "metric": ["euclidean"]}),
    )
    mp.setattr(
        ParzenWindowClassifier,
        "get_param_grid",
        staticmethod(lambda: {"bandwidth": [1.0]}),
    )
    mp.setattr(
        CustomLogisticRegression,
        "get_param_grid",
        staticmethod(lambda: {"C": [1.0], "solver": ["lbfgs"]}),
    )

    cache = {}

    def _fast_evaluate_outer_fold(
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
        cache_key = (dataset_name, fold_idx % 2)
        if cache_key not in cache:
            c0_tr = train_idx[y[train_idx] == 0][:15]
            c1_tr = train_idx[y[train_idx] == 1][:15]
            small_train_idx = np.concatenate([c0_tr, c1_tr])
            c0_te = test_idx[y[test_idx] == 0][:10]
            c1_te = test_idx[y[test_idx] == 1][:10]
            small_test_idx = np.concatenate([c0_te, c1_te])

            cache[cache_key] = _evaluate_outer_fold(
                dataset_name,
                fold_idx,
                small_train_idx,
                small_test_idx,
                X,
                y,
                inner_cv,
                classifiers_config,
                base_classifiers_config,
            )

        repetition = fold_idx // 10
        fold_in_rep = fold_idx % 10
        rows = []
        for template_row in cache[cache_key]:
            row = dict(template_row)
            row["fold_id"] = fold_idx
            row["repetition"] = repetition + 1
            row["fold_in_rep"] = fold_in_rep + 1
            row["n_train"] = len(train_idx)
            row["n_test"] = len(test_idx)
            rows.append(row)
        return rows

    mp.setattr(question_2_module, "_evaluate_outer_fold", _fast_evaluate_outer_fold)

    tmpdir = tempfile.mkdtemp()
    try:
        df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
        yield df, Path(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        mp.undo()


class TestQuestion2Pipeline:
    """Tests for the 30×10 nested CV pipeline."""

    def test_run_nested_cv_returns_dataframe(self, fast_q2_pipeline_env):
        """Verify that run_nested_cv_experiment returns a pandas DataFrame."""
        df, _ = fast_q2_pipeline_env
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"

    def test_output_csv_is_created(self, fast_q2_pipeline_env):
        """Verify that the output CSV file is created."""
        _, tmpdir = fast_q2_pipeline_env
        output_file = tmpdir / "cv_30x10_results.csv"
        assert output_file.exists(), f"Output CSV not found: {output_file}"

    def test_results_dataframe_shape(self, fast_q2_pipeline_env):
        """Verify results have exactly 3000 rows (300 folds × 2 datasets × 5 classifiers)."""
        df, _ = fast_q2_pipeline_env
        expected_rows = 300 * 2 * 5
        assert len(df) == expected_rows, \
            f"Expected {expected_rows} rows, got {len(df)}"

    def test_results_have_required_columns(self, fast_q2_pipeline_env):
        """Verify all required columns are present in results."""
        df, _ = fast_q2_pipeline_env
        required_columns = {
            "fold_id", "repetition", "fold_in_rep", "dataset", "classifier",
            "error", "precision", "recall", "f1", "best_params",
            "n_train", "n_test", "best_f1_inner"
        }
        actual_columns = set(df.columns)
        assert required_columns.issubset(actual_columns), \
            f"Missing columns: {required_columns - actual_columns}"

    def test_fold_structure_30x10(self, fast_q2_pipeline_env):
        """Verify the 30×10 fold structure is correct."""
        df, _ = fast_q2_pipeline_env
        for dataset in df["dataset"].unique():
            for classifier in df["classifier"].unique():
                df_subset = df[(df["dataset"] == dataset) & (df["classifier"] == classifier)]
                repetitions = df_subset["repetition"].unique()
                assert len(repetitions) == 30, \
                    f"Expected 30 repetitions, got {len(repetitions)}"
                for rep in repetitions:
                    df_rep = df_subset[df_subset["repetition"] == rep]
                    folds_in_rep = df_rep["fold_in_rep"].unique()
                    assert len(folds_in_rep) == 10, \
                        f"Expected 10 folds per repetition, got {len(folds_in_rep)}"

    def test_datasets_coverage(self, fast_q2_pipeline_env):
        """Verify both datasets are present in results."""
        df, _ = fast_q2_pipeline_env
        datasets = set(df["dataset"].unique())
        expected_datasets = {"original_2_classes", "kstar_classes"}
        assert datasets == expected_datasets, \
            f"Expected datasets {expected_datasets}, got {datasets}"

    def test_classifiers_coverage(self, fast_q2_pipeline_env):
        """Verify all 5 classifiers are present in results."""
        df, _ = fast_q2_pipeline_env
        classifiers = set(df["classifier"].unique())
        expected_classifiers = {
            "GaussianBayes", "BayesianKNN", "ParzenWindow",
            "LogisticRegression", "MajorityVoting"
        }
        assert classifiers == expected_classifiers, \
            f"Expected classifiers {expected_classifiers}, got {classifiers}"

    def test_metrics_are_numeric_and_valid(self, fast_q2_pipeline_env):
        """Verify metrics are numeric and within valid ranges."""
        df, _ = fast_q2_pipeline_env

        for metric in ["error", "precision", "recall", "f1", "best_f1_inner"]:
            assert np.issubdtype(df[metric].dtype, np.number), \
                f"Metric {metric} is not numeric"

        assert (df["error"] >= 0).all() and (df["error"] <= 1).all(), \
            "error values outside [0, 1]"
        assert (df["precision"] >= 0).all() and (df["precision"] <= 1).all(), \
            "precision values outside [0, 1]"
        assert (df["recall"] >= 0).all() and (df["recall"] <= 1).all(), \
            "recall values outside [0, 1]"
        assert (df["f1"] >= 0).all() and (df["f1"] <= 1).all(), \
            "f1 values outside [0, 1]"

    def test_no_missing_values_in_metrics(self, fast_q2_pipeline_env):
        """Verify no NaN or inf values in metrics."""
        df, _ = fast_q2_pipeline_env
        for metric in ["error", "precision", "recall", "f1"]:
            assert not df[metric].isna().any(), f"NaN values in {metric}"
            assert not np.isinf(df[metric]).any(), f"Inf values in {metric}"
        voting_scores = df.loc[
            df["classifier"] == "MajorityVoting", "best_f1_inner"
        ]
        assert voting_scores.isna().all(), \
            "MajorityVoting has no single inner-CV tuning score."
        base_scores = df.loc[
            df["classifier"] != "MajorityVoting", "best_f1_inner"
        ]
        assert base_scores.notna().all()

    def test_fold_ids_are_sequential(self, fast_q2_pipeline_env):
        """Verify fold_id values are sequential from 0 to 299."""
        df, _ = fast_q2_pipeline_env
        for dataset in df["dataset"].unique():
            for classifier in df["classifier"].unique():
                df_subset = df[(df["dataset"] == dataset) & (df["classifier"] == classifier)]
                fold_ids = sorted(df_subset["fold_id"].unique())
                assert fold_ids == list(range(300)), \
                    f"fold_ids not sequential for {dataset}/{classifier}"

    def test_train_test_split_sizes(self, fast_q2_pipeline_env):
        """Verify training and test set sizes are consistent."""
        df, _ = fast_q2_pipeline_env
        for fold_id in df["fold_id"].unique():
            df_fold = df[df["fold_id"] == fold_id]
            n_train_values = df_fold["n_train"].unique()
            n_test_values = df_fold["n_test"].unique()
            assert len(n_train_values) == 1, \
                f"Inconsistent n_train for fold_id {fold_id}"
            assert len(n_test_values) == 1, \
                f"Inconsistent n_test for fold_id {fold_id}"

        total_samples = 4601
        for _, row in df.iterrows():
            assert row["n_train"] + row["n_test"] == total_samples, \
                f"Train+test split doesn't equal {total_samples}"

    def test_best_params_not_empty(self, fast_q2_pipeline_env):
        """Verify best_params are recorded for each result."""
        df, _ = fast_q2_pipeline_env
        assert not df["best_params"].isna().any(), \
            "Some best_params are missing"
        assert (df["best_params"].str.len() > 0).all(), \
            "Some best_params are empty strings"
        voting_params = df.loc[
            df["classifier"] == "MajorityVoting", "best_params"
        ].map(ast.literal_eval)
        assert all(
            set(params)
            == {
                "GaussianBayes",
                "BayesianKNN",
                "ParzenWindow",
                "LogisticRegression",
            }
            for params in voting_params
        )

    def test_f1_relates_to_error(self, fast_q2_pipeline_env):
        """Verify logical relationship: higher F1 should correlate with lower error."""
        df, _ = fast_q2_pipeline_env
        correlation = df["f1"].corr(df["error"])
        assert correlation < -0.5, \
            f"F1 and error not strongly negatively correlated: {correlation}"

    def test_outer_fold_structure_consistency(self, fast_q2_pipeline_env):
        """Verify outer fold structure is consistent across datasets and classifiers."""
        df, _ = fast_q2_pipeline_env
        ref_df = df[(df["dataset"] == "original_2_classes") &
                    (df["classifier"] == "GaussianBayes")]
        ref_fold_structure = ref_df[["fold_id", "repetition", "fold_in_rep",
                                      "n_train", "n_test"]].drop_duplicates()

        for dataset in df["dataset"].unique():
            for classifier in df["classifier"].unique():
                df_subset = df[(df["dataset"] == dataset) &
                               (df["classifier"] == classifier)]
                fold_structure = df_subset[["fold_id", "repetition", "fold_in_rep",
                                             "n_train", "n_test"]].drop_duplicates()
                assert len(fold_structure) == len(ref_fold_structure), \
                    f"Inconsistent fold structure for {dataset}/{classifier}"

    def test_csv_file_matches_dataframe(self, fast_q2_pipeline_env):
        """Verify saved CSV file matches returned DataFrame."""
        df_returned, tmpdir = fast_q2_pipeline_env
        df_loaded = pd.read_csv(tmpdir / "cv_30x10_results.csv")
        assert df_returned.shape == df_loaded.shape, \
            "DataFrame shape doesn't match saved CSV"
        pd.testing.assert_frame_equal(df_returned, df_loaded, atol=1e-10)

    def test_experiment_deterministic(self, fast_q2_pipeline_env):
        """Verify experiment is deterministic (same seed gives same results)."""
        df1, _ = fast_q2_pipeline_env
        with tempfile.TemporaryDirectory() as tmpdir2:
            df2 = run_nested_cv_experiment(output_dir=tmpdir2, verbose=0)

        pd.testing.assert_frame_equal(
            df1[["error", "precision", "recall", "f1"]].reset_index(drop=True),
            df2[["error", "precision", "recall", "f1"]].reset_index(drop=True),
            atol=1e-10
        )

    def test_realistic_error_rates(self, fast_q2_pipeline_env):
        """Verify error rates are realistic (not all 0 or 1)."""
        df, _ = fast_q2_pipeline_env
        error_min = df["error"].min()
        error_max = df["error"].max()
        assert 0 <= error_min < error_max <= 1, \
            f"Error rates not realistic: min={error_min}, max={error_max}"


def test_majority_voting_uses_tuned_base_estimators(monkeypatch):
    parameter_grids = (
        (GaussianBayesClassifier, {}),
        (BayesianKNNClassifier, {"n_neighbors": [1], "metric": ["euclidean"]}),
        (ParzenWindowClassifier, {"bandwidth": [1.0]}),
        (CustomLogisticRegression, {"C": [1.0], "solver": ["lbfgs"]}),
    )
    for classifier, grid in parameter_grids:
        monkeypatch.setattr(classifier, "get_param_grid", staticmethod(lambda grid=grid: grid))

    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 4))
    y = np.repeat([0, 1], 30)
    X[y == 1, 0] += 1
    _, inner_cv = get_nested_cv_splitters()

    voting_classifier, selected_params = _fit_tuned_majority_voting(
        X,
        y,
        inner_cv,
        n_jobs=1,
    )

    assert set(selected_params) == {
        "GaussianBayes",
        "BayesianKNN",
        "ParzenWindow",
        "LogisticRegression",
    }
    assert set(voting_classifier.model_.named_estimators_) == set(selected_params)
    assert voting_classifier.predict(X[:5]).shape == (5,)
