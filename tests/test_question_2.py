"""Tests for Question 2: nested cross-validation experiment."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil

from src.experiments.question_2 import run_nested_cv_experiment


class TestQuestion2Pipeline:
    """Tests for the 30×10 nested CV pipeline."""

    def test_run_nested_cv_returns_dataframe(self):
        """Verify that run_nested_cv_experiment returns a pandas DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"

    def test_output_csv_is_created(self):
        """Verify that the output CSV file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            output_file = Path(tmpdir) / "cv_30x10_results.csv"
            assert output_file.exists(), f"Output CSV not found: {output_file}"

    def test_results_dataframe_shape(self):
        """Verify results have exactly 3000 rows (300 folds × 2 datasets × 5 classifiers)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            expected_rows = 300 * 2 * 5
            assert len(df) == expected_rows, \
                f"Expected {expected_rows} rows, got {len(df)}"

    def test_results_have_required_columns(self):
        """Verify all required columns are present in results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            required_columns = {
                "fold_id", "repetition", "fold_in_rep", "dataset", "classifier",
                "error", "precision", "recall", "f1", "best_params",
                "n_train", "n_test", "best_f1_inner"
            }
            actual_columns = set(df.columns)
            assert required_columns.issubset(actual_columns), \
                f"Missing columns: {required_columns - actual_columns}"

    def test_fold_structure_30x10(self):
        """Verify the 30×10 fold structure is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            # Extract unique (repetition, fold_in_rep) combinations per dataset/classifier
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

    def test_datasets_coverage(self):
        """Verify both datasets are present in results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            datasets = set(df["dataset"].unique())
            expected_datasets = {"original_2_classes", "kstar_classes"}
            assert datasets == expected_datasets, \
                f"Expected datasets {expected_datasets}, got {datasets}"

    def test_classifiers_coverage(self):
        """Verify all 5 classifiers are present in results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            classifiers = set(df["classifier"].unique())
            expected_classifiers = {
                "GaussianBayes", "BayesianKNN", "ParzenWindow",
                "LogisticRegression", "MajorityVoting"
            }
            assert classifiers == expected_classifiers, \
                f"Expected classifiers {expected_classifiers}, got {classifiers}"

    def test_metrics_are_numeric_and_valid(self):
        """Verify metrics are numeric and within valid ranges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)

            # All metrics should be numeric
            for metric in ["error", "precision", "recall", "f1", "best_f1_inner"]:
                assert np.issubdtype(df[metric].dtype, np.number), \
                    f"Metric {metric} is not numeric"

            # Check ranges
            assert (df["error"] >= 0).all() and (df["error"] <= 1).all(), \
                "error values outside [0, 1]"
            assert (df["precision"] >= 0).all() and (df["precision"] <= 1).all(), \
                "precision values outside [0, 1]"
            assert (df["recall"] >= 0).all() and (df["recall"] <= 1).all(), \
                "recall values outside [0, 1]"
            assert (df["f1"] >= 0).all() and (df["f1"] <= 1).all(), \
                "f1 values outside [0, 1]"

    def test_no_missing_values_in_metrics(self):
        """Verify no NaN or inf values in metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            for metric in ["error", "precision", "recall", "f1", "best_f1_inner"]:
                assert not df[metric].isna().any(), f"NaN values in {metric}"
                assert not np.isinf(df[metric]).any(), f"Inf values in {metric}"

    def test_fold_ids_are_sequential(self):
        """Verify fold_id values are sequential from 0 to 299."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            # Each dataset-classifier pair should have fold_ids 0-299
            for dataset in df["dataset"].unique():
                for classifier in df["classifier"].unique():
                    df_subset = df[(df["dataset"] == dataset) & (df["classifier"] == classifier)]
                    fold_ids = sorted(df_subset["fold_id"].unique())
                    assert fold_ids == list(range(300)), \
                        f"fold_ids not sequential for {dataset}/{classifier}"

    def test_train_test_split_sizes(self):
        """Verify training and test set sizes are consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            # All rows with same fold_id should have same n_train and n_test
            for fold_id in df["fold_id"].unique():
                df_fold = df[df["fold_id"] == fold_id]
                n_train_values = df_fold["n_train"].unique()
                n_test_values = df_fold["n_test"].unique()
                assert len(n_train_values) == 1, \
                    f"Inconsistent n_train for fold_id {fold_id}"
                assert len(n_test_values) == 1, \
                    f"Inconsistent n_test for fold_id {fold_id}"

            # Check that n_train + n_test = 4601
            total_samples = 4601
            for _, row in df.iterrows():
                assert row["n_train"] + row["n_test"] == total_samples, \
                    f"Train+test split doesn't equal {total_samples}"

    def test_best_params_not_empty(self):
        """Verify best_params are recorded for each result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            assert not df["best_params"].isna().any(), \
                "Some best_params are missing"
            assert (df["best_params"].str.len() > 0).all(), \
                "Some best_params are empty strings"

    def test_f1_relates_to_error(self):
        """Verify logical relationship: higher F1 should correlate with lower error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            # Correlation should be strongly negative
            correlation = df["f1"].corr(df["error"])
            assert correlation < -0.5, \
                f"F1 and error not strongly negatively correlated: {correlation}"

    def test_outer_fold_structure_consistency(self):
        """Verify outer fold structure is consistent across datasets and classifiers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            # Get reference fold structure from first classifier and dataset
            ref_df = df[(df["dataset"] == "original_2_classes") & 
                        (df["classifier"] == "GaussianBayes")]
            ref_fold_structure = ref_df[["fold_id", "repetition", "fold_in_rep", 
                                          "n_train", "n_test"]].drop_duplicates()

            # Check all other combinations have the same structure
            for dataset in df["dataset"].unique():
                for classifier in df["classifier"].unique():
                    df_subset = df[(df["dataset"] == dataset) & 
                                   (df["classifier"] == classifier)]
                    fold_structure = df_subset[["fold_id", "repetition", "fold_in_rep", 
                                                 "n_train", "n_test"]].drop_duplicates()
                    assert len(fold_structure) == len(ref_fold_structure), \
                        f"Inconsistent fold structure for {dataset}/{classifier}"

    def test_csv_file_matches_dataframe(self):
        """Verify saved CSV file matches returned DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df_returned = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            df_loaded = pd.read_csv(Path(tmpdir) / "cv_30x10_results.csv")
            assert df_returned.shape == df_loaded.shape, \
                "DataFrame shape doesn't match saved CSV"
            # Compare values (allowing small floating-point differences)
            pd.testing.assert_frame_equal(df_returned, df_loaded, atol=1e-10)

    def test_experiment_deterministic(self):
        """Verify experiment is deterministic (same seed gives same results)."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            df1 = run_nested_cv_experiment(output_dir=tmpdir1, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir2:
            df2 = run_nested_cv_experiment(output_dir=tmpdir2, verbose=0)

        # Results should be identical
        pd.testing.assert_frame_equal(
            df1[["error", "precision", "recall", "f1"]].reset_index(drop=True),
            df2[["error", "precision", "recall", "f1"]].reset_index(drop=True),
            atol=1e-10
        )

    def test_realistic_error_rates(self):
        """Verify error rates are realistic (not all 0 or 1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = run_nested_cv_experiment(output_dir=tmpdir, verbose=0)
            error_min = df["error"].min()
            error_max = df["error"].max()
            # Error should be between 0 and 1 and have some variation
            assert 0 < error_min < error_max < 1, \
                f"Error rates not realistic: min={error_min}, max={error_max}"
