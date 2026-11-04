"""Tests for dataloader functions, especially load_spambase_kstar_real()."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.utils.dataloader import (
    load_spambase_original,
    load_spambase_kstar_mock,
    load_spambase_kstar_real,
    get_nested_cv_splitters,
    get_learning_curve_splits,
)


class TestLoadSpambaseOriginal:
    """Tests for load_spambase_original()."""

    def test_returns_tuple(self):
        """Verify that load_spambase_original returns (X, y) tuple."""
        X, y = load_spambase_original()
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_shape_consistency(self):
        """Verify dataset shape: 4601 samples × 57 features."""
        X, y = load_spambase_original()
        assert X.shape[0] == 4601, f"Expected 4601 samples, got {X.shape[0]}"
        assert X.shape[1] == 57, f"Expected 57 features, got {X.shape[1]}"
        assert y.shape[0] == 4601, f"Expected 4601 labels, got {y.shape[0]}"

    def test_binary_labels(self):
        """Verify original labels are binary (0 or 1)."""
        X, y = load_spambase_original()
        unique_labels = np.unique(y)
        assert len(unique_labels) == 2, f"Expected 2 classes, got {len(unique_labels)}"
        assert set(unique_labels) == {0, 1}, f"Expected labels {{0, 1}}, got {set(unique_labels)}"

    def test_features_are_numeric(self):
        """Verify all features are numeric (float or int)."""
        X, _ = load_spambase_original()
        assert np.issubdtype(X.dtype, np.number), f"Expected numeric dtype, got {X.dtype}"


class TestLoadSpambaseKstarMock:
    """Tests for load_spambase_kstar_mock()."""

    def test_returns_tuple(self):
        """Verify that load_spambase_kstar_mock returns (X, y) tuple."""
        X, y = load_spambase_kstar_mock()
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_shape_consistency(self):
        """Verify mock labels preserve feature matrix size."""
        X, y = load_spambase_kstar_mock()
        assert X.shape[0] == y.shape[0], "Sample count mismatch"
        assert X.shape[0] == 4601, f"Expected 4601 samples, got {X.shape[0]}"

    def test_k_star_respected(self):
        """Verify mock labels respect k_star parameter."""
        k_star = 5
        X, y = load_spambase_kstar_mock(k_star=k_star)
        unique_labels = np.unique(y)
        assert len(unique_labels) <= k_star, \
            f"Expected ≤{k_star} unique labels, got {len(unique_labels)}"
        assert np.max(y) < k_star, f"Max label {np.max(y)} >= k_star {k_star}"

    def test_reproducibility(self):
        """Verify same random_state produces identical labels."""
        _, y1 = load_spambase_kstar_mock(random_state=42)
        _, y2 = load_spambase_kstar_mock(random_state=42)
        assert np.array_equal(y1, y2), "Same random_state should produce identical labels"

    def test_different_seeds_produce_different_labels(self):
        """Verify different random_state values produce different labels."""
        _, y1 = load_spambase_kstar_mock(random_state=42)
        _, y2 = load_spambase_kstar_mock(random_state=43)
        assert not np.array_equal(y1, y2), "Different seeds should produce different labels"


class TestLoadSpambaseKstarReal:
    """Tests for load_spambase_kstar_real()."""

    def test_returns_tuple(self):
        """Verify that load_spambase_kstar_real returns (X, y) tuple."""
        X, y = load_spambase_kstar_real()
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_shape_consistency(self):
        """Verify real K* labels preserve feature matrix size."""
        X, y = load_spambase_kstar_real()
        assert X.shape[0] == y.shape[0], "Sample count mismatch"
        assert X.shape[0] == 4601, f"Expected 4601 samples, got {X.shape[0]}"
        assert X.shape[1] == 57, f"Expected 57 features, got {X.shape[1]}"

    def test_labels_are_integers(self):
        """Verify K* labels are valid integer cluster indices."""
        X, y = load_spambase_kstar_real()
        assert np.issubdtype(y.dtype, np.integer), \
            f"Expected integer labels, got {y.dtype}"

    def test_k_star_equals_2(self):
        """Verify K* from Q1 selection is 2."""
        X, y = load_spambase_kstar_real()
        unique_labels = np.unique(y)
        k_star = len(unique_labels)
        assert k_star == 2, \
            f"Expected K*=2 from Q1, but got {k_star} unique clusters"

    def test_labels_are_0_1_indexed(self):
        """Verify cluster labels start from 0."""
        X, y = load_spambase_kstar_real()
        unique_labels = np.unique(y)
        assert np.min(unique_labels) == 0, \
            f"Expected min label 0, got {np.min(unique_labels)}"

    def test_labels_match_csv_exactly(self):
        """Verify loaded labels match the Q1 CSV file in order."""
        X, y = load_spambase_kstar_real()
        df_labels = pd.read_csv("results/q1_object_labels.csv")
        y_csv = df_labels["cluster"].values
        assert np.array_equal(y, y_csv), \
            "Loaded labels do not match CSV file"

    def test_label_count_matches_metadata(self):
        """Verify cluster count matches K* in q1_selection.json."""
        X, y = load_spambase_kstar_real()
        with open("results/q1_selection.json", "r") as f:
            metadata = json.load(f)
        k_star = metadata["K"]
        n_unique = len(np.unique(y))
        assert n_unique == k_star, \
            f"Expected {k_star} clusters from metadata, got {n_unique}"

    def test_missing_labels_file_raises_error(self):
        """Verify FileNotFoundError when q1_labels CSV is missing."""
        with pytest.raises(FileNotFoundError):
            load_spambase_kstar_real(q1_labels_path="nonexistent/path.csv")

    def test_missing_selection_file_raises_error(self):
        """Verify FileNotFoundError when q1_selection JSON is missing."""
        with pytest.raises(FileNotFoundError):
            load_spambase_kstar_real(q1_selection_path="nonexistent/path.json")

    def test_returns_original_features_not_standardized(self):
        """Verify returned features are raw, not standardized."""
        X, _ = load_spambase_kstar_real()
        X_orig, _ = load_spambase_original()
        # Features should be identical to original (no preprocessing in dataloader).
        assert np.allclose(X, X_orig), \
            "load_spambase_kstar_real should return raw features, not standardized"

    def test_does_not_return_mock_labels(self):
        """Verify real labels are different from mock random labels."""
        X, y_real = load_spambase_kstar_real()
        _, y_mock = load_spambase_kstar_mock(k_star=2, random_state=0)
        # Labels should be different (mock is random, real comes from Q1).
        assert not np.array_equal(y_real, y_mock), \
            "Real labels should differ from mock random labels"


class TestGetNestedCVSplitters:
    """Tests for get_nested_cv_splitters()."""

    def test_returns_tuple(self):
        """Verify that get_nested_cv_splitters returns a tuple."""
        result = get_nested_cv_splitters()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_outer_cv_properties(self):
        """Verify outer CV is RepeatedStratifiedKFold with 10 splits × 30 repeats."""
        outer_cv, _ = get_nested_cv_splitters()
        # Check cvargs which contains the n_splits parameter.
        assert outer_cv.cvargs["n_splits"] == 10
        assert outer_cv.n_repeats == 30

    def test_inner_cv_properties(self):
        """Verify inner CV is StratifiedKFold with 5 splits."""
        _, inner_cv = get_nested_cv_splitters()
        assert inner_cv.n_splits == 5


class TestGetLearningCurveSplits:
    """Tests for get_learning_curve_splits()."""

    def test_returns_dict(self):
        """Verify that get_learning_curve_splits returns a dictionary."""
        X, y = load_spambase_original()
        splits = get_learning_curve_splits(X, y)
        assert isinstance(splits, dict)

    def test_training_sizes_from_5_to_95_percent(self):
        """Verify training size keys range from 0.05 to 0.95 in 5% steps."""
        X, y = load_spambase_original()
        splits = get_learning_curve_splits(X, y)
        keys = sorted(splits.keys())
        expected_keys = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
        assert keys == expected_keys, \
            f"Expected keys {expected_keys}, got {keys}"

    def test_each_split_is_list_of_tuples(self):
        """Verify each training size maps to a list of (train_idx, test_idx) tuples."""
        X, y = load_spambase_original()
        splits = get_learning_curve_splits(X, y, n_repeats=5)
        for train_size, split_list in splits.items():
            assert isinstance(split_list, list), \
                f"Expected list for train_size {train_size}, got {type(split_list)}"
            assert len(split_list) == 5, \
                f"Expected 5 splits (n_repeats=5), got {len(split_list)}"
            for train_idx, test_idx in split_list:
                assert isinstance(train_idx, np.ndarray)
                assert isinstance(test_idx, np.ndarray)
