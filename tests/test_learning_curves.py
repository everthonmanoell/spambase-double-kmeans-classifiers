"""Tests for Question 2 learning-curve generation."""

import json

import numpy as np
import pandas as pd
import pytest

import src.experiments.learning_curves as learning_curves


def _synthetic_dataset():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4))
    y = np.repeat([0, 1], 50)
    X[y == 1] += 1.5
    return X, y


def test_learning_curve_experiment_saves_data_and_plots(tmp_path, monkeypatch):
    dataset = _synthetic_dataset()
    monkeypatch.setattr(learning_curves, "load_spambase_original", lambda: dataset)
    monkeypatch.setattr(learning_curves, "load_spambase_kstar_real", lambda: dataset)
    monkeypatch.setattr(learning_curves, "DEFAULT_TRAIN_SIZES", (0.2, 0.5))

    raw_results, summary = learning_curves.run_learning_curve_experiment(
        results_dir=tmp_path / "results",
        plots_dir=tmp_path / "plots",
        n_repeats=2,
    )

    assert len(raw_results) == 2 * 5 * 2 * 2
    assert len(summary) == 2 * 5 * 2
    assert set(raw_results["repeat"]) == {1, 2}
    assert set(raw_results["inner_cv_splits"]) == {5}
    assert raw_results["best_params"].map(bool).all()
    assert all(
        "GaussianBayes" in params
        and "BayesianKNN" in params
        and "ParzenWindow" in params
        and "LogisticRegression" in params
        for params in raw_results.loc[
            raw_results["classifier"] == "MajorityVoting", "best_params"
        ].map(json.loads)
    )
    assert raw_results[["train_f1", "test_f1"]].ge(0).all().all()
    assert raw_results[["train_f1", "test_f1"]].le(1).all().all()
    assert (tmp_path / "results" / "learning_curves_data.csv").is_file()
    assert (tmp_path / "results" / "learning_curves_summary.csv").is_file()
    assert (tmp_path / "plots" / "learning_curves_original.png").is_file()
    assert (tmp_path / "plots" / "learning_curves_kstar.png").is_file()

    saved_results = pd.read_csv(tmp_path / "results" / "learning_curves_data.csv")
    assert len(saved_results) == len(raw_results)


def test_learning_curve_requires_two_repeats(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        learning_curves.run_learning_curve_experiment(
            results_dir=tmp_path / "results",
            plots_dir=tmp_path / "plots",
            n_repeats=1,
        )


def test_tuning_uses_stratified_cv_for_all_base_classifiers():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 4))
    y = np.repeat([0, 1], 50)
    X[y == 1, 0] += 1

    tuned_estimators, selected_params = learning_curves._tune_classifiers(
        X,
        y,
        random_state=11,
    )

    assert {name for name, _ in tuned_estimators} == set(learning_curves.CLASSIFIER_NAMES[:4])
    assert set(selected_params) == set(learning_curves.CLASSIFIER_NAMES[:4])
    assert selected_params["GaussianBayes"] == {}
    assert selected_params["BayesianKNN"]["n_neighbors"] <= 80


def test_learning_curve_tuning_handles_rare_class_with_numerical_stabilization():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 4))
    y = np.array([0] * 98 + [1] * 2)

    tuned_estimators, selected_params = learning_curves._tune_classifiers(
        X, y, random_state=11
    )
    assert "GaussianBayes" in selected_params
    assert selected_params["GaussianBayes"] == {}


def test_gaussian_classifier_rejects_singular_mle_covariance_when_zero_reg_covar():
    import pytest
    from src.classification.bayes_gaussian import GaussianBayesClassifier

    X = np.array([[0.0, 0.0], [0.2, 0.2], [1.0, 1.0], [1.2, 1.2]])
    y = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="unregularized multivariate"):
        GaussianBayesClassifier(reg_covar=0.0).fit(X, y)
