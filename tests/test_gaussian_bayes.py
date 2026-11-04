import numpy as np
import pytest

from src.classification.bayes_gaussian import GaussianBayesClassifier


def test_covariances_are_unregularized_maximum_likelihood_estimates():
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.2],
        [0.2, 1.0],
        [3.0, 3.0],
        [4.0, 3.2],
        [3.2, 4.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    classifier = GaussianBayesClassifier().fit(X, y)

    for index, label in enumerate(classifier.classes_):
        class_samples = X[y == label]
        centered = class_samples - class_samples.mean(axis=0)
        expected = centered.T @ centered / len(class_samples)
        np.testing.assert_allclose(classifier.covariances_[index], expected)

    probabilities = classifier.predict_proba(X)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert GaussianBayesClassifier.get_param_grid() == {}


def test_singular_maximum_likelihood_covariance_stabilized_by_default():
    X = np.array([
        [0.0, 0.0],
        [0.2, 0.2],
        [1.0, 1.0],
        [1.2, 1.2],
    ])
    y = np.array([0, 0, 1, 1])

    classifier = GaussianBayesClassifier().fit(X, y)
    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)

    assert np.array_equal(predictions, y)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_singular_maximum_likelihood_covariance_is_reported_when_unregularized():
    X = np.array([
        [0.0, 0.0],
        [0.2, 0.2],
        [1.0, 1.0],
        [1.2, 1.2],
    ])
    y = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="unregularized multivariate"):
        GaussianBayesClassifier(reg_covar=0.0).fit(X, y)
