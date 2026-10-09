from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.utils.validation import check_is_fitted, validate_data


class MajorityVotingClassifier(ClassifierMixin, BaseEstimator):
    """Majority voting ensemble classifier.

    Combines the predictions of the Bayesian Gaussian, Bayesian k-NN,
    Parzen Window and Logistic Regression classifiers using hard voting.
    Ties are broken by the smallest class label (scikit-learn behavior).

    Args:
        estimators (list of tuples): List of (name, estimator) tuples for the
            base models.
    """

    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        """Fit the ensemble classifier using hard voting.

        Args:
            X (array-like of shape (n_samples, n_features)): Training vectors.
            y (array-like of shape (n_samples,)): Target values.

        Returns:
            self: The fitted classifier.
        """
        X, y = validate_data(self, X, y)
        self.model_ = VotingClassifier(estimators=self.estimators, voting='hard')
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        """Predict class labels for samples based on majority voting.

        Args:
            X (array-like of shape (n_queries, n_features)): Test samples.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.model_.predict(X)