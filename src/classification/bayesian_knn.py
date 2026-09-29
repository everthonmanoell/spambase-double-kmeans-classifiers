import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

class BayesianKNNClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian classifier based on k-nearest neighbors.

    This class encapsulates the scikit-learn KNeighborsClassifier, evaluating 
    Euclidean, City-Block, and Chebyshev distances.

    Args:
        n_neighbors (int, optional): Number of neighbors to use. Defaults to 5.
        metric (str, optional): The distance metric to use. Defaults to 'euclidean'.
    """

    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Fits the model based on the k-nearest neighbors.

        Args:
            X (array-like of shape (n_samples, n_features)): Training vectors.
            y (array-like of shape (n_samples,)): Target values.

        Returns:
            self: The fitted classifier.
        """
        self.model_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self.metric
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        """Predicts the class labels for the provided data.

        Args:
            X (array-like of shape (n_queries, n_features)): Test samples.

        Returns:
            numpy.ndarray: Predicted class labels for each data sample.
        """
        return self.model_.predict(X)
        
    def predict_proba(self, X):
        """Returns the posterior probabilities for each class.
        
        The probabilities are calculated through the mathematical proportion k_i / k.

        Args:
            X (array-like of shape (n_queries, n_features)): Test samples.
            
        Returns:
            numpy.ndarray: Posterior probability estimates.
        """
        return self.model_.predict_proba(X)

    @staticmethod
    def get_param_grid():
        """Returns the hyperparameter search grid.

        Used to fix the number of neighbors (k) and the distance metric via 
        cross-validation.

        Returns:
            dict: Dictionary containing the parameter grid for `n_neighbors` 
            and `metric`.
        """
        return {
            'n_neighbors': [1, 3, 5, 7, 11, 15, 21, 31],
            'metric': ['euclidean', 'cityblock', 'chebyshev']
        }