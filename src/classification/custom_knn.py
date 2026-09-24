import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

class CustomKNN(BaseEstimator, ClassifierMixin):
    """K-Nearest Neighbors classifier wrapper.

    This class encapsulates the scikit-learn KNeighborsClassifier to maintain 
    the project's architectural standard. It evaluates Euclidean, City-Block, 
    and Chebyshev distances for the neighborhood.

    Args:
        n_neighbors (int): Number of neighbors to use. Defaults to 5.
        metric (str): The distance metric to use. Defaults to 'euclidean'.
    """

    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Fits the k-nearest neighbors classifier from the training dataset.

        Args:
            X (array-like of shape (n_samples, n_features)): Training data.
            y (array-like of shape (n_samples,)): Target values.

        Returns:
            self: The fitted K-Nearest Neighbors classifier.
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
            ndarray of shape (n_queries,): Class labels for each data sample.
        """
        return self.model_.predict(X)
        
    def predict_proba(self, X):
        """Return probability estimates for the test data X.
        
        Args:
            X (array-like of shape (n_queries, n_features)): Test samples.
            
        Returns:
            ndarray of shape (n_queries, n_classes): Probability estimates.
        """
        return self.model_.predict_proba(X)

    @staticmethod
    def get_param_grid():
        """Returns the hyperparameter grid for cross-validation search.

        Returns:
            dict: A dictionary containing the parameter grid for n_neighbors 
            and distance metrics[cite: 4].
        """
        return {
            'n_neighbors': [1, 3, 5, 7, 11, 15, 21, 31],
            'metric': ['euclidean', 'cityblock', 'chebyshev']
        }