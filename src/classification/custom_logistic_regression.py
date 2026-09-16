import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    """Scikit-learn-compatible classifier based on logistic regression.

    This class wraps the scikit-learn implementation to preserve the project's
    classifier architecture.
    """
    def __init__(self, C=1.0, penalty='l2', solver='lbfgs', max_iter=10000):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        # A high max_iter value is required for convergence on the Spambase dataset.
        self.max_iter = max_iter 
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Fit the logistic regression model.

        Args:
            X: Training feature matrix.
            y: Target labels.

        Returns:
            The fitted classifier instance.
        """
        self.model_ = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=42
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        """Predict class labels for samples.

        Args:
            X: Feature matrix for the samples to classify.

        Returns:
            Predicted class labels.
        """
        return self.model_.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities for samples.

        Args:
            X: Feature matrix for the samples to classify.

        Returns:
            Class probabilities, useful for soft majority voting.
        """
        return self.model_.predict_proba(X)

    @staticmethod
    def get_param_grid():
        """Return the hyperparameter grid for cross-validation searches.

        Returns:
            A dictionary containing the candidate hyperparameter values.
        """
        return {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }