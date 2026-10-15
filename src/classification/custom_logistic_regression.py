from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

class CustomLogisticRegression(ClassifierMixin, BaseEstimator):
    """Scikit-learn-compatible classifier based on logistic regression.

    This class wraps the scikit-learn implementation to preserve the project's
    classifier architecture.
    """
    def __init__(self, C=1.0, solver='lbfgs', max_iter=10000):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter

    def fit(self, X, y):
        """Fit the logistic regression model.

        Args:
            X: Training feature matrix.
            y: Target labels.

        Returns:
            The fitted classifier instance.
        """
        X, y = validate_data(self, X, y)
        self.model_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=42
            )
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
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.model_.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities for samples.

        Args:
            X: Feature matrix for the samples to classify.

        Returns:
            Class probabilities, useful for soft majority voting.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.model_.predict_proba(X)

    @staticmethod
    def get_param_grid():
        """Return the hyperparameter grid for cross-validation searches.

        Returns:
            A dictionary containing the candidate hyperparameter values.
        """
        return {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['lbfgs']
        }