import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class GaussianBayesClassifier(ClassifierMixin, BaseEstimator):
    """Bayesian classifier with multivariate normal class densities.

    Priors, means and covariances are ML estimates, but the density uses the
    regularized covariance Sigma_ML + lambda * I, so for lambda > 0 it is no
    longer a pure ML estimate. Computations use Cholesky factors in log-space.

    Args:
        reg_lambda (float, optional): Diagonal regularization, must be > 0.
            Defaults to 1e-4.
    """

    def __init__(self, reg_lambda=1e-4):
        self.reg_lambda = reg_lambda

    def fit(self, X, y):
        """Estimate priors, means and covariance Cholesky factors per class.

        Args:
            X (array-like of shape (n_samples, n_features)): Training vectors.
            y (array-like of shape (n_samples,)): Target values.

        Returns:
            self: The fitted classifier.
        """
        if not self.reg_lambda > 0:
            raise ValueError(f"reg_lambda must be > 0, got {self.reg_lambda!r}.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        self.priors_ = np.empty(n_classes)
        self.means_ = np.empty((n_classes, n_features))
        self.cholesky_factors_ = np.empty((n_classes, n_features, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]

            # ML estimates: P(w_i) = N_i / N; covariance divides by N_i (bias=True).
            self.priors_[i] = len(X_c) / n_samples
            self.means_[i] = X_c.mean(axis=0)
            cov = np.cov(X_c, rowvar=False, bias=True)
            cov = cov + self.reg_lambda * np.eye(n_features)

            try:
                self.cholesky_factors_[i] = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as err:
                raise ValueError(
                    f"Covariance of class {c!r} is not positive definite with "
                    f"reg_lambda={self.reg_lambda}. Increase reg_lambda."
                ) from err

        return self

    def _log_likelihood(self, X, mean, chol):
        """Compute log p(x | w_i) for every sample, given the class mean and lower 
        Cholesky factor L (L @ L.T = Sigma)."""
        n_features = X.shape[1]

       # Mahalanobis distance = ||z||^2, where L z = x - mu.
        z = solve_triangular(chol, (X - mean).T, lower=True)
        mahalanobis = np.sum(z ** 2, axis=0)

        # log|Sigma| = 2 * sum(log(diag(L))).
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))

        return -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + mahalanobis)

    def _joint_log_likelihood(self, X):
        """Compute log p(x | w_i) + log P(w_i) for every sample and class."""
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)

        jll = np.empty((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            jll[:, i] = (
                self._log_likelihood(X, self.means_[i], self.cholesky_factors_[i])
                + np.log(self.priors_[i])
            )
        return jll

    def predict(self, X):
        """Predict the class with the highest posterior (Bayes' rule).

        The evidence is common to all classes, so maximizing log p(x | w_i) + log P(w_i) is enough.

        Args:
            X (array-like of shape (n_queries, n_features)): Test samples.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        """Return posteriors P(w_i | x), normalized with logsumexp to avoid underflow.

        Args:
            X (array-like of shape (n_queries, n_features)): Test samples.

        Returns:
            numpy.ndarray: Posterior probability estimates.
        """
        jll = self._joint_log_likelihood(X)
        return np.exp(jll - logsumexp(jll, axis=1, keepdims=True))

    @staticmethod
    def get_param_grid():
        """Return the `reg_lambda` grid for cross-validation searches."""
        return {
            'reg_lambda': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        }