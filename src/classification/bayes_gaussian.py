import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class GaussianBayesClassifier(ClassifierMixin, BaseEstimator):
    """Bayesian classifier with multivariate normal class densities.

    Class priors, means, and covariance matrices use maximum-likelihood
    estimates (dividing by N_i). A small non-negative diagonal term
    ``reg_covar`` (default ``1e-6``) is added when computing the Cholesky
    factorization to guarantee positive definiteness when a class covariance
    is rank-deficient (e.g. N_i <= d or zero-variance features in small splits).

    Args:
        reg_covar (float, optional): Non-negative diagonal stabilization added
            to each class covariance matrix before Cholesky factorization.
            Defaults to 1e-6.
    """

    def __init__(self, reg_covar: float = 1e-6):
        self.reg_covar = reg_covar

    def fit(self, X, y):
        """Estimate class priors, means, covariances, and Cholesky factors.

        Args:
            X (array-like of shape (n_samples, n_features)): Training vectors.
            y (array-like of shape (n_samples,)): Target values.

        Returns:
            GaussianBayesClassifier: The fitted classifier.

        Raises:
            ValueError: If ``reg_covar`` is negative or if a class covariance
                matrix is singular or not positive definite.
        """
        if self.reg_covar < 0:
            raise ValueError(f"reg_covar must be non-negative, got {self.reg_covar}.")

        # Also stores n_features_in_ and rejects NaN/inf values.
        X, y = validate_data(self, X, y, dtype=float)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        self.priors_ = np.empty(n_classes)
        self.means_ = np.empty((n_classes, n_features))
        self.covariances_ = np.empty((n_classes, n_features, n_features))
        self.cholesky_factors_ = np.empty((n_classes, n_features, n_features))

        eye = np.eye(n_features)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]

            # ML estimates: P(w_i) = N_i / N; covariance divides by N_i (bias=True).
            self.priors_[i] = len(X_c) / n_samples
            self.means_[i] = X_c.mean(axis=0)
            centered = X_c - self.means_[i]
            cov = centered.T @ centered / len(X_c)
            self.covariances_[i] = cov

            stabilized_cov = cov + self.reg_covar * eye

            if self.reg_covar == 0.0 and np.linalg.matrix_rank(centered) < n_features:
                raise ValueError(
                    f"Maximum-likelihood covariance of class {c!r} is singular "
                    "or not positive definite; the unregularized multivariate "
                    "Gaussian density is undefined for this training fold."
                )

            try:
                self.cholesky_factors_[i] = np.linalg.cholesky(stabilized_cov)
            except np.linalg.LinAlgError as err:
                raise ValueError(
                    f"Maximum-likelihood covariance of class {c!r} is singular "
                    "or not positive definite; the multivariate Gaussian "
                    "density is undefined for this training fold."
                ) from err

        return self

    def _log_likelihood(self, X, mean, chol):
        """Compute log p(x | w_i) for every sample using a class Cholesky factor.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            mean (np.ndarray): Class mean vector of shape (n_features,).
            chol (np.ndarray): Lower Cholesky factor L of shape
                (n_features, n_features) such that L @ L.T = Sigma.

        Returns:
            np.ndarray: Log-density values of shape (n_samples,).
        """
        n_features = X.shape[1]

        # Mahalanobis distance = ||z||^2, where L z = x - mu.
        z = solve_triangular(chol, (X - mean).T, lower=True)
        mahalanobis = np.sum(z ** 2, axis=0)

        # log|Sigma| = 2 * sum(log(diag(L))).
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))

        return -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + mahalanobis)

    def _joint_log_likelihood(self, X):
        """Compute log p(x | w_i) + log P(w_i) for every sample and class.

        Args:
            X (array-like of shape (n_queries, n_features)): Query samples.

        Returns:
            np.ndarray: Joint log-likelihood matrix of shape
                (n_queries, n_classes).
        """
        check_is_fitted(self)
        # reset=False checks that X has the same number of features as in fit.
        X = validate_data(self, X, dtype=float, reset=False)

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
        """Return the empty hyperparameter grid for cross-validation.

        Returns:
            dict: Empty dictionary because the maximum-likelihood Gaussian model
                has no tunable hyperparameters.
        """
        return {}