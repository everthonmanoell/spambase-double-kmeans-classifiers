import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity

class ParzenWindowClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian classifier based on the Parzen window method using KDE.

    The classifier uses a multivariate product kernel composed of univariate
    Gaussian kernels.

    Args:
        bandwidth: Bandwidth used by each kernel density estimator.
    """
    def __init__(self, bandwidth=1.0):
        # Bandwidth h required by the project.
        self.bandwidth = bandwidth
        self.classes_ = None
        self.kdes_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        """
        Fit a density estimator for each class and calculate prior probabilities.

        Args:
            X: Training samples.
            y: Target class labels.

        Returns:
            The fitted classifier.
        """
        self.classes_ = np.unique(y)
        n_total = len(y)
        
        for c in self.classes_:
            # Filter samples belonging to the current class.
            X_c = X[y == c]
            
            # Maximum likelihood estimate for P(w_i): P(w_i) = N_i / N_total.
            self.priors_[c] = len(X_c) / n_total
            
            # Fit the kernel density estimator for class c.
            kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
            kde.fit(X_c)
            self.kdes_[c] = kde
            
        return self

    def predict(self, X):
        """
        Predict classes using Bayes' rule.

        The posterior probability is proportional to p(x | w_i) * P(w_i).

        Args:
            X: Samples to classify.

        Returns:
            Predicted class labels for each sample.
        """
        # Store the log probabilities for each class.
        log_probs = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, c in enumerate(self.classes_):
            # score_samples returns the log density: log(p(x | w_i)).
            log_density = self.kdes_[c].score_samples(X)
            
            # Add the log prior to obtain the log posterior probability.
            # log(p(x | w_i) * P(w_i)) = log(p(x | w_i)) + log(P(w_i)).
            log_probs[:, i] = log_density + np.log(self.priors_[c])
            
        # Return the class with the highest posterior probability.
        return self.classes_[np.argmax(log_probs, axis=1)]