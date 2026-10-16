import numbers

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data


class DoubleKMeans(ClusterMixin, BaseEstimator):
    """Double k-means (Vichi, 2001): hard partition of objects and variables.

    Minimizes W(G, U, V) with the three steps of the course slides. Each `fit`
    is one run from a random start. Design decisions (ties, empty groups,
    stopping) are in src/clustering/README.md.

    Args:
        n_row_clusters (int, optional): Number of object groups K. Defaults to 2.
        n_col_clusters (int, optional): Number of variable groups H. Defaults to 2.
        max_iter (int, optional): Safety limit on iterations, must be >= 1.
            Defaults to 100.
        random_state (int, RandomState or None, optional): Seed of the random
            initial partitions. Defaults to None.
    """

    def __init__(self, n_row_clusters=2, n_col_clusters=2, max_iter=100,
                 random_state=None):
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        """Run double k-means from a random initial partition.

        Args:
            X (array-like of shape (n_samples, n_features)): Data matrix.
            y: Ignored, present for API consistency.

        Returns:
            self: The fitted estimator.

        Raises:
            ValueError: If `n_row_clusters`, `n_col_clusters` or `max_iter`
                is out of range.
        """
        if not (isinstance(self.max_iter, numbers.Integral) and self.max_iter >= 1):
            raise ValueError(
                f"max_iter must be an integer >= 1, got {self.max_iter!r}."
            )

        X = validate_data(self, X, dtype=float)
        n_samples, n_features = X.shape
        K, H = self.n_row_clusters, self.n_col_clusters
        if not 1 <= K <= n_samples:
            raise ValueError(f"n_row_clusters must be in [1, {n_samples}], got {K}.")
        if not 1 <= H <= n_features:
            raise ValueError(f"n_col_clusters must be in [1, {n_features}], got {H}.")

        rng = check_random_state(self.random_state)

        row_labels = self._random_partition(n_samples, K, rng)
        col_labels = self._random_partition(n_features, H, rng)
        G = self._update_prototypes(X, row_labels, col_labels, K, H)
        history = [self._objective(X, G, row_labels, col_labels)]

        converged = False
        n_relocations = 0
        for n_iter in range(1, self.max_iter + 1):
            # Step 1: prototypes.
            G = self._update_prototypes(X, row_labels, col_labels, K, H)

            # Step 2: objects (costs without the x_ij^2 term).
            col_onehot = np.eye(H)[col_labels]
            n_h = col_onehot.sum(axis=0)
            row_cost = -2.0 * (X @ col_onehot) @ G.T + (G ** 2) @ n_h
            new_row_labels = self._assign(row_cost, row_labels)
            new_row_labels, G, n_moved = self._relocate_empty_groups(
                X, G, new_row_labels, col_labels)
            n_relocations += n_moved

            # Step 3: variables, with the G of step 2.
            row_onehot = np.eye(K)[new_row_labels]
            n_k = row_onehot.sum(axis=0)
            col_cost = -2.0 * (row_onehot.T @ X).T @ G + n_k @ (G ** 2)
            new_col_labels = self._assign(col_cost, col_labels)
            new_col_labels, G_t, n_moved = self._relocate_empty_groups(
                X.T, G.T, new_col_labels, new_row_labels)
            G = G_t.T
            n_relocations += n_moved

            changed = (np.any(new_row_labels != row_labels)
                       or np.any(new_col_labels != col_labels))
            row_labels, col_labels = new_row_labels, new_col_labels

            # Update G for the current partition before saving it in the history.
            G_opt = self._update_prototypes(X, row_labels, col_labels, K, H)
            history.append(self._objective(X, G_opt, row_labels, col_labels))

            if not changed:
                converged = True
                break

        G = self._update_prototypes(X, row_labels, col_labels, K, H)

        self.row_labels_ = row_labels
        self.column_labels_ = col_labels
        self.labels_ = row_labels
        self.prototypes_ = G
        self.objective_ = self._objective(X, G, row_labels, col_labels)
        self.objective_history_ = np.array(history)
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.n_relocations_ = n_relocations
        return self

    @staticmethod
    def _assign(cost, current_labels):
        """Move each item to its lowest-cost group only on a strict improvement."""
        best = np.argmin(cost, axis=1)
        items = np.arange(len(current_labels))
        improves = cost[items, best] < cost[items, current_labels]
        return np.where(improves, best, current_labels)

    @staticmethod
    def _relocate_empty_groups(X, G, labels, other_labels):
        """Refill groups emptied by steps 2 and 3.

        Written for objects; variables pass X.T and G.T.

        Returns:
            tuple: (labels, G, number of relocated items).
        """
        n_groups, n_other_groups = G.shape
        counts = np.bincount(labels, minlength=n_groups)
        empty_groups = np.flatnonzero(counts == 0)
        if empty_groups.size == 0:
            return labels, G, 0

        labels, G = labels.copy(), G.copy()
        other_sizes = np.bincount(other_labels, minlength=n_other_groups)
        for group in empty_groups:
            cost = np.sum((X - G[np.ix_(labels, other_labels)]) ** 2, axis=1)
            # Donors keep at least one member.
            cost[counts[labels] < 2] = -np.inf
            item = np.argmax(cost)

            counts[labels[item]] -= 1
            counts[group] += 1
            labels[item] = group
            G[group] = np.bincount(other_labels, weights=X[item],
                                   minlength=n_other_groups) / other_sizes

        return labels, G, empty_groups.size

    @staticmethod
    def _random_partition(n_items, n_groups, rng):
        """Assign items to groups at random, with no empty group."""
        labels = rng.randint(n_groups, size=n_items)
        labels[rng.permutation(n_items)[:n_groups]] = np.arange(n_groups)
        return labels

    @staticmethod
    def _update_prototypes(X, row_labels, col_labels, K, H):
        """Step 1: g_kh = sum of block kh / (n_k * n_h)."""
        row_onehot = np.eye(K)[row_labels]
        col_onehot = np.eye(H)[col_labels]
        block_sums = row_onehot.T @ X @ col_onehot
        block_sizes = np.outer(row_onehot.sum(axis=0), col_onehot.sum(axis=0))

        # Safety check: an empty block would put NaN in G.
        if np.any(block_sizes == 0):
            raise RuntimeError("Empty group in the prototype update.")

        return block_sums / block_sizes

    @staticmethod
    def _objective(X, G, row_labels, col_labels):
        """W(G, U, V): squared distance of each x_ij to its block prototype."""
        return float(np.sum((X - G[np.ix_(row_labels, col_labels)]) ** 2))
