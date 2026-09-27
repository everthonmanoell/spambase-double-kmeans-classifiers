import numbers

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data


class DoubleKMeans(ClusterMixin, BaseEstimator):
    """Double k-means: simultaneous hard partition of objects and variables.

    Minimizes W(G, U, V) = sum_k sum_h sum_i sum_j u_ik v_jh (x_ij - g_kh)^2,
    alternating the three steps from the course slides until no object and no
    variable changes group:
        1. G fixed U, V: g_kh is the mean of block kh.
        2. U fixed G, V: each object goes to the closest object group.
        3. V fixed G, U: each variable goes to the closest variable group.

    The slides do not cover empty groups. When step 2 or 3 empties a group, it
    receives the highest-cost item of a group with at least 2 members, so the
    result always has K object groups and H variable groups and W does not
    increase. `n_relocations_` counts these moves.

    A single call to `fit` is one run from one random start. The project runs
    it 100 times per (K, H) and keeps the run with the smallest `objective_`.

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
        # Same constraint as scikit-learn's KMeans: an integer >= 1.
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

        # Initialization: random U and V without empty groups; G from step 1.
        row_labels = self._random_partition(n_samples, K, rng)
        col_labels = self._random_partition(n_features, H, rng)
        G = self._update_prototypes(X, row_labels, col_labels, K, H)
        history = [self._objective(X, G, row_labels, col_labels)]

        converged = False
        n_relocations = 0
        for n_iter in range(1, self.max_iter + 1):
            # Step 1: optimal prototypes for fixed U and V.
            G = self._update_prototypes(X, row_labels, col_labels, K, H)

            # Step 2: object cost_ik = sum_h sum_{j in Q_h} (x_ij - g_kh)^2,
            # dropping sum_j x_ij^2, which does not depend on k.
            col_onehot = np.eye(H)[col_labels]
            n_h = col_onehot.sum(axis=0)
            row_cost = -2.0 * (X @ col_onehot) @ G.T + (G ** 2) @ n_h
            new_row_labels = np.argmin(row_cost, axis=1)
            new_row_labels, G, n_moved = self._relocate_empty_groups(
                X, G, new_row_labels, col_labels)
            n_relocations += n_moved

            # Step 3: variable cost_jh = sum_k sum_{i in P_k} (x_ij - g_kh)^2,
            # with the same G (the slides do not recompute G between 2 and 3;
            # only the prototypes of refilled groups changed).
            row_onehot = np.eye(K)[new_row_labels]
            n_k = row_onehot.sum(axis=0)
            col_cost = -2.0 * (row_onehot.T @ X).T @ G + n_k @ (G ** 2)
            new_col_labels = np.argmin(col_cost, axis=1)
            # Same relocation for variables: the transposed problem.
            new_col_labels, G_t, n_moved = self._relocate_empty_groups(
                X.T, G.T, new_col_labels, new_row_labels)
            G = G_t.T
            n_relocations += n_moved

            changed = (np.any(new_row_labels != row_labels)
                       or np.any(new_col_labels != col_labels))
            row_labels, col_labels = new_row_labels, new_col_labels
            history.append(self._objective(X, G, row_labels, col_labels))

            if not changed:
                converged = True
                break

        # Final prototypes consistent with the final partitions.
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
    def _relocate_empty_groups(X, G, labels, other_labels):
        """Refill groups emptied by an assignment step (steps 2 and 3).

        Written for objects (rows of X); variables use X.T and G.T. Each empty
        group receives the item with the largest cost to its own prototype,
        taken from a group with at least 2 members, and its prototypes become
        that item's means over the other partition. The moved item's cost can
        only drop, so W does not increase (same idea as scikit-learn's KMeans).

        Args:
            X (numpy.ndarray of shape (n_items, n_other)): Data matrix.
            G (numpy.ndarray of shape (n_groups, n_other_groups)): Prototypes.
            labels (numpy.ndarray of shape (n_items,)): Groups of the items.
            other_labels (numpy.ndarray of shape (n_other,)): Groups of the
                other partition, all non-empty.

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
            # Donors keep at least one member, so no new empty group appears.
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
        """Assign items to groups at random, guaranteeing no empty group."""
        labels = rng.randint(n_groups, size=n_items)
        # The first n_groups positions of a random permutation get one item
        # per group, so every group has at least one member.
        labels[rng.permutation(n_items)[:n_groups]] = np.arange(n_groups)
        return labels

    @staticmethod
    def _update_prototypes(X, row_labels, col_labels, K, H):
        """Step 1: g_kh = sum of block kh / (n_k * n_h)."""
        row_onehot = np.eye(K)[row_labels]
        col_onehot = np.eye(H)[col_labels]
        block_sums = row_onehot.T @ X @ col_onehot
        block_sizes = np.outer(row_onehot.sum(axis=0), col_onehot.sum(axis=0))

        # _relocate_empty_groups keeps every group non-empty; an empty block
        # here would put NaN in G and silently corrupt the argmin steps.
        if np.any(block_sizes == 0):
            raise RuntimeError("Empty group in the prototype update.")

        return block_sums / block_sizes

    @staticmethod
    def _objective(X, G, row_labels, col_labels):
        """W(G, U, V): squared distance of each x_ij to its block prototype."""
        return float(np.sum((X - G[np.ix_(row_labels, col_labels)]) ** 2))
