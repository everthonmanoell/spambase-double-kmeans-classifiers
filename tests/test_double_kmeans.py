import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from src.clustering.double_kmeans import DoubleKMeans

N_SAMPLES, N_FEATURES = 60, 12
BLOCK_NOISE = 0.5


@pytest.fixture
def random_data():
    """Standard normal data with no group structure."""
    return np.random.RandomState(0).normal(size=(N_SAMPLES, N_FEATURES))


@pytest.fixture
def block_data():
    """Data with 3 object groups and 2 variable groups, and the true labels."""
    rng = np.random.RandomState(1)
    true_rows = rng.permutation(np.repeat(np.arange(3), N_SAMPLES // 3))
    true_cols = rng.permutation(np.repeat(np.arange(2), N_FEATURES // 2))
    block_means = np.array([[0.0, 5.0], [5.0, 0.0], [10.0, 10.0]])
    X = (block_means[np.ix_(true_rows, true_cols)]
         + BLOCK_NOISE * rng.normal(size=(N_SAMPLES, N_FEATURES)))
    return X, true_rows, true_cols


def reference_prototypes(X, rows, cols, K, H):
    """Block means computed with plain loops."""
    G = np.empty((K, H))
    for k in range(K):
        for h in range(H):
            G[k, h] = X[np.ix_(rows == k, cols == h)].mean()
    return G


def reference_objective(X, G, rows, cols):
    """W(G, U, V) computed with plain loops."""
    return sum((X[i, j] - G[rows[i], cols[j]]) ** 2
               for i in range(X.shape[0]) for j in range(X.shape[1]))


# Parameter validation.

@pytest.mark.parametrize("name, value", [
    ("n_row_clusters", 0), ("n_row_clusters", N_SAMPLES + 1),
    ("n_col_clusters", 0), ("n_col_clusters", N_FEATURES + 1),
    ("max_iter", 0), ("max_iter", -1), ("max_iter", 1.5),
])
def test_invalid_parameters_raise(random_data, name, value):
    with pytest.raises(ValueError, match=name):
        DoubleKMeans(**{name: value}).fit(random_data)


# Fitted attributes.

def test_fitted_attributes(random_data):
    K, H = 3, 2
    model = DoubleKMeans(K, H, random_state=0).fit(random_data)

    assert model.row_labels_.shape == (N_SAMPLES,)
    assert model.column_labels_.shape == (N_FEATURES,)
    assert model.prototypes_.shape == (K, H)
    np.testing.assert_array_equal(model.labels_, model.row_labels_)
    np.testing.assert_array_equal(np.unique(model.row_labels_), np.arange(K))
    np.testing.assert_array_equal(np.unique(model.column_labels_), np.arange(H))
    assert 1 <= model.n_iter_ <= model.max_iter
    assert len(model.objective_history_) == model.n_iter_ + 1


def test_fit_predict_returns_row_labels(random_data):
    model = DoubleKMeans(3, 2, random_state=0)
    labels = model.fit_predict(random_data)
    np.testing.assert_array_equal(labels, model.row_labels_)


# Consistency with the definitions of G and W.

@pytest.mark.parametrize("max_iter", [1, 100])
def test_prototypes_and_objective_match_reference(random_data, max_iter):
    K, H = 3, 2
    model = DoubleKMeans(K, H, max_iter=max_iter, random_state=0).fit(random_data)
    rows, cols = model.row_labels_, model.column_labels_

    G = reference_prototypes(random_data, rows, cols, K, H)
    np.testing.assert_allclose(model.prototypes_, G)
    assert model.objective_ == pytest.approx(
        reference_objective(random_data, G, rows, cols))


@pytest.mark.parametrize("max_iter, converged", [(1, False), (100, True)])
def test_history_ends_at_objective_and_never_increases(random_data, max_iter,
                                                        converged):
    model = DoubleKMeans(3, 2, max_iter=max_iter, random_state=0).fit(random_data)
    history = model.objective_history_

    assert model.converged_ is converged
    assert history[-1] == model.objective_
    assert np.all(np.diff(history) <= 1e-12 * history[0])


# Stopping rule of the slides.

def test_converged_partition_is_a_fixed_point(random_data):
    """After convergence, steps 2 and 3 would not move any object or variable."""
    X, K, H = random_data, 3, 2
    model = DoubleKMeans(K, H, random_state=0).fit(X)
    rows, cols, G = model.row_labels_, model.column_labels_, model.prototypes_
    assert model.converged_

    row_cost = np.array([[np.sum((X[i] - G[k, cols]) ** 2) for k in range(K)]
                         for i in range(N_SAMPLES)])
    col_cost = np.array([[np.sum((X[:, j] - G[rows, h]) ** 2) for h in range(H)]
                         for j in range(N_FEATURES)])

    # Margin of _assign (1e-9) plus room for rounding in these costs.
    for cost, labels in ((row_cost, rows), (col_cost, cols)):
        gain = cost[np.arange(len(labels)), labels] - cost.min(axis=1)
        assert np.all(gain <= 1e-8)


# Internal steps.

@pytest.mark.parametrize("gain, moves", [(0.0, False), (1e-12, False),
                                         (1e-6, True)])
def test_assign_moves_only_beyond_the_margin(gain, moves):
    cost = np.array([[1.0, 1.0 - gain]])
    new_labels = DoubleKMeans._assign(cost, np.array([0]))
    assert new_labels[0] == (1 if moves else 0)


def test_relocation_fills_empty_object_group(random_data):
    X = random_data
    rows = np.arange(N_SAMPLES) % 2  # group 2 of 3 is empty
    cols = np.arange(N_FEATURES) % 2
    G = np.vstack([reference_prototypes(X, rows, cols, 2, 2), np.zeros(2)])

    new_rows, new_G, n_moved = DoubleKMeans._relocate_empty_groups(
        X, G, rows, cols)

    assert n_moved == 1
    assert np.all(np.bincount(new_rows, minlength=3) >= 1)
    assert (reference_objective(X, new_G, new_rows, cols)
            <= reference_objective(X, G, rows, cols))


def test_relocation_fills_empty_variable_group(random_data):
    X = random_data
    rows = np.arange(N_SAMPLES) % 2
    cols = np.arange(N_FEATURES) % 2  # group 2 of 3 is empty
    G = np.hstack([reference_prototypes(X, rows, cols, 2, 2), np.zeros((2, 1))])

    new_cols, new_G_t, n_moved = DoubleKMeans._relocate_empty_groups(
        X.T, G.T, cols, rows)

    assert n_moved == 1
    assert np.all(np.bincount(new_cols, minlength=3) >= 1)
    assert (reference_objective(X, new_G_t.T, rows, new_cols)
            <= reference_objective(X, G, rows, cols))


def test_relocation_never_empties_the_donor():
    """The highest-cost object is alone in its group, so it must stay there."""
    X = np.zeros((6, 2))
    X[5] = [100.0, -100.0]
    rows = np.array([0, 0, 0, 0, 0, 1])
    cols = np.array([0, 0])
    G = np.vstack([reference_prototypes(X, rows, cols, 2, 1), np.zeros(1)])

    new_rows, _, _ = DoubleKMeans._relocate_empty_groups(X, G, rows, cols)

    assert new_rows[5] == 1
    assert np.all(np.bincount(new_rows, minlength=3) >= 1)


def test_update_prototypes_rejects_empty_group(random_data):
    rows = np.zeros(N_SAMPLES, dtype=int)  # group 1 of 2 is empty
    cols = np.arange(N_FEATURES) % 2
    with pytest.raises(RuntimeError, match="Empty group"):
        DoubleKMeans._update_prototypes(random_data, rows, cols, 2, 2)


# Behaviour on data.

def test_same_random_state_gives_same_result(random_data):
    a = DoubleKMeans(3, 2, random_state=42).fit(random_data)
    b = DoubleKMeans(3, 2, random_state=42).fit(random_data)
    np.testing.assert_array_equal(a.row_labels_, b.row_labels_)
    np.testing.assert_array_equal(a.column_labels_, b.column_labels_)
    np.testing.assert_array_equal(a.objective_history_, b.objective_history_)


def test_recovers_planted_blocks(block_data):
    """On well-separated blocks, the true partitions are recovered."""
    X, true_rows, true_cols = block_data
    # A single random start often stops at a local minimum, so keep the run
    # with the lowest W out of 20, as the Q1 script does.
    models = [DoubleKMeans(3, 2, random_state=seed).fit(X) for seed in range(20)]
    best = min(models, key=lambda m: m.objective_)

    assert adjusted_rand_score(true_rows, best.row_labels_) == 1.0
    assert adjusted_rand_score(true_cols, best.column_labels_) == 1.0


# scikit-learn estimator checks.

@parametrize_with_checks([DoubleKMeans(random_state=0)])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
