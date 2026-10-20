import matplotlib
import numpy as np
import pandas as pd
import pytest

from src.clustering.double_kmeans import DoubleKMeans
from src.experiments.q1_double_kmeans import (load_feature_names,
                                              plot_objective_history,
                                              run_pair, select_best_pair)

# The tests only save figures, so no window backend is needed.
matplotlib.use("Agg")

N_SAMPLES, N_FEATURES = 40, 8
SEEDS = [0, 1, 2, 3, 4]

# Same layout as dataset/spambase.names: comments start with "|", the class
# line ends in "." and only the attribute lines end in "continuous.".
NAMES_FILE = """\
| 48 continuous real [0,100] attributes of type word_freq_WORD
|
1, 0.    | spam, non-spam classes

word_freq_make:         continuous.
char_freq_(:            continuous.
capital_run_length_total: continuous.
"""


@pytest.fixture
def random_data():
    """Standard normal data with no group structure."""
    return np.random.RandomState(0).normal(size=(N_SAMPLES, N_FEATURES))


def make_summary(rows):
    """Summary table with the columns built by main(), from (K, H, Sil)."""
    return pd.DataFrame([{"K": K, "H": H, "objective": 1.0, "silhouette": sil}
                         for K, H, sil in rows])


# Choice of (K*, H*).

def test_select_best_pair_picks_largest_silhouette():
    summary = make_summary([(2, 1, 0.30), (3, 2, 0.45), (4, 1, 0.20)])
    assert select_best_pair(summary) == (3, 2)


def test_select_best_pair_ties_go_to_smallest_k_then_h():
    # (3, 2) and (3, 3) tie as when pairs with the same K reach the same
    # object partition. (4, 1) also ties and has the smallest H, so the
    # answer differs if H were compared before K. (2, 1) has the smallest
    # K but a lower silhouette, so it must not win.
    summary = make_summary([(2, 1, 0.30), (3, 3, 0.45), (4, 1, 0.45),
                            (3, 2, 0.45)])
    for order in (summary, summary.iloc[::-1],
                  summary.sample(frac=1, random_state=0)):
        assert select_best_pair(order) == (3, 2)


def test_select_best_pair_returns_python_ints():
    # (K*, H*) goes to json.dump in main(), which rejects numpy.int64.
    K, H = select_best_pair(make_summary([(2, 1, 0.30), (3, 1, 0.40)]))
    assert type(K) is int and type(H) is int


# Runs of one (K, H) pair.

def test_run_pair_records_every_run(random_data):
    _, runs = run_pair(random_data, K=3, H=2, seeds=SEEDS)
    assert [run["seed"] for run in runs] == SEEDS
    assert all(run["K"] == 3 and run["H"] == 2 for run in runs)
    assert set(runs[0]) == {"K", "H", "seed", "objective", "n_iter",
                            "converged", "n_relocations"}


def test_run_pair_keeps_the_run_with_smallest_objective(random_data):
    best, runs = run_pair(random_data, K=3, H=2, seeds=SEEDS)
    objectives = [run["objective"] for run in runs]
    # Different local minima, so the choice is not trivial.
    assert len(set(objectives)) > 1
    smallest = runs[int(np.argmin(objectives))]
    assert best.objective_ == smallest["objective"]
    assert best.random_state == smallest["seed"]


# Variable names.

def test_load_feature_names_reads_only_attribute_lines(tmp_path):
    path = tmp_path / "spambase.names"
    path.write_text(NAMES_FILE)
    assert load_feature_names(path) == ["word_freq_make", "char_freq_(",
                                        "capital_run_length_total"]


# Plot of W over the iterations.

def test_plot_objective_history_saves_non_converged_run(random_data, tmp_path):
    model = DoubleKMeans(n_row_clusters=3, n_col_clusters=2, max_iter=1,
                         random_state=0).fit(random_data)
    assert not model.converged_
    path = tmp_path / "objective_vs_iteration.png"
    plot_objective_history(model, path)
    assert path.stat().st_size > 0
