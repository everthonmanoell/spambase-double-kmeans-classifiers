import numpy as np
import pytest

from src.experiments.question_1 import run_question_1


@pytest.fixture
def experiment(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 6)) * np.arange(1, 7) + np.arange(6) * 10
    data = tmp_path / "spambase.data"
    np.savetxt(data, np.column_stack((X, np.arange(24) % 2)), delimiter=",")
    results = tmp_path / "results"
    run_question_1(data, results, n_runs=2)
    return data, results, tmp_path / "plots"