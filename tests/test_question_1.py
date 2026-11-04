import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.experiments import question_1


@pytest.fixture
def spambase_file(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 6)) * np.arange(1, 7) + np.arange(6) * 10
    y = np.arange(20) % 2
    path = tmp_path / "spambase.data"
    np.savetxt(path, np.column_stack((X, y)), delimiter=",")
    return path


def load_history(output_dir):
    with (output_dir / "q1_full_history.pkl").open("rb") as stream:
        return pickle.load(stream)


def test_real_fits_persist_grid_and_best_runs(spambase_file, tmp_path):
    output_dir = tmp_path / "results"
    summary = question_1.run_question_1(spambase_file, output_dir, n_runs=3)
    history = load_history(output_dir)
    X = np.loadtxt(spambase_file, delimiter=",")[:, :-1]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    assert question_1.Q1_GRID == (
        (2, 1), (2, 2), (3, 1), (3, 2), (3, 3),
        (4, 1), (4, 2), (4, 3), (4, 4),
    )
    assert len(history["runs"]) == 27
    assert list(zip(summary.K, summary.H)) == list(question_1.Q1_GRID)
    np.testing.assert_allclose(history["scaler_mean"], scaler.mean_, rtol=1e-14)
    np.testing.assert_allclose(history["scaler_scale"], scaler.scale_, rtol=1e-14)
    pd.testing.assert_frame_equal(
        summary, pd.read_csv(output_dir / "q1_executions_summary.csv")
    )

    for row in summary.itertuples(index=False):
        runs = [r for r in history["runs"] if (r["K"], r["H"]) == (row.K, row.H)]
        best = min(runs, key=lambda r: r["objective"])
        assert [r["run"] for r in runs] == [0, 1, 2]
        assert row.n_runs == 3
        assert row.best_run_id == best["run"]
        assert row.best_objective == pytest.approx(best["objective"])
        assert row.n_iterations == best["n_iter"]
        assert row.best_converged == best["converged"]
        assert row.convergence_rate == pytest.approx(
            sum(r["converged"] for r in runs) / 3
        )
        with np.load(output_dir / f"q1_best_dkm_K{row.K}_H{row.H}.npz",
                     allow_pickle=False) as artifact:
            assert set(artifact.files) == set(best)
            for key, value in best.items():
                np.testing.assert_array_equal(artifact[key], value)

    for run in history["runs"]:
        assert run["labels"].shape == (20,)
        assert run["column_labels"].shape == (6,)
        assert run["prototypes"].shape == (run["K"], run["H"])
        np.testing.assert_array_equal(np.unique(run["labels"]), np.arange(run["K"]))
        np.testing.assert_array_equal(
            np.unique(run["column_labels"]), np.arange(run["H"])
        )
        reconstructed = run["prototypes"][np.ix_(run["labels"], run["column_labels"])]
        assert run["objective"] == pytest.approx(np.sum((X_scaled - reconstructed) ** 2))
        assert run["objective_history"][-1] == run["objective"]


def test_defaults_produce_900_runs(spambase_file, tmp_path):
    output_dir = tmp_path / "full"
    summary = question_1.run_question_1(spambase_file, output_dir)
    assert len(load_history(output_dir)["runs"]) == 900
    assert summary.n_runs.tolist() == [100] * 9


def test_resume_reuses_completed_pairs_and_repairs_outputs(spambase_file, tmp_path,
                                                         monkeypatch):
    output_dir = tmp_path / "resumed"
    original_fit = question_1.DoubleKMeans.fit
    calls = []

    def interrupt_fit(model, X, y=None):
        calls.append((model.n_row_clusters, model.n_col_clusters, model.random_state))
        if len(calls) == 4:
            raise RuntimeError("interrupted")
        return original_fit(model, X, y)

    monkeypatch.setattr(question_1.DoubleKMeans, "fit", interrupt_fit)
    with pytest.raises(RuntimeError, match="interrupted"):
        question_1.run_question_1(spambase_file, output_dir, n_runs=2)
    assert len(load_history(output_dir)["runs"]) == 2
    (output_dir / "q1_executions_summary.csv").unlink()
    (output_dir / "q1_best_dkm_K2_H1.npz").unlink()

    calls.clear()

    def tracking_fit(model, X, y=None):
        calls.append((model.n_row_clusters, model.n_col_clusters, model.random_state))
        return original_fit(model, X, y)

    monkeypatch.setattr(question_1.DoubleKMeans, "fit", tracking_fit)
    resumed = question_1.run_question_1(spambase_file, output_dir, n_runs=2, resume=True)
    assert len(calls) == 16
    assert all((K, H) != (2, 1) for K, H, _ in calls)
    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted = question_1.run_question_1(spambase_file, uninterrupted_dir, n_runs=2)
    pd.testing.assert_frame_equal(resumed, uninterrupted)
    for actual, expected in zip(load_history(output_dir)["runs"],
                                load_history(uninterrupted_dir)["runs"]):
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    calls.clear()
    question_1.run_question_1(spambase_file, output_dir, n_runs=2, resume=True)
    assert calls == []


def test_resume_before_first_completed_pair(spambase_file, tmp_path, monkeypatch):
    output_dir = tmp_path / "results"
    original_fit = question_1.DoubleKMeans.fit

    def interrupt_fit(model, X, y=None):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(question_1.DoubleKMeans, "fit", interrupt_fit)
    with pytest.raises(RuntimeError, match="interrupted"):
        question_1.run_question_1(spambase_file, output_dir, n_runs=1)
    assert load_history(output_dir)["runs"] == []
    monkeypatch.setattr(question_1.DoubleKMeans, "fit", original_fit)
    summary = question_1.run_question_1(spambase_file, output_dir, n_runs=1, resume=True)
    assert len(summary) == 9


@pytest.mark.parametrize("change", ["n_runs", "max_iter", "data"])
def test_resume_rejects_incompatible_input(spambase_file, tmp_path, change):
    output_dir = tmp_path / "results"
    question_1.run_question_1(spambase_file, output_dir, n_runs=1)
    kwargs = {"n_runs": 1, "resume": True}
    if change == "data":
        with spambase_file.open("a") as stream:
            stream.write("1,2,3,4,5,6,0\n")
    else:
        kwargs[change] = 2
    with pytest.raises(ValueError, match="configuration differs"):
        question_1.run_question_1(spambase_file, output_dir, **kwargs)


def test_checkpoint_is_not_silently_overwritten(spambase_file, tmp_path):
    output_dir = tmp_path / "results"
    with pytest.raises(FileNotFoundError, match="No Q1 checkpoint"):
        question_1.run_question_1(spambase_file, output_dir, n_runs=1, resume=True)
    question_1.run_question_1(spambase_file, output_dir, n_runs=1)
    original = (output_dir / "q1_full_history.pkl").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        question_1.run_question_1(spambase_file, output_dir, n_runs=1)
    assert (output_dir / "q1_full_history.pkl").read_bytes() == original


def test_invalid_checkpoint_prefix_raises(spambase_file, tmp_path):
    output_dir = tmp_path / "results"
    question_1.run_question_1(spambase_file, output_dir, n_runs=1)
    history = load_history(output_dir)
    history["runs"].reverse()
    with (output_dir / "q1_full_history.pkl").open("wb") as stream:
        pickle.dump(history, stream)
    with pytest.raises(ValueError, match="ordered prefix"):
        question_1.run_question_1(spambase_file, output_dir, n_runs=1, resume=True)


@pytest.mark.parametrize("shape", [(3, 6), (20, 3)])
def test_input_too_small_for_grid_raises(tmp_path, shape):
    path = tmp_path / "small.data"
    np.savetxt(path, np.zeros((shape[0], shape[1] + 1)), delimiter=",")
    with pytest.raises(ValueError, match="at least 4"):
        question_1.run_question_1(path, tmp_path / "results", n_runs=1)


def test_nonfinite_input_raises(spambase_file, tmp_path):
    data = np.loadtxt(spambase_file, delimiter=",")
    data[0, 0] = np.inf
    np.savetxt(spambase_file, data, delimiter=",")
    with pytest.raises(ValueError, match="finite features"):
        question_1.run_question_1(spambase_file, tmp_path / "results", n_runs=1)


def test_original_labels_do_not_affect_clustering(spambase_file, tmp_path):
    original_dir = tmp_path / "original"
    question_1.run_question_1(spambase_file, original_dir, n_runs=1)
    data = pd.read_csv(spambase_file, header=None)
    data.iloc[:, -1] = 999
    altered_path = tmp_path / "altered.data"
    data.to_csv(altered_path, index=False, header=False)
    altered_dir = tmp_path / "altered"
    question_1.run_question_1(altered_path, altered_dir, n_runs=1)
    for original, altered in zip(load_history(original_dir)["runs"],
                                 load_history(altered_dir)["runs"]):
        np.testing.assert_array_equal(original["labels"], altered["labels"])
        assert original["objective"] == pytest.approx(altered["objective"])


def test_atomic_output_keeps_previous_file_on_failure(tmp_path):
    path = tmp_path / "checkpoint"
    path.write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="write interrupted"):
        with question_1._atomic_output(path) as stream:
            stream.write(b"incomplete")
            raise RuntimeError("write interrupted")
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("parameter", ["n_runs", "max_iter"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_parameters_raise(spambase_file, tmp_path, parameter, value):
    with pytest.raises(ValueError, match=parameter):
        question_1.run_question_1(spambase_file, tmp_path / "results",
                                 **{parameter: value})


def test_nonconvergence_is_visible(spambase_file, tmp_path, caplog):
    summary = question_1.run_question_1(spambase_file, tmp_path / "results",
                                      n_runs=1, max_iter=1)
    assert (summary.convergence_rate < 1).any()
    assert "without convergence" in caplog.text


def test_cli_uses_real_input(spambase_file, tmp_path, capsys):
    question_1.main([
        "--data", str(spambase_file),
        "--output-dir", str(tmp_path / "cli"),
        "--n-runs", "1",
    ])
    assert "best_objective" in capsys.readouterr().out
    assert len(load_history(tmp_path / "cli")["runs"]) == 9
