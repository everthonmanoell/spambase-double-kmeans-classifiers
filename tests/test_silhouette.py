import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.analysis import silhouette
from src.clustering.double_kmeans import DoubleKMeans
from src.experiments.question_1 import Q1_GRID
from src.utils.dataloader import load_spambase_original


def read_history(results):
    with (results / "q1_full_history.pkl").open("rb") as stream:
        return pickle.load(stream)


def write_history(results, history):
    with (results / "q1_full_history.pkl").open("wb") as stream:
        pickle.dump(history, stream)


def test_full_scores_and_selection_match_reference(experiment, monkeypatch, caplog):
    data, results, plots = experiment
    history = read_history(results)
    original_checkpoint = (results / "q1_full_history.pkl").read_bytes()

    def forbid_refit(*args, **kwargs):
        pytest.fail("Silhouette analysis must not fit clustering or scaler.")

    monkeypatch.setattr(DoubleKMeans, "fit", forbid_refit)
    monkeypatch.setattr(StandardScaler, "fit", forbid_refit)
    X, _ = load_spambase_original(data)
    X_scaled = (X - history["scaler_mean"]) / history["scaler_scale"]
    scores = silhouette.analyze_silhouette(data, results, plots)
    assert list(zip(scores.K, scores.H)) == list(Q1_GRID)
    assert scores.n_runs.eq(2).all()
    for row in scores.itertuples(index=False):
        runs = [r for r in history["runs"] if (r["K"], r["H"]) == (row.K, row.H)]
        best = min(runs, key=lambda r: r["objective"])
        assert row.best_run_id == best["run"]
        assert row.best_objective == best["objective"]
        assert row.silhouette == pytest.approx(
            silhouette_score(X_scaled, best["labels"], metric="euclidean"), abs=1e-14
        )
    pd.testing.assert_frame_equal(
        scores, pd.read_csv(results / "silhouette_scores.csv"), check_exact=False
    )
    selected = json.loads((results / "q1_selection.json").read_text())
    winner = scores.loc[scores.silhouette.idxmax()]
    assert (selected["K"], selected["H"]) == (winner.K, winner.H)
    assert selected["best_run_id"] == winner.best_run_id
    assert selected["silhouette"] == winner.silhouette
    assert selected["dataset_sha256"] == history["metadata"]["dataset_sha256"]
    assert (plots / "silhouette_heatmap.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (results / "q1_full_history.pkl").read_bytes() == original_checkpoint
    assert "Reduced experiment" in caplog.text


def test_saved_scaling_matches_original_transform(experiment):
    data, results, _ = experiment
    X, _ = load_spambase_original(data)
    expected = StandardScaler().fit_transform(X)
    actual, _ = silhouette._load_analysis_inputs(data, results / "q1_full_history.pkl")
    np.testing.assert_array_equal(actual, expected)


def test_ties_use_lowest_k_then_h_and_first_run(experiment, monkeypatch):
    data, results, plots = experiment
    history = read_history(results)
    for run in history["runs"]:
        run["objective"] = 1.0
    write_history(results, history)
    monkeypatch.setattr(silhouette, "silhouette_score", lambda *args, **kwargs: 0.25)
    scores = silhouette.analyze_silhouette(data, results, plots)
    assert scores.best_run_id.eq(0).all()
    selected = json.loads((results / "q1_selection.json").read_text())
    assert (selected["K"], selected["H"], selected["best_run_id"]) == (2, 1, 0)


def test_heatmap_masks_only_invalid_pairs(experiment):
    scores = pd.DataFrame([
        {"K": K, "H": H, "silhouette": K / 10 + H / 100} for K, H in Q1_GRID
    ])
    figure = silhouette._heatmap(scores)
    try:
        image = figure.axes[0].images[0].get_array()
        assert image.shape == (3, 4)
        np.testing.assert_array_equal(np.ma.getmaskarray(image), [
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ])
        assert len(figure.axes[0].texts) == 9
    finally:
        figure.clear()


@pytest.mark.parametrize("corruption, message", [
    ("incomplete", "complete Q1 history"),
    ("labels", "Invalid winner labels"),
    ("scale", "standardization"),
    ("objective", "non-finite objectives"),
    ("version", "pandas_version"),
    ("format", "Unsupported Q1 checkpoint"),
])
def test_invalid_history_fails_before_output(experiment, corruption, message):
    data, results, plots = experiment
    history = read_history(results)
    if corruption == "incomplete":
        history["runs"].pop()
    elif corruption == "labels":
        for run in history["runs"]:
            run["labels"] = np.zeros(24, dtype=int)
    elif corruption == "scale":
        history["scaler_scale"][0] = 0
    elif corruption == "objective":
        history["runs"][0]["objective"] = np.nan
    elif corruption == "version":
        history["metadata"]["pandas_version"] = "different"
    else:
        history["metadata"]["format_version"] = 999
    write_history(results, history)
    with pytest.raises(ValueError, match=message):
        silhouette.analyze_silhouette(data, results, plots)
    assert not (results / "silhouette_scores.csv").exists()
    assert not plots.exists()


def test_different_dataset_is_rejected(experiment):
    data, results, plots = experiment
    with data.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        silhouette.analyze_silhouette(data, results, plots)


def test_nonconverged_winner_warns(experiment, caplog):
    data, results, plots = experiment
    history = read_history(results)
    for run in history["runs"]:
        run["converged"] = False
    write_history(results, history)
    scores = silhouette.analyze_silhouette(data, results, plots)
    assert not scores.best_converged.any()
    assert "did not converge" in caplog.text


def test_cli(experiment, capsys):
    data, results, plots = experiment
    silhouette.main(["--data", str(data), "--results-dir", str(results),
                     "--plots-dir", str(plots)])
    output = capsys.readouterr().out
    assert "K* =" in output
    assert "H* =" in output
    assert "silhouette" in output
