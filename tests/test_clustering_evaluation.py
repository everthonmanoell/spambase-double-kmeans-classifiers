import json
import pickle
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler

from src.analysis import clustering_evaluation, silhouette
from src.clustering.double_kmeans import DoubleKMeans


def load_history(results):
    with (results / "q1_full_history.pkl").open("rb") as stream:
        return pickle.load(stream)


def test_evaluation_persists_exact_partition_and_metrics(experiment, tmp_path, monkeypatch):
    data, results, plots = experiment
    report = tmp_path / "report" / "Q1_ANALYSIS.md"
    history = load_history(results)
    original_history = (results / "q1_full_history.pkl").read_bytes()

    def forbid_refit(*args, **kwargs):
        pytest.fail("Evaluation must not refit clustering or preprocessing.")

    monkeypatch.setattr(DoubleKMeans, "fit", forbid_refit)
    monkeypatch.setattr(StandardScaler, "fit", forbid_refit)
    metrics = clustering_evaluation.evaluate_clustering(data, results, plots, report)
    best = min((r for r in history["runs"]
                if (r["K"], r["H"]) == (metrics["K"], metrics["H"])),
               key=lambda r: r["objective"])
    y = pd.read_csv(data, header=None).iloc[:, -1].to_numpy()
    assert metrics["ari"] == adjusted_rand_score(y, best["labels"])
    assert metrics["best_run_id"] == best["run"]
    assert metrics["best_objective"] == best["objective"]
    assert metrics["total_runs"] == 18
    assert metrics == json.loads((results / "q1_evaluation.json").read_text())

    table = pd.read_csv(results / "q1_contingency.csv", index_col=0)
    np.testing.assert_array_equal(table.to_numpy(), contingency_matrix(y, best["labels"]))
    assert table.to_numpy().sum() == len(y)
    sizes = pd.read_csv(results / "q1_cluster_sizes.csv")
    np.testing.assert_array_equal(sizes.n_objects, np.bincount(best["labels"]))
    assert sizes.fraction.sum() == pytest.approx(1)
    column_sizes = pd.read_csv(results / "q1_variable_cluster_sizes.csv")
    assert column_sizes.n_features.sum() == 6
    G = pd.read_csv(results / "q1_prototypes.csv", index_col=0)
    np.testing.assert_allclose(G, best["prototypes"])
    values = pd.read_csv(results / "q1_objective_history.csv")
    np.testing.assert_array_equal(values.iteration, np.arange(best["n_iter"] + 1))
    np.testing.assert_allclose(values.objective, best["objective_history"])
    labels = pd.read_csv(results / "q1_object_labels.csv")
    np.testing.assert_array_equal(labels.row_index, np.arange(len(y)))
    np.testing.assert_array_equal(labels.cluster, best["labels"])
    np.testing.assert_array_equal(labels.original_class, y)
    columns = pd.read_csv(results / "q1_variable_labels.csv")
    np.testing.assert_array_equal(columns.cluster, best["column_labels"])
    np.testing.assert_array_equal(columns.feature_index, np.arange(6))
    with np.load(results / "q1_selected_partition.npz", allow_pickle=False) as saved:
        for key, value in best.items():
            np.testing.assert_array_equal(saved[key], value)
        np.testing.assert_array_equal(saved["scaler_mean"], history["scaler_mean"])
        np.testing.assert_array_equal(saved["scaler_scale"], history["scaler_scale"])
        assert saved["dataset_sha256"].item() == metrics["dataset_sha256"]
    assert (results / "q1_full_history.pkl").read_bytes() == original_history
    assert f"{metrics['ari']:.10f}" in (results / "ari_score.txt").read_text()

    for name in ("confusion_matrix_q1.png", "prototypes_heatmap.png",
                 "objective_convergence.png", "silhouette_heatmap.png"):
        assert (plots / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    text = report.read_text()
    assert "experimento reduzido" in text
    assert "IDs dos clusters sao arbitrarios" in text
    assert f"{metrics['ari']:.10f}" in text
    for target in re.findall(r"\]\(([^)]+)\)", text):
        assert (report.parent / target).exists(), target


def test_rectangular_contingency_for_three_clusters(experiment, tmp_path, monkeypatch):
    data, results, plots = experiment

    def prefer_three_clusters(X, labels, **kwargs):
        return 0.9 if len(np.unique(labels)) == 3 else 0.1

    monkeypatch.setattr(silhouette, "silhouette_score", prefer_three_clusters)
    metrics = clustering_evaluation.evaluate_clustering(
        data, results, plots, tmp_path / "Q1_ANALYSIS.md"
    )
    assert (metrics["K"], metrics["H"]) == (3, 1)
    table = pd.read_csv(results / "q1_contingency.csv", index_col=0)
    assert table.shape == (2, 3)
    assert table.to_numpy().sum() == 24
    assert metrics["ari"] == pytest.approx(adjusted_rand_score(
        pd.read_csv(data, header=None).iloc[:, -1],
        pd.read_csv(results / "q1_object_labels.csv").cluster,
    ))


@pytest.mark.parametrize("field", ["prototypes", "column_labels", "objective_history"])
def test_invalid_winner_is_rejected(experiment, tmp_path, field):
    data, results, plots = experiment
    history = load_history(results)
    for run in history["runs"]:
        if field == "prototypes":
            run[field] = run[field] + 100
        elif field == "column_labels":
            run[field] = np.full(6, -1, dtype=int)
        else:
            run[field] = np.array([np.nan])
    with (results / "q1_full_history.pkl").open("wb") as stream:
        pickle.dump(history, stream)
    report = tmp_path / "report.md"
    with pytest.raises(ValueError, match="Winning objective|Invalid winning"):
        clustering_evaluation.evaluate_clustering(data, results, plots, report)
    assert not (results / "q1_evaluation.json").exists()
    assert not report.exists()


@pytest.mark.parametrize("ari, expected", [
    (-0.005, "negativa"), (0.0, "ao acaso"),
    (0.5, "positiva"), (1.0, "coincidem"),
])
def test_ari_comment(ari, expected):
    assert expected in clustering_evaluation._ari_comment(ari)


def test_cli(experiment, tmp_path, capsys):
    data, results, plots = experiment
    report = tmp_path / "report.md"
    clustering_evaluation.main([
        "--data", str(data), "--results-dir", str(results),
        "--plots-dir", str(plots), "--report-path", str(report),
    ])
    output = capsys.readouterr().out
    assert "ARI=" in output
    assert str(report) in output
    assert report.exists()
