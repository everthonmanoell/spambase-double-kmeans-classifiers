"""Compare the nine Q1 winners by full-data Euclidean silhouette."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import pickle

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import silhouette_score

from src.experiments.question_1 import Q1_GRID, _atomic_output
from src.utils.dataloader import load_spambase_original


LOGGER = logging.getLogger(__name__)


def _load_analysis_inputs(filepath, history_path):
    # Pickle must only be loaded from a trusted local experiment.
    with history_path.open("rb") as stream:
        history = pickle.load(stream)
    metadata = history["metadata"]
    if (metadata["format_version"] != 1
            or metadata["preprocessing"] != "StandardScaler"
            or metadata["grid"] != Q1_GRID):
        raise ValueError("Unsupported Q1 checkpoint format, preprocessing or grid.")
    if hashlib.sha256(filepath.read_bytes()).hexdigest() != metadata["dataset_sha256"]:
        raise ValueError("Dataset differs from the Q1 checkpoint (SHA-256 mismatch).")
    for name, version in (
        ("numpy_version", np.__version__),
        ("pandas_version", pd.__version__),
        ("sklearn_version", sklearn.__version__),
    ):
        if metadata[name] != version:
            raise ValueError(f"{name} differs from the Q1 checkpoint.")

    n_runs = metadata["n_runs"]
    if not isinstance(n_runs, int) or isinstance(n_runs, bool) or n_runs < 1:
        raise ValueError("Checkpoint n_runs must be a positive integer.")
    expected = [(K, H, run) for K, H in Q1_GRID for run in range(n_runs)]
    actual = [(run["K"], run["H"], run["run"]) for run in history["runs"]]
    if actual != expected:
        raise ValueError("Analysis requires a complete Q1 history for all nine pairs.")
    if not all(np.isfinite(run["objective"]) for run in history["runs"]):
        raise ValueError("Q1 history contains non-finite objectives.")

    X, _ = load_spambase_original(filepath)
    mean = np.asarray(history["scaler_mean"])
    scale = np.asarray(history["scaler_scale"])
    if (X.shape != metadata["shape"]
            or mean.shape != (X.shape[1],) or scale.shape != mean.shape
            or not np.isfinite(X).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Invalid feature matrix or saved standardization parameters.")
    # StandardScaler.transform performs these two operations, in this order.
    X = X.astype(float, copy=True)
    X -= mean
    X /= scale
    return X, history


def _heatmap(scores):
    matrix = scores.pivot(index="K", columns="H", values="silhouette").reindex(
        index=[2, 3, 4], columns=[1, 2, 3, 4]
    )
    figure = Figure(figsize=(7, 4.5), layout="constrained")
    axes = figure.subplots()
    image = axes.imshow(
        np.ma.masked_invalid(matrix.to_numpy()), cmap="viridis", vmin=-1, vmax=1
    )
    axes.set_xticks(range(4), labels=[1, 2, 3, 4])
    axes.set_yticks(range(3), labels=[2, 3, 4])
    axes.set_xlabel("H (variable clusters)")
    axes.set_ylabel("K (object clusters)")
    winner = scores.loc[scores["silhouette"].idxmax()]
    axes.set_title(f"Q1 silhouette: K*={int(winner.K)}, H*={int(winner.H)}")
    for row in scores.itertuples(index=False):
        axes.text(row.H - 1, row.K - 2, f"{row.silhouette:.4f}",
                  ha="center", va="center", color="white")
    figure.colorbar(image, ax=axes, label="Euclidean silhouette")
    return figure


def analyze_silhouette(filepath="dataset/spambase.data", results_dir="results",
                       plots_dir="plots"):
    """Score each minimum-objective Q1 winner, without refitting clustering.

    The checkpoint is authoritative. Original labels are not used; features
    retain their original row order and use the saved Q1 standardization.
    Exact silhouette ties are resolved by ascending K, then H.

    Returns:
        A nine-row DataFrame with the selected run and silhouette per pair.
    """
    results_dir, plots_dir = Path(results_dir), Path(plots_dir)
    X, history = _load_analysis_inputs(
        Path(filepath), results_dir / "q1_full_history.pkl"
    )
    if history["metadata"]["n_runs"] != 100:
        LOGGER.warning(
            "Reduced experiment: %d starts per pair, not the full 100-start Q1.",
            history["metadata"]["n_runs"],
        )
    rows = []
    for K, H in Q1_GRID:
        runs = [run for run in history["runs"] if (run["K"], run["H"]) == (K, H)]
        best = min(runs, key=lambda run: run["objective"])
        labels = np.asarray(best["labels"])
        if (labels.shape != (X.shape[0],)
                or not np.issubdtype(labels.dtype, np.integer)
                or not np.array_equal(np.unique(labels), np.arange(K))):
            raise ValueError(f"Invalid winner labels for K={K}, H={H}.")
        if not best["converged"]:
            LOGGER.warning("Winner K=%d H=%d did not converge.", K, H)
        score = float(silhouette_score(X, labels, metric="euclidean"))
        if not np.isfinite(score):
            raise RuntimeError(f"Non-finite silhouette for K={K}, H={H}.")
        rows.append({
            "K": K,
            "H": H,
            "n_runs": len(runs),
            "best_run_id": best["run"],
            "best_objective": best["objective"],
            "best_converged": bool(best["converged"]),
            "silhouette": score,
        })
        LOGGER.info("K=%d H=%d seed=%d silhouette=%.8f", K, H, best["run"], score)

    scores = pd.DataFrame(rows)
    selection = scores.loc[scores["silhouette"].idxmax()].to_dict()
    selection.update({
        "metric": "euclidean",
        "preprocessing": "StandardScaler",
        "dataset_sha256": history["metadata"]["dataset_sha256"],
    })
    plots_dir.mkdir(parents=True, exist_ok=True)
    figure = _heatmap(scores)
    try:
        with _atomic_output(plots_dir / "silhouette_heatmap.png") as stream:
            figure.savefig(stream, format="png", dpi=160)
    finally:
        figure.clear()
    with _atomic_output(results_dir / "silhouette_scores.csv") as stream:
        stream.write(scores.to_csv(index=False).encode("utf-8"))
    with _atomic_output(results_dir / "q1_selection.json") as stream:
        stream.write(json.dumps(selection, indent=2, allow_nan=False).encode("utf-8"))
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/spambase.data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--plots-dir", default="plots")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    scores = analyze_silhouette(args.data, args.results_dir, args.plots_dir)
    print(scores.to_string(index=False))
    winner = scores.loc[scores["silhouette"].idxmax()]
    print(f"\nK* = {int(winner.K)}, H* = {int(winner.H)}, "
          f"silhouette = {winner.silhouette:.8f}")


if __name__ == "__main__":
    main()
