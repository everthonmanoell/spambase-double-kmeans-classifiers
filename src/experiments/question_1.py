"""Run the Q1 grid on the real Spambase features."""

import argparse
from contextlib import contextmanager
import hashlib
import logging
import numbers
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from src.clustering.double_kmeans import DoubleKMeans
from src.utils.dataloader import load_spambase_original


Q1_GRID = tuple((K, H) for K in (2, 3, 4) for H in range(1, K + 1))
LOGGER = logging.getLogger(__name__)


@contextmanager
def _atomic_output(path):
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
        try:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _save_outputs(output_dir, history):
    rows = []
    for K, H in Q1_GRID:
        runs = [run for run in history["runs"] if (run["K"], run["H"]) == (K, H)]
        if not runs:
            continue
        # min keeps the first run in case of an exact objective tie.
        best = min(runs, key=lambda run: run["objective"])
        rows.append({
            "K": K,
            "H": H,
            "n_runs": len(runs),
            "best_objective": best["objective"],
            "best_run_id": best["run"],
            "n_iterations": best["n_iter"],
            "convergence_rate": sum(run["converged"] for run in runs) / len(runs),
            "best_converged": best["converged"],
        })
        with _atomic_output(output_dir / f"q1_best_dkm_K{K}_H{H}.npz") as stream:
            np.savez_compressed(stream, **best)

    summary = pd.DataFrame(rows, columns=[
        "K", "H", "n_runs", "best_objective", "best_run_id", "n_iterations",
        "convergence_rate", "best_converged",
    ])
    with _atomic_output(output_dir / "q1_executions_summary.csv") as stream:
        stream.write(summary.to_csv(index=False).encode("utf-8"))
    return summary


def run_question_1(
    filepath="dataset/spambase.data",
    output_dir="results",
    n_runs=100,
    max_iter=100,
    resume=False,
):
    """Run 100 seeded starts per (K, H), checkpointing each completed pair.

    Features are standardized with a single z-score transform on the full Q1
    dataset. Original class labels are never passed to the clustering model.
    Resume accepts only a checkpoint with the same data and configuration.
    Existing checkpoints are not overwritten unless ``resume=True``.

    Returns:
        A DataFrame with the minimum-objective run and convergence rate per pair.
    """
    for name, value in (("n_runs", n_runs), ("max_iter", max_iter)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")

    filepath, output_dir = Path(filepath), Path(output_dir)
    history_path = output_dir / "q1_full_history.pkl"
    if history_path.exists() and not resume:
        raise FileExistsError(
            f"{history_path} already exists. Use --resume or a new output directory."
        )
    if resume and not history_path.exists():
        raise FileNotFoundError(f"No Q1 checkpoint found at {history_path}.")

    X, _ = load_spambase_original(filepath)
    if X.shape[0] < 4 or X.shape[1] < 4 or not np.isfinite(X).all():
        raise ValueError("Q1 requires finite features and at least 4 rows and 4 columns.")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    metadata = {
        "format_version": 1,
        "dataset_sha256": hashlib.sha256(filepath.read_bytes()).hexdigest(),
        "shape": X.shape,
        "n_runs": int(n_runs),
        "max_iter": int(max_iter),
        "grid": Q1_GRID,
        "preprocessing": "StandardScaler",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }
    if resume:
        # Only load checkpoints produced locally by this experiment.
        with history_path.open("rb") as stream:
            history = pickle.load(stream)
        if history["metadata"] != metadata:
            raise ValueError("Checkpoint data or configuration differs from this run.")
        expected = [(K, H, run) for K, H in Q1_GRID for run in range(n_runs)]
        actual = [(run["K"], run["H"], run["run"]) for run in history["runs"]]
        if len(actual) % n_runs or actual != expected[:len(actual)]:
            raise ValueError("Checkpoint must contain an ordered prefix of complete pairs.")
        _save_outputs(output_dir, history)
    else:
        history = {
            "metadata": metadata,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "runs": [],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with _atomic_output(history_path) as stream:
            pickle.dump(history, stream, protocol=pickle.HIGHEST_PROTOCOL)

    completed_pairs = len(history["runs"]) // n_runs
    for K, H in Q1_GRID[completed_pairs:]:
        for run in range(n_runs):
            model = DoubleKMeans(
                n_row_clusters=K,
                n_col_clusters=H,
                max_iter=max_iter,
                random_state=run,
            ).fit(X)
            if not np.isfinite(model.objective_):
                raise RuntimeError(f"Non-finite objective at K={K}, H={H}, run={run}.")
            history["runs"].append({
                "run": run,
                "K": K,
                "H": H,
                "objective": model.objective_,
                "n_iter": model.n_iter_,
                "converged": model.converged_,
                "n_relocations": model.n_relocations_,
                "labels": model.labels_.copy(),
                "column_labels": model.column_labels_.copy(),
                "prototypes": model.prototypes_.copy(),
                "objective_history": model.objective_history_.copy(),
            })
            LOGGER.info(
                "K=%d H=%d run=%d/%d objective=%.10g iterations=%d",
                K, H, run + 1, n_runs, model.objective_, model.n_iter_,
            )
            if not model.converged_:
                LOGGER.warning(
                    "K=%d H=%d seed=%d reached max_iter=%d without convergence.",
                    K, H, run, max_iter,
                )

        with _atomic_output(history_path) as stream:
            pickle.dump(history, stream, protocol=pickle.HIGHEST_PROTOCOL)
        _save_outputs(output_dir, history)
        LOGGER.info("Checkpoint saved for K=%d H=%d.", K, H)

    return pd.read_csv(output_dir / "q1_executions_summary.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/spambase.data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run_question_1(
        filepath=args.data,
        output_dir=args.output_dir,
        n_runs=args.n_runs,
        max_iter=args.max_iter,
        resume=args.resume,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
