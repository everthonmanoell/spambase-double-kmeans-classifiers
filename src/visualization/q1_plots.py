"""Required Q1 plots from the saved winning partition."""

from pathlib import Path

from matplotlib.figure import Figure
import numpy as np

from src.experiments.question_1 import _atomic_output


def _save_figure(figure, path):
    """Atomically write a Matplotlib figure to disk as a PNG file.

    Args:
        figure (Figure): Matplotlib figure instance to render.
        path (Path): Destination file path.
    """
    try:
        with _atomic_output(path) as stream:
            figure.savefig(stream, format="png", dpi=160)
    finally:
        figure.clear()


def plot_q1_results(best, contingency, plots_dir):
    """Save contingency, block prototypes, and objective convergence plots.

    Args:
        best (dict): Winning Double k-means run dictionary containing ``K``,
            ``H``, ``run``, ``prototypes``, and ``objective_history``.
        contingency (pd.DataFrame): Contingency table comparing original classes
            and object clusters.
        plots_dir (str or Path): Output directory where PNG plots are written.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(7, 4.5), layout="constrained")
    axes = figure.subplots()
    counts = contingency.to_numpy()
    image = axes.imshow(counts, cmap="Blues", aspect="auto")
    axes.set_xticks(range(len(contingency.columns)), labels=contingency.columns)
    axes.set_yticks(range(len(contingency.index)), labels=contingency.index)
    axes.set_xlabel("Object cluster (arbitrary ID)")
    axes.set_ylabel("Original class")
    axes.set_title("Q1: original classes vs object clusters")
    for (i, j), count in np.ndenumerate(counts):
        axes.text(j, i, str(count), ha="center", va="center",
                  color="white" if count > counts.max() / 2 else "black")
    figure.colorbar(image, ax=axes, label="Number of observations")
    _save_figure(figure, plots_dir / "confusion_matrix_q1.png")

    figure = Figure(figsize=(7, 4.5), layout="constrained")
    axes = figure.subplots()
    G = best["prototypes"]
    limit = max(float(np.max(np.abs(G))), np.finfo(float).eps)
    image = axes.imshow(G, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes.set_xticks(range(best["H"]))
    axes.set_yticks(range(best["K"]))
    axes.set_xlabel("Variable cluster")
    axes.set_ylabel("Object cluster")
    axes.set_title(f"Block prototypes G (K={best['K']}, H={best['H']})")
    for (i, j), value in np.ndenumerate(G):
        axes.text(j, i, f"{value:.4f}", ha="center", va="center",
                  color="white" if abs(value) > limit / 2 else "black")
    figure.colorbar(image, ax=axes, label="Mean in standardized feature units")
    _save_figure(figure, plots_dir / "prototypes_heatmap.png")

    figure = Figure(figsize=(7, 4.5), layout="constrained")
    axes = figure.subplots()
    values = best["objective_history"]
    axes.plot(np.arange(len(values)), values, marker="o", markersize=3)
    axes.set_xlabel("Iteration (0 = initialization)")
    axes.set_ylabel("Objective W(G, U, V)")
    axes.set_title(f"Q1 convergence (K={best['K']}, H={best['H']}, seed={best['run']})")
    axes.grid(alpha=0.25)
    _save_figure(figure, plots_dir / "objective_convergence.png")
