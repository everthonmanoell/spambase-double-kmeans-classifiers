"""Question 1: double k-means on Spambase.

Runs DoubleKMeans 100 times for each K in {2, 3, 4} and H in {1, ..., K}, keeps the run with 
the smallest W for each (K, H), computes the silhouette of each pair's object partition and
picks (K*, H*) by the largest silhouette. 
For that run it reports the adjusted Rand index against the a priori classes, the prototype 
matrix G, the confusion matrix and the plot of W over the iterations.

Run from the repository root:
    python -m src.experiments.q1_double_kmeans
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.clustering.double_kmeans import DoubleKMeans
from src.utils.dataloader import load_spambase_original

DATA_PATH = "dataset/spambase.data"
NAMES_PATH = "dataset/spambase.names"
OUTPUT_DIR = Path("results/q1")

K_VALUES = (2, 3, 4)
N_RUNS = 100
RANDOM_STATE = 42

# Light chart tokens: one accent for the selected pair, gray for the rest.
ACCENT = "#2a78d6"
DEEMPHASIS = "#c3c2b7"
SURFACE = "#fcfcfb"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def load_feature_names(filepath=NAMES_PATH):
    """Read the 57 variable names from the "name: continuous." lines."""
    with open(filepath) as f:
        return [line.split(":")[0] for line in f
                if line.rstrip().endswith("continuous.")]


def run_pair(X, K, H, seeds):
    """Run double k-means once per seed and keep the run with the smallest W.

    Args:
        X (numpy.ndarray of shape (n_samples, n_features)): Data matrix.
        K (int): Number of object groups.
        H (int): Number of variable groups.
        seeds (list of int): One seed per run.

    Returns:
        tuple: (best fitted DoubleKMeans, list with one dict per run).
    """
    best, runs = None, []
    for seed in seeds:
        model = DoubleKMeans(n_row_clusters=K, n_col_clusters=H,
                             random_state=seed).fit(X)
        runs.append({"K": K, "H": H, "seed": seed,
                     "objective": model.objective_,
                     "n_iter": model.n_iter_,
                     "converged": model.converged_,
                     "n_relocations": model.n_relocations_})
        # Strict "<": on equal W the earlier run is kept.
        if best is None or model.objective_ < best.objective_:
            best = model
    return best, runs


def select_best_pair(summary):
    """Choose (K*, H*) from the silhouette of each pair's best run.

    Args:
        summary (pandas.DataFrame): One row per (K, H), in the order (2, 1), (2, 2), (3, 1), ..., (4, 4), 
            with columns `K`, `H`, `objective` (W of the best run) and `silhouette`.

    Returns:
        tuple: (K*, H*) as Python ints.
    """
    # TODO(human): pick the pair that maximizes the silhouette and decide
    # what happens on ties.
    raise NotImplementedError


def style_axes(ax):
    """Recessive chart chrome: hairline horizontal grid, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def plot_silhouette(summary, best_pair, path):
    """Bar chart of Sil x (K, H), with the selected pair highlighted."""
    pairs = list(zip(summary["K"], summary["H"]))
    colors = [ACCENT if pair == best_pair else DEEMPHASIS for pair in pairs]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE)
    positions = np.arange(len(pairs))
    ax.bar(positions, summary["silhouette"], width=0.35, color=colors)
    for x, value in zip(positions, summary["silhouette"]):
        ax.annotate(f"{value:.3f}", (x, value), textcoords="offset points",
                    xytext=(0, 3 if value >= 0 else -11), ha="center",
                    fontsize=8, color=TEXT_SECONDARY)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(positions, [f"({k}, {h})" for k, h in pairs])
    ax.set_xlabel("(K, H)")
    ax.set_ylabel("Silhueta")
    ax.set_title(f"Silhueta da melhor execução de cada (K, H); "
                 f"destaque: (K*, H*) = {best_pair}",
                 fontsize=10, color=TEXT_SECONDARY, loc="left")
    style_axes(ax)
    # The silhouette can be negative: the zero line is the baseline, and the margin leaves room for labels under negative bars.
    ax.spines["bottom"].set_visible(False)
    ax.margins(y=0.15)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_objective_history(model, path):
    """Line plot of W at the initialization (iteration 0) and each iteration."""
    history = model.objective_history_

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE)
    ax.plot(np.arange(len(history)), history, color=ACCENT, linewidth=2,
            marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Iteração (0 = inicialização)")
    ax.set_ylabel("Função objetivo W")
    ax.set_title(f"W por iteração, K* = {model.n_row_clusters}, "
                 f"H* = {model.n_col_clusters}",
                 fontsize=10, color=TEXT_SECONDARY, loc="left")
    style_axes(ax)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    X, y = load_spambase_original(DATA_PATH)
    feature_names = load_feature_names()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # One seed per run, drawn in pair order, so the whole experiment depends only on RANDOM_STATE.
    rng = np.random.default_rng(RANDOM_STATE)
    best_models, run_records, summary_rows = {}, [], []
    for K in K_VALUES:
        for H in range(1, K + 1):
            seeds = [int(s) for s in rng.integers(2**31 - 1, size=N_RUNS)]
            best, runs = run_pair(X, K, H, seeds)
            best_models[K, H] = best
            run_records.extend(runs)
            summary_rows.append({
                "K": K, "H": H,
                "objective": best.objective_,
                "silhouette": silhouette_score(X, best.row_labels_),
                "seed": best.random_state,
                "n_iter": best.n_iter_,
                "converged": best.converged_,
                "n_runs_not_converged": sum(not r["converged"] for r in runs),
            })
            print(f"(K={K}, H={H}): W = {best.objective_:.6g}, "
                  f"Sil = {summary_rows[-1]['silhouette']:.4f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    pd.DataFrame(run_records).to_csv(OUTPUT_DIR / "runs.csv", index=False)

    k_star, h_star = select_best_pair(summary)
    best = best_models[k_star, h_star]
    plot_silhouette(summary, (k_star, h_star),
                    OUTPUT_DIR / "silhouette_vs_kh.png")

    # Object groups P1..PK and variable groups Q1..QH, as in the slides.
    row_groups = [f"P{k + 1}" for k in range(k_star)]
    col_groups = [f"Q{h + 1}" for h in range(h_star)]

    # i) Prototype matrix G and the variables in each variable group.
    G = pd.DataFrame(best.prototypes_, index=row_groups, columns=col_groups)
    G.to_csv(OUTPUT_DIR / "prototypes_G.csv")
    variable_groups = pd.DataFrame({
        "variable": feature_names,
        "group": [col_groups[h] for h in best.column_labels_],
    })
    variable_groups.to_csv(OUTPUT_DIR / "variable_groups.csv", index=False)

    # ii) Confusion matrix: K* clusters x 2 classes. pd.crosstab instead of sklearn's confusion_matrix, 
    # which would pad it to a square matrix.
    confusion = pd.crosstab(
        pd.Series([row_groups[k] for k in best.row_labels_], name="cluster"),
        pd.Series(np.where(y == 1, "spam", "non-spam"), name="class"),
    )
    confusion.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    # iv) W over the iterations of the selected run.
    plot_objective_history(best, OUTPUT_DIR / "objective_vs_iteration.png")

    # Object partition for the K*-class version of the dataset in Question 2, in the row order of dataset/spambase.data.
    pd.DataFrame({"cluster": best.row_labels_}).to_csv(
        OUTPUT_DIR / "row_labels_kstar.csv", index=False)

    ari = adjusted_rand_score(y, best.row_labels_)
    selected = {
        "K_star": k_star, "H_star": h_star,
        "seed": best.random_state,
        "objective": best.objective_,
        "silhouette": float(summary.loc[(summary["K"] == k_star)
                                        & (summary["H"] == h_star),
                                        "silhouette"].iloc[0]),
        "adjusted_rand_index": ari,
        "n_iter": best.n_iter_,
        "converged": best.converged_,
        "n_relocations": best.n_relocations_,
    }
    with open(OUTPUT_DIR / "selected_run.json", "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\n(K*, H*) = ({k_star}, {h_star}), ARI = {ari:.4f}")
    if not best.converged_:
        print("Warning: the selected run stopped at max_iter.")
    print(f"\nG:\n{G}\n\nConfusion matrix:\n{confusion}")
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
