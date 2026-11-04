"""Summarize and statistically compare the Question 2 cross-validation results."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem, t

from src.utils.statistical_tests import perform_friedman_nemenyi


DATASET_NAMES = ("original_2_classes", "kstar_classes")
CLASSIFIER_NAMES = (
    "GaussianBayes",
    "BayesianKNN",
    "ParzenWindow",
    "LogisticRegression",
    "MajorityVoting",
)
METRICS = ("error", "precision", "recall", "f1")
METRIC_LABELS = {
    "error": "Taxa de erro",
    "precision": "Precisão macro",
    "recall": "Recall macro",
    "f1": "F1 macro",
}
EXPECTED_REPETITIONS = 30
EXPECTED_FOLDS_PER_REPETITION = 10


def validate_results(df_results: pd.DataFrame) -> None:
    """Validate that the DataFrame contains a complete, aligned 30×10 experiment.

    Args:
        df_results (pd.DataFrame): Cross-validation results DataFrame loaded from
            ``cv_30x10_results.csv``.

    Raises:
        ValueError: If required columns are missing, row counts or categories
            do not match the 30×10 protocol, values are non-numeric/non-finite
            or outside [0, 1], or folds are misaligned.
    """
    required_columns = {
        "fold_id",
        "repetition",
        "fold_in_rep",
        "dataset",
        "classifier",
        *METRICS,
    }
    missing_columns = required_columns.difference(df_results.columns)
    if missing_columns:
        raise ValueError(f"Results CSV is missing columns: {sorted(missing_columns)}")

    expected_rows = (
        EXPECTED_REPETITIONS
        * EXPECTED_FOLDS_PER_REPETITION
        * len(DATASET_NAMES)
        * len(CLASSIFIER_NAMES)
    )
    if len(df_results) != expected_rows:
        raise ValueError(f"Expected {expected_rows} result rows, found {len(df_results)}.")
    if set(df_results["dataset"].unique()) != set(DATASET_NAMES):
        raise ValueError("Results must contain exactly the original and kstar datasets.")
    if set(df_results["classifier"].unique()) != set(CLASSIFIER_NAMES):
        raise ValueError("Results must contain all five expected classifiers.")
    if df_results.duplicated(["dataset", "classifier", "fold_id"]).any():
        raise ValueError("Results contain duplicate dataset/classifier/fold rows.")

    for column in ("fold_id", "repetition", "fold_in_rep", *METRICS):
        try:
            values = pd.to_numeric(df_results[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Column {column!r} must contain numeric values.") from error
        if not np.isfinite(values).all():
            raise ValueError(f"Column {column!r} contains non-finite values.")

    for metric in METRICS:
        if not df_results[metric].between(0, 1).all():
            raise ValueError(f"Metric {metric!r} must be between 0 and 1.")

    expected_keys = {
        (repetition, fold_in_rep, (repetition - 1) * EXPECTED_FOLDS_PER_REPETITION + fold_in_rep - 1)
        for repetition in range(1, EXPECTED_REPETITIONS + 1)
        for fold_in_rep in range(1, EXPECTED_FOLDS_PER_REPETITION + 1)
    }
    for dataset_name in DATASET_NAMES:
        for classifier_name in CLASSIFIER_NAMES:
            subset = df_results.loc[
                (df_results["dataset"] == dataset_name)
                & (df_results["classifier"] == classifier_name)
            ]
            actual_keys = set(
                subset[["repetition", "fold_in_rep", "fold_id"]]
                .itertuples(index=False, name=None)
            )
            if actual_keys != expected_keys:
                raise ValueError(
                    f"Incomplete or misaligned 30×10 folds for "
                    f"{dataset_name}/{classifier_name}."
                )


def compute_confidence_intervals(
    df_results: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compute t-based CIs from the 30 per-repetition mean scores.

    Averaging the ten folds within each repetition avoids treating all 300
    overlapping outer folds as independent observations for these intervals.

    Args:
        df_results (pd.DataFrame): Aligned 30×10 cross-validation results.
        confidence_level (float, optional): Confidence level in (0, 1).
            Defaults to 0.95.

    Returns:
        pd.DataFrame: Table containing ``dataset``, ``classifier``, ``metric``,
            ``mean``, ``ci_lower``, ``ci_upper``, ``n_repetitions``, and
            ``confidence_level``.

    Raises:
        ValueError: If ``confidence_level`` is outside (0, 1), required columns
            are missing, or fewer than two repetitions are present.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if "repetition" not in df_results or "dataset" not in df_results or "classifier" not in df_results:
        raise ValueError("Results must include repetition, dataset, and classifier columns.")
    missing_metrics = set(METRICS).difference(df_results.columns)
    if missing_metrics:
        raise ValueError(f"Results are missing metrics: {sorted(missing_metrics)}")

    rows = []
    for (dataset_name, classifier_name), group in df_results.groupby(
        ["dataset", "classifier"], sort=False
    ):
        for metric in METRICS:
            repetition_means = group.groupby("repetition", sort=True)[metric].mean()
            n_repetitions = len(repetition_means)
            if n_repetitions < 2:
                raise ValueError("At least two repetitions are required to compute a CI.")
            mean = float(repetition_means.mean())
            margin = float(
                t.ppf((1 + confidence_level) / 2, df=n_repetitions - 1)
                * sem(repetition_means.to_numpy())
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "classifier": classifier_name,
                    "metric": metric,
                    "mean": mean,
                    "ci_lower": mean - margin,
                    "ci_upper": mean + margin,
                    "n_repetitions": n_repetitions,
                    "confidence_level": confidence_level,
                }
            )
    return pd.DataFrame(rows)


def _print_confidence_intervals(confidence_intervals: pd.DataFrame) -> None:
    """Print formatted point estimates and 95% confidence intervals to stdout.

    Args:
        confidence_intervals (pd.DataFrame): Output of
            ``compute_confidence_intervals``.
    """
    print("=" * 96)
    print("MÉTRICAS E IC 95% — MÉDIA DAS 30 REPETIÇÕES (10 FOLDS POR REPETIÇÃO)")
    print("=" * 96)
    for dataset_name in DATASET_NAMES:
        print(f"\nDataset: {dataset_name}")
        subset = confidence_intervals[confidence_intervals["dataset"] == dataset_name]
        for classifier_name in CLASSIFIER_NAMES:
            classifier = subset[subset["classifier"] == classifier_name]
            metrics = " | ".join(
                f"{METRIC_LABELS[row.metric]}={row.mean:.4f} "
                f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}]"
                for row in classifier.itertuples(index=False)
            )
            print(f"  {classifier_name:20s} | {metrics}")


def _plot_metric_distributions(df_results: pd.DataFrame, output_path: Path) -> None:
    """Generate and save boxplots comparing classifiers across datasets and metrics.

    Args:
        df_results (pd.DataFrame): Aligned 30×10 cross-validation results.
        output_path (Path): Destination file path for the PNG plot.
    """
    figure, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=len(DATASET_NAMES),
        figsize=(19, 17),
    )
    colors = plt.get_cmap("tab10").colors
    classifier_labels = (
        "Gaussian\nBayes",
        "Bayesian\nk-NN",
        "Parzen",
        "Logistic\nRegression",
        "Majority\nVoting",
    )

    for row_index, metric in enumerate(METRICS):
        for column_index, dataset_name in enumerate(DATASET_NAMES):
            axis = axes[row_index, column_index]
            dataset = df_results[df_results["dataset"] == dataset_name]
            values = [
                dataset.loc[dataset["classifier"] == classifier_name, metric].to_numpy()
                for classifier_name in CLASSIFIER_NAMES
            ]
            boxes = axis.boxplot(
                values,
                patch_artist=True,
                showfliers=False,
                widths=0.65,
                medianprops={"color": "black", "linewidth": 1.4},
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": 4,
                },
            )
            for box, color in zip(boxes["boxes"], colors):
                box.set_facecolor(color)
                box.set_alpha(0.75)
            if row_index == 0:
                axis.set_title(
                    dataset_name.replace("_", " "),
                    fontsize=12,
                    fontweight="bold",
                    pad=12,
                )
            if column_index == 0:
                axis.set_ylabel(METRIC_LABELS[metric], fontsize=10)
            axis.set_xticks(range(1, len(CLASSIFIER_NAMES) + 1))
            axis.set_xticklabels(classifier_labels, fontsize=9)
            axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.3f}"))
            all_values = np.concatenate(values)
            lower, upper = np.quantile(all_values, [0.02, 0.98])
            padding = max((upper - lower) * 0.12, np.finfo(float).eps)
            axis.set_ylim(lower - padding, upper + padding)
            axis.grid(axis="y", alpha=0.25, linestyle="--")
            axis.set_axisbelow(True)
            axis.tick_params(axis="y", labelsize=8)

    figure.suptitle(
        "Comparação dos classificadores — validação cruzada estratificada 30×10\n"
        "Cada painel usa escala vertical própria; diamantes indicam a média",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.2, w_pad=1.8)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def run_statistical_analysis(
    input_csv: str | Path = "results/q2/cv_30x10_results.csv",
    results_dir: str | Path = "results",
    plots_dir: str | Path = "plots",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Analyze the 30×10 CSV and save/print the requested 2.3 outputs.

    Args:
        input_csv (str or Path, optional): Path to the 30×10 CV results CSV.
            Defaults to "results/q2/cv_30x10_results.csv".
        results_dir (str or Path, optional): Directory where CSV and TXT
            statistical summaries are saved. Defaults to "results".
        plots_dir (str or Path, optional): Directory where the comparison
            boxplot PNG is saved. Defaults to "plots".
        alpha (float, optional): Significance threshold for Friedman and
            Nemenyi tests. Defaults to 0.05.

    Returns:
        pd.DataFrame: DataFrame of point estimates and confidence intervals.

    Raises:
        ValueError: If the input CSV fails ``validate_results``.
    """
    input_csv = Path(input_csv)
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    df_results = pd.read_csv(input_csv)
    validate_results(df_results)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    confidence_intervals = compute_confidence_intervals(df_results)
    confidence_path = results_dir / "confidence_intervals.csv"
    confidence_intervals.to_csv(confidence_path, index=False)
    _print_confidence_intervals(confidence_intervals)
    print(f"\nIntervalos de confiança salvos em: {confidence_path}")

    test_summaries = []
    nemenyi_matrices = []
    for dataset_name in DATASET_NAMES:
        dataset = df_results[df_results["dataset"] == dataset_name]
        for metric in METRICS:
            aligned = dataset.pivot(
                index="fold_id",
                columns="classifier",
                values=metric,
            ).reindex(columns=CLASSIFIER_NAMES).sort_index()
            result = perform_friedman_nemenyi(
                aligned,
                metric_name=f"{dataset_name} — {METRIC_LABELS[metric]}",
                alpha=alpha,
            )
            test_summaries.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "friedman_statistic": result.statistic,
                    "p_value": result.p_value,
                    "alpha": alpha,
                    "significant": result.p_value < alpha,
                    "nemenyi_performed": result.nemenyi_p_values is not None,
                }
            )
            if result.nemenyi_p_values is not None:
                matrix = result.nemenyi_p_values.reindex(
                    index=CLASSIFIER_NAMES,
                    columns=CLASSIFIER_NAMES,
                )
                matrix.index.name = "classifier"
                matrix = matrix.reset_index()
                matrix.insert(0, "metric", metric)
                matrix.insert(0, "dataset", dataset_name)
                nemenyi_matrices.append(matrix)

    tests_path = results_dir / "friedman_results.txt"
    with tests_path.open("w", encoding="utf-8") as output:
        output.write("TESTES DE FRIEDMAN E PÓS-TESTE DE NEMENYI\n")
        output.write(f"Alpha: {alpha}\n")
        output.write("Bloco: fold externo alinhado entre classificadores (300 por dataset).\n\n")
        for result in test_summaries:
            output.write(
                f"{result['dataset']} | {METRIC_LABELS[result['metric']]}: "
                f"chi-square={result['friedman_statistic']:.6f}, "
                f"p={result['p_value']:.6e}, "
                f"significativo={result['significant']}, "
                f"Nemenyi executado={result['nemenyi_performed']}\n"
            )
        output.write(
            "\nNota: folds e repetições se sobrepõem; os p-valores não devem ser "
            "interpretados como evidência de generalização para outros datasets.\n"
        )
    print(f"\nResultados de Friedman salvos em: {tests_path}")

    classifier_columns = ["dataset", "metric", "classifier", *CLASSIFIER_NAMES]
    nemenyi_path = results_dir / "nemenyi_pvalues.csv"
    nemenyi_output = (
        pd.concat(nemenyi_matrices, ignore_index=True)
        if nemenyi_matrices
        else pd.DataFrame(columns=classifier_columns)
    )
    nemenyi_output.to_csv(nemenyi_path, index=False)
    print(f"Matrizes de Nemenyi salvas em: {nemenyi_path}")

    plot_path = plots_dir / "comparison_boxplots.png"
    _plot_metric_distributions(df_results, plot_path)
    print(f"Gráfico comparativo salvo em: {plot_path}")
    return confidence_intervals


def main() -> None:
    """Parse CLI arguments and run the Question 2 statistical analysis."""
    parser = argparse.ArgumentParser(
        description="Summarize and compare Question 2 30×10 CV results."
    )
    parser.add_argument(
        "--input",
        default="results/q2/cv_30x10_results.csv",
        help="Path to the nested-CV results CSV.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for confidence intervals and statistical test outputs.",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory for the comparison plot.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    run_statistical_analysis(
        input_csv=args.input,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
