"""Summarize the 30x10 CV results into the boxplot statistics drawn in the report.

The report draws Figure 5 with pgfplots from ``dados/boxplots.csv``. This script
creates that file from ``results/q2/cv_30x10_results.csv`` using the same
conventions as ``src/analysis/statistical_analysis.py`` (matplotlib defaults):
quartiles by linear interpolation, whiskers at the most extreme fold still
within 1.5 IQR of the box, outliers not drawn, and the mean marked separately.

Row order is metric, then dataset, then classifier, so each figure panel
(metric, dataset) uses five consecutive rows: panel k uses rows 5k to 5k+4.

Run from the repository root:
    poetry run python docs/relatorio/gerar_boxplots.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "q2" / "cv_30x10_results.csv"
TARGET = Path(__file__).resolve().parent / "dados" / "boxplots.csv"

METRICS = ["error", "precision", "recall", "f1"]
DATASETS = ["original_2_classes", "kstar_classes"]
CLASSIFIERS = [
    "GaussianBayes",
    "BayesianKNN",
    "ParzenWindow",
    "LogisticRegression",
    "MajorityVoting",
]
WHISKER_IQR = 1.5
EXPECTED_FOLDS = 300


def boxplot_stats(values: np.ndarray) -> dict[str, float]:
    """Return quartiles, whiskers and mean as matplotlib's boxplot computes them."""
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lower_whisker = values[values >= q1 - WHISKER_IQR * iqr].min()
    upper_whisker = values[values <= q3 + WHISKER_IQR * iqr].max()
    return {
        "lower_whisker": lower_whisker,
        "q1": q1,
        "median": median,
        "q3": q3,
        "upper_whisker": upper_whisker,
        "mean": values.mean(),
    }


def main() -> None:
    results = pd.read_csv(SOURCE)
    rows = []
    for metric in METRICS:
        for dataset in DATASETS:
            for classifier in CLASSIFIERS:
                mask = (results["dataset"] == dataset) & (results["classifier"] == classifier)
                values = results.loc[mask, metric].to_numpy(dtype=float)
                if len(values) != EXPECTED_FOLDS:
                    raise ValueError(
                        f"{dataset}/{classifier}: expected {EXPECTED_FOLDS} folds, got {len(values)}"
                    )
                rows.append(
                    {"metric": metric, "dataset": dataset, "classifier": classifier, "n": len(values)}
                    | boxplot_stats(values)
                )
    TARGET.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(TARGET, index=False, float_format="%.6f")
    print(f"Wrote {len(rows)} rows to {TARGET}")


if __name__ == "__main__":
    main()
