"""Tests for the Question 2 statistical-analysis outputs."""

import pandas as pd
import pytest

from src.analysis.statistical_analysis import (
    CLASSIFIER_NAMES,
    DATASET_NAMES,
    compute_confidence_intervals,
    run_statistical_analysis,
    validate_results,
)
from src.utils.statistical_tests import perform_friedman_nemenyi


def _make_cv_results() -> pd.DataFrame:
    rows = []
    for dataset_index, dataset in enumerate(DATASET_NAMES):
        for classifier_index, classifier in enumerate(CLASSIFIER_NAMES):
            for repetition in range(1, 31):
                for fold_in_rep in range(1, 11):
                    fold_id = (repetition - 1) * 10 + fold_in_rep - 1
                    score = (
                        0.4
                        + 0.08 * classifier_index
                        + 0.01 * dataset_index
                        + 0.0001 * fold_id
                    )
                    rows.append(
                        {
                            "fold_id": fold_id,
                            "repetition": repetition,
                            "fold_in_rep": fold_in_rep,
                            "dataset": dataset,
                            "classifier": classifier,
                            "error": 1 - score,
                            "precision": score,
                            "recall": score,
                            "f1": score,
                        }
                    )
    return pd.DataFrame(rows)


def test_validate_results_accepts_complete_30_by_10_results():
    df_results = _make_cv_results()

    validate_results(df_results)


def test_validate_results_rejects_missing_fold():
    df_results = _make_cv_results().iloc[1:]

    with pytest.raises(ValueError, match="Expected 3000"):
        validate_results(df_results)


def test_confidence_intervals_summarize_30_repetition_means():
    df_results = _make_cv_results()

    confidence_intervals = compute_confidence_intervals(df_results)

    assert len(confidence_intervals) == 40
    assert set(confidence_intervals["n_repetitions"]) == {30}
    assert (confidence_intervals["ci_lower"] <= confidence_intervals["mean"]).all()
    assert (confidence_intervals["mean"] <= confidence_intervals["ci_upper"]).all()


def test_friedman_nemenyi_returns_statistics_and_posthoc():
    classifier_count = len(CLASSIFIER_NAMES)
    scores = pd.DataFrame(
        {
            classifier: [
                0.2 + 0.15 * index + 0.001 * block
                for block in range(30)
            ]
            for index, classifier in enumerate(CLASSIFIER_NAMES)
        }
    )

    result = perform_friedman_nemenyi(scores, "synthetic metric")

    assert classifier_count == 5
    assert result.statistic > 0
    assert result.p_value < 0.05
    assert result.nemenyi_p_values is not None
    assert result.nemenyi_p_values.shape == (5, 5)


def test_friedman_rejects_undefined_identical_results():
    scores = pd.DataFrame(
        {classifier: [0.5] * 10 for classifier in CLASSIFIER_NAMES}
    )

    with pytest.raises(ValueError, match="undefined"):
        perform_friedman_nemenyi(scores, "identical metric")


def test_analysis_saves_outputs_and_prints_summary(tmp_path, capsys):
    input_csv = tmp_path / "cv_results.csv"
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    _make_cv_results().to_csv(input_csv, index=False)

    run_statistical_analysis(input_csv, results_dir, plots_dir)
    terminal_output = capsys.readouterr().out

    assert "F1 macro=" in terminal_output
    assert "Friedman chi-square=" in terminal_output
    assert (results_dir / "confidence_intervals.csv").is_file()
    assert (results_dir / "friedman_results.txt").is_file()
    assert (results_dir / "nemenyi_pvalues.csv").is_file()
    assert (plots_dir / "comparison_boxplots.png").is_file()
