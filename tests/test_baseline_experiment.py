"""Tests for the prespecified baseline experiment runner."""

from __future__ import annotations

import pytest

from src.baselines.retrieval_baselines import BaselineKind
from src.evaluation.baseline_experiment import (
    BaselineExperimentError,
    BaselineExperimentResult,
    BaselineMetricSummary,
    run_baseline_experiment,
)
from src.evaluation.synthetic_generator import generate_synthetic_pilot
from src.evaluation.synthetic_records import SplitLabel


SYNTHETIC_CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"


@pytest.fixture(scope="module")
def pilot_dataset():
    """Create one deterministic synthetic pilot dataset."""

    return generate_synthetic_pilot(
        SYNTHETIC_CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )


@pytest.fixture(scope="module")
def experiment_result(
    pilot_dataset,
) -> BaselineExperimentResult:
    """Run the complete locked baseline experiment once."""

    return run_baseline_experiment(
        pilot_dataset,
        top_k=10,
        random_seed=2026,
        ridge_alpha=1.0,
    )


def test_experiment_runs_all_baseline_split_combinations(
    experiment_result: BaselineExperimentResult,
) -> None:
    """Four baselines across two splits must produce eight results."""

    assert len(experiment_result.summaries) == 8
    assert len(experiment_result.evaluations) == 8

    observed = {
        (
            summary.baseline,
            summary.split,
        )
        for summary in experiment_result.summaries
    }

    expected = {
        (
            baseline,
            split,
        )
        for split in (
            SplitLabel.VALIDATION,
            SplitLabel.OOD_TEST,
        )
        for baseline in BaselineKind
    }

    assert observed == expected


def test_validation_and_ood_counts_are_correct(
    experiment_result: BaselineExperimentResult,
) -> None:
    """Summaries must preserve the locked evaluation sizes."""

    for summary in experiment_result.summaries:
        if summary.split is SplitLabel.VALIDATION:
            assert summary.event_count == 20
        elif summary.split is SplitLabel.OOD_TEST:
            assert summary.event_count == 40
        else:
            pytest.fail(
                f"Unexpected split: {summary.split}"
            )


def test_all_metrics_are_finite_probabilities(
    experiment_result: BaselineExperimentResult,
) -> None:
    """Every reported retrieval metric must remain in [0, 1]."""

    for summary in experiment_result.summaries:
        metrics = (
            summary.recall_at_1,
            summary.recall_at_10,
            summary.mean_reciprocal_rank,
            summary.ndcg_at_10,
        )

        assert all(
            0.0 <= metric <= 1.0
            for metric in metrics
        )
        assert summary.recall_at_10 >= summary.recall_at_1


def test_get_summary_returns_requested_result(
    experiment_result: BaselineExperimentResult,
) -> None:
    """A summary must be retrievable by its exact key."""

    summary = experiment_result.get_summary(
        BaselineKind.RIDGE_FUSION,
        SplitLabel.VALIDATION,
    )

    assert summary.baseline is BaselineKind.RIDGE_FUSION
    assert summary.split is SplitLabel.VALIDATION
    assert summary.event_count == 20


def test_unknown_summary_key_is_rejected(
    pilot_dataset,
) -> None:
    """Requesting an unexecuted split must fail explicitly."""

    result = run_baseline_experiment(
        pilot_dataset,
        baselines=(BaselineKind.RANDOM,),
        splits=(SplitLabel.VALIDATION,),
    )

    with pytest.raises(
        BaselineExperimentError,
        match="No summary exists",
    ):
        result.get_summary(
            BaselineKind.RANDOM,
            SplitLabel.OOD_TEST,
        )


def test_strongest_baseline_is_observed_result(
    experiment_result: BaselineExperimentResult,
) -> None:
    """Strongest-baseline selection must use maximum observed MRR."""

    strongest = experiment_result.strongest_baseline(
        SplitLabel.VALIDATION
    )

    validation_summaries = tuple(
        summary
        for summary in experiment_result.summaries
        if summary.split is SplitLabel.VALIDATION
    )

    maximum_mrr = max(
        summary.mean_reciprocal_rank
        for summary in validation_summaries
    )

    assert strongest in validation_summaries
    assert (
        strongest.mean_reciprocal_rank
        == pytest.approx(maximum_mrr)
    )


def test_records_preserve_deterministic_order(
    experiment_result: BaselineExperimentResult,
) -> None:
    """Table records must follow the executed split-baseline order."""

    records = experiment_result.to_records()

    assert len(records) == 8
    assert records[0]["split"] == "validation"
    assert records[0]["baseline"] == "random"
    assert records[-1]["split"] == "ood_test"
    assert records[-1]["baseline"] == "ridge_fusion"


def test_repeated_experiments_are_deterministic(
    pilot_dataset,
    experiment_result: BaselineExperimentResult,
) -> None:
    """Identical settings must reproduce identical metric records."""

    repeated = run_baseline_experiment(
        pilot_dataset,
        top_k=10,
        random_seed=2026,
        ridge_alpha=1.0,
    )

    assert (
        experiment_result.to_records()
        == repeated.to_records()
    )


def test_empty_baseline_selection_is_rejected(
    pilot_dataset,
) -> None:
    """At least one comparison system must be selected."""

    with pytest.raises(
        BaselineExperimentError,
        match="At least one baseline",
    ):
        run_baseline_experiment(
            pilot_dataset,
            baselines=(),
        )


def test_empty_split_selection_is_rejected(
    pilot_dataset,
) -> None:
    """At least one evaluation split must be selected."""

    with pytest.raises(
        BaselineExperimentError,
        match="At least one split",
    ):
        run_baseline_experiment(
            pilot_dataset,
            splits=(),
        )


def test_duplicate_baselines_are_rejected(
    pilot_dataset,
) -> None:
    """The same baseline cannot be counted twice."""

    with pytest.raises(
        BaselineExperimentError,
        match="cannot contain duplicates",
    ):
        run_baseline_experiment(
            pilot_dataset,
            baselines=(
                BaselineKind.RANDOM,
                BaselineKind.RANDOM,
            ),
        )


def test_duplicate_splits_are_rejected(
    pilot_dataset,
) -> None:
    """The same split cannot be evaluated twice."""

    with pytest.raises(
        BaselineExperimentError,
        match="cannot contain duplicates",
    ):
        run_baseline_experiment(
            pilot_dataset,
            splits=(
                SplitLabel.VALIDATION,
                SplitLabel.VALIDATION,
            ),
        )


@pytest.mark.parametrize(
    "invalid_top_k",
    (
        0,
        9,
        True,
    ),
)
def test_top_k_below_locked_metric_depth_is_rejected(
    pilot_dataset,
    invalid_top_k,
) -> None:
    """Recall@10 and nDCG@10 require rankings of depth ten."""

    with pytest.raises(
        BaselineExperimentError,
        match="at least 10",
    ):
        run_baseline_experiment(
            pilot_dataset,
            top_k=invalid_top_k,
        )


def test_invalid_metric_summary_is_rejected() -> None:
    """Metric summaries cannot contain values outside [0, 1]."""

    with pytest.raises(
        BaselineExperimentError,
        match="between 0 and 1",
    ):
        BaselineMetricSummary(
            baseline=BaselineKind.RANDOM,
            split=SplitLabel.VALIDATION,
            event_count=20,
            recall_at_1=1.5,
            recall_at_10=1.5,
            mean_reciprocal_rank=0.5,
            ndcg_at_10=0.5,
        )