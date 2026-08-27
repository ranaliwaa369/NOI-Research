"""Tests for aggregate repeated Track A analysis."""

from statistics import fmean, stdev

import pytest

from src.evaluation.seen_item_repeated_analysis import (
    RepeatedTrackAAnalysisError,
    analyze_repeated_track_a,
)
from src.evaluation.seen_item_repeated_config import (
    load_seen_item_repeated_config,
)


RESULTS_DIRECTORY = (
    "results/v0.2.1/repeated_track_a"
)
CONFIG_PATH = (
    "configs/"
    "seen_item_repeated_evaluation_v0.2.1.yaml"
)
HASH_PATH = (
    "configs/"
    "seen_item_repeated_evaluation_v0.2.1.sha256"
)


@pytest.fixture(scope="module")
def repeated_config():
    return load_seen_item_repeated_config(
        CONFIG_PATH,
        HASH_PATH,
    )


@pytest.fixture(scope="module")
def analysis(repeated_config):
    protocol_sha256 = (
        open(
            HASH_PATH,
            encoding="utf-8",
        )
        .read()
        .strip()
        .split()[0]
    )

    return analyze_repeated_track_a(
        RESULTS_DIRECTORY,
        repeated_config=repeated_config,
        repeated_protocol_sha256=(
            protocol_sha256
        ),
    )


def test_all_ten_locked_runs_are_analyzed(
    analysis,
) -> None:
    assert analysis.run_ids == tuple(
        f"seed-{index:02d}"
        for index in range(1, 11)
    )

    assert len(analysis.source_sha256) == 10
    assert all(
        len(value) == 64
        for value in analysis.source_sha256
    )

    assert analysis.reachable_event_fractions == (
        1.0,
    ) * 10

    assert analysis.oracle_used is False
    assert analysis.final_test_tuning_used is False


def test_every_system_and_metric_is_summarized(
    analysis,
) -> None:
    systems = {
        "memory_only",
        "ridge_only",
        "hybrid",
    }
    metrics = {
        "recall_at_1",
        "recall_at_10",
        "mean_reciprocal_rank",
        "ndcg_at_10",
    }

    observed = {
        (
            summary.system,
            summary.metric,
        )
        for summary in analysis.system_summaries
    }

    assert observed == {
        (system, metric)
        for system in systems
        for metric in metrics
    }


def test_descriptive_statistics_use_seed_as_unit(
    analysis,
) -> None:
    summary = analysis.summary_for(
        "memory_only",
        "mean_reciprocal_rank",
    )

    assert summary.count == 10
    assert len(summary.values) == 10
    assert summary.mean == fmean(summary.values)
    assert summary.standard_deviation == stdev(
        summary.values
    )
    assert summary.minimum == min(summary.values)
    assert summary.maximum == max(summary.values)


def test_primary_comparison_is_memory_minus_ridge(
    analysis,
) -> None:
    comparison = analysis.primary_comparison

    assert comparison.left_system == "memory_only"
    assert comparison.right_system == "ridge_only"
    assert (
        comparison.direction
        == "memory_only minus ridge_only"
    )
    assert (
        comparison.metric
        == "mean_reciprocal_rank"
    )

    memory = analysis.summary_for(
        "memory_only",
        "mean_reciprocal_rank",
    )
    ridge = analysis.summary_for(
        "ridge_only",
        "mean_reciprocal_rank",
    )

    expected_differences = tuple(
        left - right
        for left, right in zip(
            memory.values,
            ridge.values,
            strict=True,
        )
    )

    assert comparison.differences == (
        expected_differences
    )
    assert comparison.mean_difference == fmean(
        expected_differences
    )

    assert (
        comparison.wins
        + comparison.ties
        + comparison.losses
        == 10
    )

    assert (
        comparison.confidence_interval_lower
        <= comparison.mean_difference
        <= comparison.confidence_interval_upper
    )


def test_locked_bootstrap_is_deterministic(
    repeated_config,
    analysis,
) -> None:
    protocol_sha256 = (
        open(
            HASH_PATH,
            encoding="utf-8",
        )
        .read()
        .strip()
        .split()[0]
    )

    repeated = analyze_repeated_track_a(
        RESULTS_DIRECTORY,
        repeated_config=repeated_config,
        repeated_protocol_sha256=(
            protocol_sha256
        ),
    )

    assert repeated == analysis
    assert (
        analysis.bootstrap_seed
        == repeated_config.bootstrap_seed
        == 4243
    )
    assert (
        analysis.bootstrap_resamples
        == repeated_config.bootstrap_resamples
        == 10000
    )
    assert analysis.confidence_level == 0.95


def test_wrong_protocol_hash_is_rejected(
    repeated_config,
) -> None:
    with pytest.raises(
        RepeatedTrackAAnalysisError,
        match="protocol SHA-256",
    ):
        analyze_repeated_track_a(
            RESULTS_DIRECTORY,
            repeated_config=repeated_config,
            repeated_protocol_sha256="0" * 64,
        )
