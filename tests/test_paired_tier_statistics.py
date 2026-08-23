"""Tests for paired bootstrap tier statistics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import isfinite

import numpy as np
import pytest

from src.baselines.retrieval_baselines import BaselineKind
from src.evaluation.graded_ood import OODTier
from src.evaluation.graded_ood_experiment import (
    run_graded_ood_experiment,
)
from src.evaluation.paired_ood_source_generator import (
    generate_paired_graded_ood_bundle,
)
from src.evaluation.paired_tier_statistics import (
    LOCKED_BOOTSTRAP_RESAMPLES,
    LOCKED_BOOTSTRAP_SEED,
    LOCKED_CONFIDENCE_LEVEL,
    LOCKED_TIER_CONTRASTS,
    PairedTierStatistics,
    PairedTierStatisticsError,
    compute_paired_tier_statistics,
)


@pytest.fixture(scope="module")
def experiment():
    """Run the deterministic graded-OOD pilot experiment."""

    bundle = generate_paired_graded_ood_bundle(
        "configs/synthetic_data.yaml",
        "configs/research_protocol.yaml",
        "configs/protocol_amendment_v0.2.yaml",
        "configs/graded_ood_generation.yaml",
        event_count=200,
    )

    return run_graded_ood_experiment(bundle)


@pytest.fixture(scope="module")
def statistics(experiment) -> PairedTierStatistics:
    """Compute the locked paired statistics once."""

    return compute_paired_tier_statistics(experiment)


def test_all_preregistered_comparisons_exist(
    statistics: PairedTierStatistics,
) -> None:
    """Four baselines by three contrasts must yield twelve results."""

    assert len(statistics.comparisons) == 12

    assert {
        (
            result.baseline,
            result.lower_severity_tier,
            result.higher_severity_tier,
        )
        for result in statistics.comparisons
    } == {
        (baseline, lower, higher)
        for baseline in BaselineKind
        for lower, higher in LOCKED_TIER_CONTRASTS
    }


def test_statistics_use_locked_settings(
    statistics: PairedTierStatistics,
) -> None:
    """Bootstrap settings and analysis unit must remain locked."""

    assert statistics.bootstrap_seed == 4242
    assert statistics.bootstrap_resamples == 10_000
    assert statistics.confidence_level == 0.95
    assert statistics.paired_analysis_unit == "latent_event_id"
    assert statistics.oracle_used is False


def test_every_comparison_uses_40_latent_events(
    statistics: PairedTierStatistics,
) -> None:
    """Observed tier rows must not be treated as independent units."""

    assert {
        result.paired_event_count
        for result in statistics.comparisons
    } == {40}


def test_every_comparison_preserves_latent_order(
    statistics: PairedTierStatistics,
) -> None:
    """All contrasts must use identical latent-event ordering."""

    reference = statistics.comparisons[0].latent_event_ids

    assert all(
        result.latent_event_ids == reference
        for result in statistics.comparisons
    )


def test_difference_mean_matches_reported_mrr_contrast(
    experiment,
    statistics: PairedTierStatistics,
) -> None:
    """Paired mean differences must equal lower-minus-higher MRR."""

    for result in statistics.comparisons:
        lower = experiment.get(
            tier=result.lower_severity_tier,
            baseline=result.baseline,
        )
        higher = experiment.get(
            tier=result.higher_severity_tier,
            baseline=result.baseline,
        )

        expected = (
            lower.mean_reciprocal_rank
            - higher.mean_reciprocal_rank
        )

        assert result.mean_mrr_difference == pytest.approx(
            expected,
            abs=1e-15,
        )

        assert result.mean_mrr_difference == pytest.approx(
            np.mean(result.reciprocal_rank_differences),
            abs=1e-15,
        )


def test_outcome_counts_sum_to_paired_count(
    statistics: PairedTierStatistics,
) -> None:
    """Better, tied, and worse counts must partition latent units."""

    for result in statistics.comparisons:
        assert (
            result.improved_count
            + result.tied_count
            + result.worsened_count
            == result.paired_event_count
        )


def test_all_statistics_are_finite(
    statistics: PairedTierStatistics,
) -> None:
    """No bootstrap result may contain NaN or infinity."""

    for result in statistics.comparisons:
        values = (
            result.mean_mrr_difference,
            result.standard_deviation_difference,
            result.bootstrap_ci_lower,
            result.bootstrap_ci_upper,
            *result.reciprocal_rank_differences,
        )

        assert all(isfinite(value) for value in values)


def test_confidence_bounds_are_ordered(
    statistics: PairedTierStatistics,
) -> None:
    """Every bootstrap lower bound must not exceed its upper bound."""

    assert all(
        result.bootstrap_ci_lower
        <= result.bootstrap_ci_upper
        for result in statistics.comparisons
    )


def test_random_control_has_zero_differences(
    statistics: PairedTierStatistics,
) -> None:
    """Paired random rankings must yield exactly zero tier effects."""

    random_results = [
        result
        for result in statistics.comparisons
        if result.baseline is BaselineKind.RANDOM
    ]

    assert len(random_results) == 3

    for result in random_results:
        assert result.mean_mrr_difference == 0.0
        assert result.standard_deviation_difference == 0.0
        assert result.bootstrap_ci_lower == 0.0
        assert result.bootstrap_ci_upper == 0.0
        assert result.improved_count == 0
        assert result.tied_count == 40
        assert result.worsened_count == 0
        assert set(result.reciprocal_rank_differences) == {0.0}


def test_contrast_names_are_explicit(
    statistics: PairedTierStatistics,
) -> None:
    """Every label must identify the lower-minus-higher direction."""

    assert {
        result.contrast_name
        for result in statistics.comparisons
    } == {
        "mild_minus_moderate",
        "moderate_minus_severe",
        "mild_minus_severe",
    }


def test_statistics_are_deterministic(
    experiment,
    statistics: PairedTierStatistics,
) -> None:
    """Repeated bootstrap execution must return identical output."""

    repeated = compute_paired_tier_statistics(
        experiment
    )

    assert repeated == statistics


def test_statistics_are_immutable(
    statistics: PairedTierStatistics,
) -> None:
    """Completed statistical metadata cannot be modified."""

    with pytest.raises(FrozenInstanceError):
        statistics.oracle_used = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "invalid_seed",
    (
        0,
        1,
        4243,
        -1,
        True,
    ),
)
def test_nonlocked_bootstrap_seed_is_rejected(
    experiment,
    invalid_seed,
) -> None:
    """The registered bootstrap seed cannot change silently."""

    with pytest.raises(
        PairedTierStatisticsError,
        match="locked at 4242",
    ):
        compute_paired_tier_statistics(
            experiment,
            bootstrap_seed=invalid_seed,
        )


@pytest.mark.parametrize(
    "invalid_resamples",
    (
        9999,
        10001,
        1000,
        True,
    ),
)
def test_nonlocked_resample_count_is_rejected(
    experiment,
    invalid_resamples,
) -> None:
    """The registered number of resamples must remain 10000."""

    with pytest.raises(
        PairedTierStatisticsError,
        match="locked at 10000",
    ):
        compute_paired_tier_statistics(
            experiment,
            bootstrap_resamples=invalid_resamples,
        )


@pytest.mark.parametrize(
    "invalid_confidence",
    (
        0.90,
        0.99,
        95,
        True,
    ),
)
def test_nonlocked_confidence_level_is_rejected(
    experiment,
    invalid_confidence,
) -> None:
    """The registered confidence level must remain 0.95."""

    with pytest.raises(
        PairedTierStatisticsError,
        match="locked at 0.95",
    ):
        compute_paired_tier_statistics(
            experiment,
            confidence_level=invalid_confidence,
        )


def test_invalid_experiment_type_is_rejected() -> None:
    """Statistics require a completed graded-OOD experiment."""

    with pytest.raises(
        PairedTierStatisticsError,
        match="GradedOODExperiment",
    ):
        compute_paired_tier_statistics(
            "invalid"  # type: ignore[arg-type]
        )


def test_get_returns_requested_comparison(
    statistics: PairedTierStatistics,
) -> None:
    """Registered comparisons must be retrievable by exact key."""

    result = statistics.get(
        baseline=BaselineKind.RIDGE_FUSION,
        lower_severity_tier=OODTier.MILD,
        higher_severity_tier=OODTier.SEVERE,
    )

    assert result.baseline is BaselineKind.RIDGE_FUSION
    assert result.lower_severity_tier is OODTier.MILD
    assert result.higher_severity_tier is OODTier.SEVERE


def test_get_rejects_unregistered_direction(
    statistics: PairedTierStatistics,
) -> None:
    """Reverse or post-hoc contrasts must be rejected."""

    with pytest.raises(
        PairedTierStatisticsError,
        match="not preregistered",
    ):
        statistics.get(
            baseline=BaselineKind.RIDGE_FUSION,
            lower_severity_tier=OODTier.SEVERE,
            higher_severity_tier=OODTier.MILD,
        )


def test_constants_match_protocol_amendment() -> None:
    """Statistical constants must match the locked amendment."""

    assert LOCKED_BOOTSTRAP_SEED == 4242
    assert LOCKED_BOOTSTRAP_RESAMPLES == 10_000
    assert LOCKED_CONFIDENCE_LEVEL == 0.95
