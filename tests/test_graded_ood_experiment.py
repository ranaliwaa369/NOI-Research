"""Tests for paired graded-OOD baseline evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.baselines.retrieval_baselines import BaselineKind
from src.evaluation.graded_ood import OODTier
from src.evaluation.graded_ood_experiment import (
    LOCKED_TOP_K,
    GradedOODExperiment,
    GradedOODExperimentError,
    run_graded_ood_experiment,
)
from src.evaluation.paired_ood_source_generator import (
    generate_paired_graded_ood_bundle,
)
from src.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


@pytest.fixture(scope="module")
def bundle():
    """Generate the replay-verified paired pilot."""

    return generate_paired_graded_ood_bundle(
        "configs/synthetic_data.yaml",
        "configs/research_protocol.yaml",
        "configs/protocol_amendment_v0.2.yaml",
        "configs/graded_ood_generation.yaml",
        event_count=200,
    )


@pytest.fixture(scope="module")
def experiment(bundle) -> GradedOODExperiment:
    """Run the complete locked graded-OOD experiment."""

    return run_graded_ood_experiment(bundle)


def test_experiment_contains_all_tier_baseline_pairs(
    experiment: GradedOODExperiment,
) -> None:
    """Four baselines by three tiers must yield twelve results."""

    assert len(experiment.evaluations) == 12

    assert {
        (evaluation.tier, evaluation.baseline)
        for evaluation in experiment.evaluations
    } == {
        (tier, baseline)
        for tier in OODTier
        for baseline in BaselineKind
    }


def test_experiment_uses_paired_latent_units(
    experiment: GradedOODExperiment,
) -> None:
    """The experiment must use 40 independent latent units."""

    assert experiment.latent_event_count == 40
    assert experiment.paired_analysis_unit == "latent_event_id"

    assert all(
        evaluation.event_count == 40
        for evaluation in experiment.evaluations
    )


def test_experiment_uses_locked_library_and_cutoff(
    experiment: GradedOODExperiment,
) -> None:
    """The fixed library has 200 items and top-k remains ten."""

    assert experiment.odor_library_size == 200
    assert experiment.top_k == LOCKED_TOP_K == 10


def test_oracle_is_not_used(
    experiment: GradedOODExperiment,
) -> None:
    """OOD labels cannot calibrate any baseline in this experiment."""

    assert experiment.oracle_used is False


def test_training_count_is_locked(
    experiment: GradedOODExperiment,
) -> None:
    """Every evaluation must report the original 140 training events."""

    assert {
        evaluation.training_event_count
        for evaluation in experiment.evaluations
    } == {140}


def test_every_ranking_has_ten_unique_items(
    experiment: GradedOODExperiment,
) -> None:
    """Every result must contain a valid ten-item ranking."""

    for evaluation in experiment.evaluations:
        for ranking in evaluation.rankings:
            assert len(ranking) == 10
            assert len(set(ranking)) == 10


def test_every_event_has_one_relevant_target(
    experiment: GradedOODExperiment,
) -> None:
    """Synthetic ground truth contains one target per latent event."""

    for evaluation in experiment.evaluations:
        assert all(
            len(relevant) == 1
            for relevant in evaluation.relevant_items
        )


def test_latent_order_is_identical_across_evaluations(
    experiment: GradedOODExperiment,
) -> None:
    """Paired comparisons require identical latent-event order."""

    reference = experiment.evaluations[0].latent_event_ids

    assert all(
        evaluation.latent_event_ids == reference
        for evaluation in experiment.evaluations
    )


def test_random_rankings_are_identical_across_tiers(
    experiment: GradedOODExperiment,
) -> None:
    """Random rankings must not vary merely because tier labels differ."""

    rankings = tuple(
        experiment.get(
            tier=tier,
            baseline=BaselineKind.RANDOM,
        ).rankings
        for tier in OODTier
    )

    assert rankings[0] == rankings[1] == rankings[2]


def test_random_metrics_are_identical_across_tiers(
    experiment: GradedOODExperiment,
) -> None:
    """The paired random control must be severity invariant."""

    evaluations = tuple(
        experiment.get(
            tier=tier,
            baseline=BaselineKind.RANDOM,
        )
        for tier in OODTier
    )

    metrics = {
        (
            evaluation.recall_at_1,
            evaluation.recall_at_10,
            evaluation.mean_reciprocal_rank,
            evaluation.ndcg_at_10,
        )
        for evaluation in evaluations
    }

    assert len(metrics) == 1


def test_reported_metrics_recompute_exactly(
    experiment: GradedOODExperiment,
) -> None:
    """Stored metrics must equal independent metric-function outputs."""

    for evaluation in experiment.evaluations:
        assert evaluation.recall_at_1 == recall_at_k(
            evaluation.rankings,
            evaluation.relevant_items,
            k=1,
        )
        assert evaluation.recall_at_10 == recall_at_k(
            evaluation.rankings,
            evaluation.relevant_items,
            k=10,
        )
        assert (
            evaluation.mean_reciprocal_rank
            == mean_reciprocal_rank(
                evaluation.rankings,
                evaluation.relevant_items,
            )
        )
        assert evaluation.ndcg_at_10 == ndcg_at_k(
            evaluation.rankings,
            evaluation.relevant_items,
            k=10,
        )


def test_all_metrics_are_in_valid_range(
    experiment: GradedOODExperiment,
) -> None:
    """No reported metric may fall outside [0, 1]."""

    for evaluation in experiment.evaluations:
        metrics = (
            evaluation.recall_at_1,
            evaluation.recall_at_10,
            evaluation.mean_reciprocal_rank,
            evaluation.ndcg_at_10,
        )

        assert all(
            0.0 <= metric <= 1.0
            for metric in metrics
        )
        assert (
            evaluation.recall_at_1
            <= evaluation.recall_at_10
        )


def test_experiment_is_deterministic(
    bundle,
    experiment: GradedOODExperiment,
) -> None:
    """Repeated execution must produce an identical result."""

    repeated = run_graded_ood_experiment(bundle)

    assert repeated == experiment


def test_experiment_is_immutable(
    experiment: GradedOODExperiment,
) -> None:
    """Completed experiment metadata cannot be modified."""

    with pytest.raises(FrozenInstanceError):
        experiment.oracle_used = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "invalid_top_k",
    (
        1,
        9,
        11,
        0,
        True,
        10.0,
    ),
)
def test_nonlocked_top_k_is_rejected(
    bundle,
    invalid_top_k,
) -> None:
    """The preregistered cutoff must remain exactly ten."""

    with pytest.raises(
        GradedOODExperimentError,
        match="locked at 10",
    ):
        run_graded_ood_experiment(
            bundle,
            top_k=invalid_top_k,
        )


@pytest.mark.parametrize(
    "invalid_seed",
    (
        -1,
        True,
        2026.0,
    ),
)
def test_invalid_random_seed_is_rejected(
    bundle,
    invalid_seed,
) -> None:
    """Random control seeds must be nonnegative integers."""

    with pytest.raises(
        GradedOODExperimentError,
        match="random_seed",
    ):
        run_graded_ood_experiment(
            bundle,
            random_seed=invalid_seed,
        )


@pytest.mark.parametrize(
    "invalid_alpha",
    (
        -1.0,
        float("inf"),
        float("nan"),
        True,
    ),
)
def test_invalid_ridge_alpha_is_rejected(
    bundle,
    invalid_alpha,
) -> None:
    """Ridge regularization must be finite and nonnegative."""

    with pytest.raises(
        GradedOODExperimentError,
        match="ridge_alpha",
    ):
        run_graded_ood_experiment(
            bundle,
            ridge_alpha=invalid_alpha,
        )


def test_invalid_bundle_type_is_rejected() -> None:
    """The runner requires a replay-verified bundle."""

    with pytest.raises(
        GradedOODExperimentError,
        match="PairedGradedOODBundle",
    ):
        run_graded_ood_experiment(
            "invalid"  # type: ignore[arg-type]
        )


def test_get_rejects_invalid_tier(
    experiment: GradedOODExperiment,
) -> None:
    """Tier lookup requires the registered enum."""

    with pytest.raises(
        GradedOODExperimentError,
        match="OODTier",
    ):
        experiment.get(
            tier="mild",  # type: ignore[arg-type]
            baseline=BaselineKind.RANDOM,
        )


def test_get_rejects_invalid_baseline(
    experiment: GradedOODExperiment,
) -> None:
    """Baseline lookup requires the registered enum."""

    with pytest.raises(
        GradedOODExperimentError,
        match="BaselineKind",
    ):
        experiment.get(
            tier=OODTier.MILD,
            baseline="ridge",  # type: ignore[arg-type]
        )
