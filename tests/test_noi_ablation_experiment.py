"""Tests for the paired graded-OOD NOI ablation experiment."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from src.evaluation.graded_ood import OODTier
from src.evaluation.noi_ablation_experiment import (
    LOCKED_TEMPORAL_DISPLACEMENTS,
    LOCKED_TOP_K,
    NOIAblationExperiment,
    NOIAblationExperimentError,
    NOIAblationSystem,
    run_noi_ablation_experiment,
)
from src.evaluation.paired_ood_source_generator import (
    generate_paired_graded_ood_bundle,
)
from src.safety.policy_gate import load_policy_rules
from src.system.noi_pipeline import (
    load_noi_system_configuration,
)


PROTOCOL_HASH = (
    "e885f537b3209d7052d1517efc5be753"
    "24df39689cb4e54b38a241c95f22e512"
)

TRAINED_AT = datetime(
    2026,
    8,
    22,
    19,
    5,
    tzinfo=timezone.utc,
)


@pytest.fixture(scope="module")
def bundle():
    """Create the replay-verified paired graded-OOD bundle."""

    return generate_paired_graded_ood_bundle(
        "configs/synthetic_data.yaml",
        "configs/research_protocol.yaml",
        "configs/protocol_amendment_v0.2.yaml",
        "configs/graded_ood_generation.yaml",
        event_count=200,
    )


@pytest.fixture(scope="module")
def system_configuration():
    """Load the preregistered integrated-system definition."""

    return load_noi_system_configuration(
        "configs/noi_system_v0.1.yaml"
    )


@pytest.fixture(scope="module")
def policy_configuration():
    """Load the locked simulated policy rules."""

    return load_policy_rules(
        "configs/policy_rules.yaml"
    )


@pytest.fixture(scope="module")
def experiment(
    bundle,
    system_configuration,
    policy_configuration,
) -> NOIAblationExperiment:
    """Run the locked ablation once."""

    return run_noi_ablation_experiment(
        bundle,
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=PROTOCOL_HASH,
        trained_at_utc=TRAINED_AT,
    )


def test_all_locked_conditions_are_present(
    experiment: NOIAblationExperiment,
) -> None:
    """Four systems, three tiers, and five times give 60 cells."""

    assert len(experiment.evaluations) == 60

    observed = {
        (
            evaluation.system,
            evaluation.tier,
            evaluation.temporal_displacement_days,
        )
        for evaluation in experiment.evaluations
    }

    expected = {
        (system, tier, day)
        for system in NOIAblationSystem
        for tier in OODTier
        for day in LOCKED_TEMPORAL_DISPLACEMENTS
    }

    assert observed == expected


def test_locked_counts_are_preserved(
    experiment: NOIAblationExperiment,
) -> None:
    """The experiment must retain locked train and OOD counts."""

    assert experiment.latent_event_count == 40
    assert experiment.odor_library_size == 200
    assert experiment.training_event_count == 140
    assert experiment.validation_event_count == 20
    assert experiment.top_k == 10


def test_oracle_and_ood_tuning_are_absent(
    experiment: NOIAblationExperiment,
) -> None:
    """No OOD labels may tune the deployable system."""

    assert experiment.oracle_used is False
    assert experiment.ood_tuning_used is False
    assert experiment.paired_analysis_unit == "latent_event_id"


def test_validation_alpha_is_valid(
    experiment: NOIAblationExperiment,
) -> None:
    """The selected alpha must come from the locked candidate range."""

    assert experiment.selected_validation_alpha in (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    )


@pytest.mark.parametrize(
    ("system", "expected_alpha_source", "expected_decay"),
    (
        (
            NOIAblationSystem.RIDGE_ONLY,
            "fixed_one",
            False,
        ),
        (
            NOIAblationSystem.MEMORY_ONLY,
            "fixed_zero",
            True,
        ),
        (
            NOIAblationSystem.HYBRID_WITHOUT_TEMPORAL_DECAY,
            "validation",
            False,
        ),
        (
            NOIAblationSystem.FULL_HYBRID,
            "validation",
            True,
        ),
    ),
)
def test_system_runtime_mapping_is_locked(
    experiment: NOIAblationExperiment,
    system: NOIAblationSystem,
    expected_alpha_source: str,
    expected_decay: bool,
) -> None:
    """Every ablation must use its preregistered runtime settings."""

    cells = [
        evaluation
        for evaluation in experiment.evaluations
        if evaluation.system is system
    ]

    assert len(cells) == 15
    assert {
        evaluation.apply_temporal_decay
        for evaluation in cells
    } == {expected_decay}

    alphas = {
        evaluation.selected_alpha
        for evaluation in cells
    }

    if expected_alpha_source == "fixed_one":
        assert alphas == {1.0}
    elif expected_alpha_source == "fixed_zero":
        assert alphas == {0.0}
    else:
        assert alphas == {
            experiment.selected_validation_alpha
        }


def test_each_cell_contains_same_forty_latent_events(
    experiment: NOIAblationExperiment,
) -> None:
    """Every comparison must use the same paired analysis units."""

    reference = experiment.evaluations[0].latent_event_ids

    assert len(reference) == 40
    assert len(set(reference)) == 40

    for evaluation in experiment.evaluations:
        assert evaluation.latent_event_ids == reference
        assert len(evaluation.rankings) == 40
        assert len(evaluation.relevant_items) == 40


def test_metrics_are_in_valid_ranges(
    experiment: NOIAblationExperiment,
) -> None:
    """Every reported retrieval metric must be bounded."""

    for evaluation in experiment.evaluations:
        assert 0.0 <= evaluation.recall_at_1 <= 1.0
        assert 0.0 <= evaluation.recall_at_10 <= 1.0
        assert (
            0.0
            <= evaluation.mean_reciprocal_rank
            <= 1.0
        )
        assert 0.0 <= evaluation.ndcg_at_10 <= 1.0


@pytest.mark.parametrize(
    "tier",
    tuple(OODTier),
)
def test_ridge_only_is_time_invariant(
    experiment: NOIAblationExperiment,
    tier: OODTier,
) -> None:
    """The ridge-only ablation cannot depend on memory age."""

    reference = experiment.get(
        system=NOIAblationSystem.RIDGE_ONLY,
        tier=tier,
        temporal_displacement_days=0,
    )

    for day in LOCKED_TEMPORAL_DISPLACEMENTS:
        comparison = experiment.get(
            system=NOIAblationSystem.RIDGE_ONLY,
            tier=tier,
            temporal_displacement_days=day,
        )

        assert comparison.rankings == reference.rankings
        assert (
            comparison.mean_reciprocal_rank
            == reference.mean_reciprocal_rank
        )


@pytest.mark.parametrize(
    "tier",
    tuple(OODTier),
)
def test_no_decay_hybrid_is_time_invariant(
    experiment: NOIAblationExperiment,
    tier: OODTier,
) -> None:
    """The no-decay hybrid must not change with timestamp."""

    reference = experiment.get(
        system=(
            NOIAblationSystem
            .HYBRID_WITHOUT_TEMPORAL_DECAY
        ),
        tier=tier,
        temporal_displacement_days=0,
    )

    for day in LOCKED_TEMPORAL_DISPLACEMENTS:
        comparison = experiment.get(
            system=(
                NOIAblationSystem
                .HYBRID_WITHOUT_TEMPORAL_DECAY
            ),
            tier=tier,
            temporal_displacement_days=day,
        )

        assert comparison.rankings == reference.rankings


def test_alpha_one_makes_full_hybrid_equal_ridge(
    experiment: NOIAblationExperiment,
) -> None:
    """If validation selects one, memory has zero hybrid weight."""

    if experiment.selected_validation_alpha != 1.0:
        pytest.skip("Validation did not select alpha one.")

    for tier in OODTier:
        for day in LOCKED_TEMPORAL_DISPLACEMENTS:
            ridge = experiment.get(
                system=NOIAblationSystem.RIDGE_ONLY,
                tier=tier,
                temporal_displacement_days=day,
            )
            full = experiment.get(
                system=NOIAblationSystem.FULL_HYBRID,
                tier=tier,
                temporal_displacement_days=day,
            )

            assert full.rankings == ridge.rankings
            assert (
                full.mean_reciprocal_rank
                == ridge.mean_reciprocal_rank
            )


def test_get_returns_exact_condition(
    experiment: NOIAblationExperiment,
) -> None:
    """Condition lookup must not silently return another cell."""

    evaluation = experiment.get(
        system=NOIAblationSystem.FULL_HYBRID,
        tier=OODTier.SEVERE,
        temporal_displacement_days=90,
    )

    assert evaluation.system is NOIAblationSystem.FULL_HYBRID
    assert evaluation.tier is OODTier.SEVERE
    assert evaluation.temporal_displacement_days == 90


def test_experiment_is_immutable(
    experiment: NOIAblationExperiment,
) -> None:
    """Completed experimental output must be immutable."""

    with pytest.raises(FrozenInstanceError):
        experiment.oracle_used = True  # type: ignore[misc]


def test_invalid_bundle_type_is_rejected(
    system_configuration,
    policy_configuration,
) -> None:
    """The experiment requires a validated paired bundle."""

    with pytest.raises(
        NOIAblationExperimentError,
        match="PairedGradedOODBundle",
    ):
        run_noi_ablation_experiment(
            "invalid",  # type: ignore[arg-type]
            system_configuration=system_configuration,
            policy_configuration=policy_configuration,
            protocol_hash=PROTOCOL_HASH,
            trained_at_utc=TRAINED_AT,
        )


def test_naive_training_time_is_rejected(
    bundle,
    system_configuration,
    policy_configuration,
) -> None:
    """Training time must be timezone-aware."""

    with pytest.raises(
        NOIAblationExperimentError,
        match="timezone-aware",
    ):
        run_noi_ablation_experiment(
            bundle,
            system_configuration=system_configuration,
            policy_configuration=policy_configuration,
            protocol_hash=PROTOCOL_HASH,
            trained_at_utc=datetime(2026, 8, 22),
        )


@pytest.mark.parametrize(
    "invalid_top_k",
    (1, 9, 11, True),
)
def test_modified_top_k_is_rejected(
    bundle,
    system_configuration,
    policy_configuration,
    invalid_top_k,
) -> None:
    """The locked top-k value cannot change silently."""

    with pytest.raises(
        NOIAblationExperimentError,
        match="locked value 10",
    ):
        run_noi_ablation_experiment(
            bundle,
            system_configuration=system_configuration,
            policy_configuration=policy_configuration,
            protocol_hash=PROTOCOL_HASH,
            trained_at_utc=TRAINED_AT,
            top_k=invalid_top_k,
        )


def test_experiment_is_deterministic(
    bundle,
    system_configuration,
    policy_configuration,
    experiment: NOIAblationExperiment,
) -> None:
    """Repeating the full ablation must produce identical output."""

    repeated = run_noi_ablation_experiment(
        bundle,
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=PROTOCOL_HASH,
        trained_at_utc=TRAINED_AT,
    )

    assert repeated == experiment
