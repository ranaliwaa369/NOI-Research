"""Tests for the paired corrective-memory experiment."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from src.evaluation.corrective_memory_config import (
    load_corrective_memory_configuration,
)
from src.evaluation.corrective_memory_experiment import (
    LOCKED_BOOTSTRAP_RESAMPLES,
    LOCKED_BOOTSTRAP_SEED,
    LOCKED_CONFIDENCE_LEVEL,
    LOCKED_EXPECTED_ELIGIBLE_EVENTS,
    LOCKED_EXPECTED_ELIGIBLE_TARGETS,
    LOCKED_MAXIMUM_OLD_MEMORY_DEGRADATION,
    LOCKED_MINIMUM_MRR_IMPROVEMENT,
    CorrectiveMemoryExperiment,
    CorrectiveMemoryExperimentError,
    run_corrective_memory_experiment,
)
from src.evaluation.synthetic_generator import (
    generate_synthetic_pilot,
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
def dataset():
    return generate_synthetic_pilot(
        "configs/synthetic_data.yaml",
        "configs/research_protocol.yaml",
        event_count=200,
    )


@pytest.fixture(scope="module")
def evaluation_configuration():
    return load_corrective_memory_configuration(
        "configs/corrective_memory_evaluation_v0.1.yaml",
        "configs/corrective_memory_evaluation_v0.1.sha256",
    )


@pytest.fixture(scope="module")
def system_configuration():
    return load_noi_system_configuration(
        "configs/noi_system_v0.1.yaml"
    )


@pytest.fixture(scope="module")
def policy_configuration():
    return load_policy_rules(
        "configs/policy_rules.yaml"
    )


@pytest.fixture(scope="module")
def experiment(
    dataset,
    evaluation_configuration,
    system_configuration,
    policy_configuration,
) -> CorrectiveMemoryExperiment:
    return run_corrective_memory_experiment(
        dataset,
        evaluation_configuration=(
            evaluation_configuration
        ),
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=PROTOCOL_HASH,
        trained_at_utc=TRAINED_AT,
    )


def test_locked_counts_are_preserved(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    assert experiment.training_event_count == 140
    assert experiment.validation_event_count == 20
    assert (
        experiment.eligible_target_count
        == LOCKED_EXPECTED_ELIGIBLE_TARGETS
    )
    assert (
        experiment.eligible_validation_event_count
        == LOCKED_EXPECTED_ELIGIBLE_EVENTS
    )
    assert len(experiment.target_results) == 14


def test_eligible_identifiers_are_unique_and_sorted(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    assert experiment.eligible_target_ids == tuple(
        sorted(experiment.eligible_target_ids)
    )
    assert len(set(experiment.eligible_target_ids)) == 14
    assert (
        len(
            set(
                experiment.eligible_validation_event_ids
            )
        )
        == 15
    )


def test_unknown_validation_targets_are_excluded(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    assert len(
        experiment.excluded_validation_target_ids
    ) == 5
    assert set(
        experiment.excluded_validation_target_ids
    ).isdisjoint(
        experiment.eligible_target_ids
    )


def test_primary_runtime_is_memory_only(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    assert experiment.alpha == 0.0
    assert experiment.apply_temporal_decay is False
    assert experiment.top_k == 10


def test_no_oracle_or_ood_tuning(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    assert experiment.oracle_used is False
    assert experiment.ood_tuning_used is False


def test_bootstrap_settings_are_locked(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    assert (
        experiment.bootstrap_seed
        == LOCKED_BOOTSTRAP_SEED
    )
    assert (
        experiment.bootstrap_resamples
        == LOCKED_BOOTSTRAP_RESAMPLES
    )
    assert (
        experiment.confidence_level
        == LOCKED_CONFIDENCE_LEVEL
    )


def test_every_target_uses_a_different_decoy(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    for result in experiment.target_results:
        assert result.decoy_item_id != result.target_item_id


def test_every_corrupted_memory_has_restoration_audit(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    for result in experiment.target_results:
        assert result.corrupted_memory_ids
        assert (
            result.restoration_audit_count
            == len(result.corrupted_memory_ids)
        )


def test_query_arrays_are_paired(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    for result in experiment.target_results:
        count = len(result.validation_event_ids)

        assert len(result.no_update_rankings) == count
        assert len(result.corrected_rankings) == count
        assert len(result.relevant_items) == count


def test_target_differences_are_computed_exactly(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    for result in experiment.target_results:
        assert (
            result.reciprocal_rank_difference
            == pytest.approx(
                result.corrected_mean_reciprocal_rank
                - result.no_update_mean_reciprocal_rank,
                abs=1e-15,
            )
        )


def test_old_memory_degradation_is_computed_exactly(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    for result in experiment.target_results:
        assert result.old_memory_degradation == pytest.approx(
            result.old_memory_baseline_mrr
            - result.old_memory_post_correction_mrr,
            abs=1e-15,
        )


def test_metrics_are_finite_and_bounded(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    for result in experiment.target_results:
        for value in (
            result.no_update_mean_reciprocal_rank,
            result.corrected_mean_reciprocal_rank,
            result.no_update_recall_at_1,
            result.corrected_recall_at_1,
            result.no_update_recall_at_10,
            result.corrected_recall_at_10,
            result.no_update_ndcg_at_10,
            result.corrected_ndcg_at_10,
            result.old_memory_baseline_mrr,
            result.old_memory_post_correction_mrr,
        ):
            assert 0.0 <= value <= 1.0


def test_bootstrap_interval_contains_mean(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    assert (
        experiment.bootstrap_ci_lower
        <= experiment.mean_mrr_improvement
        <= experiment.bootstrap_ci_upper
    )


def test_correction_success_rule_is_computed(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    expected = (
        experiment.mean_mrr_improvement
        >= LOCKED_MINIMUM_MRR_IMPROVEMENT
        and experiment.bootstrap_ci_lower > 0.0
    )

    assert (
        experiment.correction_success_rule_passed
        is expected
    )


def test_old_memory_rule_is_computed(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    expected = (
        experiment.mean_old_memory_degradation
        <= LOCKED_MAXIMUM_OLD_MEMORY_DEGRADATION
    )

    assert (
        experiment.old_memory_degradation_rule_passed
        is expected
    )


def test_experiment_is_immutable(
    experiment: CorrectiveMemoryExperiment,
) -> None:
    with pytest.raises(FrozenInstanceError):
        experiment.oracle_used = True  # type: ignore[misc]


def test_invalid_dataset_type_is_rejected(
    evaluation_configuration,
    system_configuration,
    policy_configuration,
) -> None:
    with pytest.raises(
        CorrectiveMemoryExperimentError,
        match="SyntheticDataset",
    ):
        run_corrective_memory_experiment(
            "invalid",  # type: ignore[arg-type]
            evaluation_configuration=(
                evaluation_configuration
            ),
            system_configuration=system_configuration,
            policy_configuration=policy_configuration,
            protocol_hash=PROTOCOL_HASH,
            trained_at_utc=TRAINED_AT,
        )


def test_naive_training_time_is_rejected(
    dataset,
    evaluation_configuration,
    system_configuration,
    policy_configuration,
) -> None:
    with pytest.raises(
        CorrectiveMemoryExperimentError,
        match="timezone-aware",
    ):
        run_corrective_memory_experiment(
            dataset,
            evaluation_configuration=(
                evaluation_configuration
            ),
            system_configuration=system_configuration,
            policy_configuration=policy_configuration,
            protocol_hash=PROTOCOL_HASH,
            trained_at_utc=datetime(2026, 8, 22),
        )


@pytest.mark.parametrize(
    ("section", "key", "invalid_value"),
    (
        (
            "eligibility",
            "expected_eligible_targets",
            13,
        ),
        (
            "eligibility",
            "expected_eligible_validation_events",
            14,
        ),
        (
            "retrieval",
            "alpha",
            0.5,
        ),
        (
            "retrieval",
            "apply_temporal_decay",
            True,
        ),
        (
            "retrieval",
            "top_k",
            5,
        ),
        (
            "statistics",
            "bootstrap_seed",
            1234,
        ),
        (
            "statistics",
            "bootstrap_resamples",
            9999,
        ),
        (
            "statistics",
            "confidence_level",
            0.90,
        ),
    ),
)
def test_modified_runtime_setting_is_rejected(
    dataset,
    evaluation_configuration,
    system_configuration,
    policy_configuration,
    section: str,
    key: str,
    invalid_value,
) -> None:
    modified = {
        name: (
            dict(value)
            if isinstance(value, dict)
            else value
        )
        for name, value in evaluation_configuration.items()
    }
    modified[section] = dict(
        evaluation_configuration[section]
    )
    modified[section][key] = invalid_value

    with pytest.raises(
        CorrectiveMemoryExperimentError,
    ):
        run_corrective_memory_experiment(
            dataset,
            evaluation_configuration=modified,
            system_configuration=system_configuration,
            policy_configuration=policy_configuration,
            protocol_hash=PROTOCOL_HASH,
            trained_at_utc=TRAINED_AT,
        )


def test_experiment_is_deterministic(
    dataset,
    evaluation_configuration,
    system_configuration,
    policy_configuration,
    experiment: CorrectiveMemoryExperiment,
) -> None:
    repeated = run_corrective_memory_experiment(
        dataset,
        evaluation_configuration=(
            evaluation_configuration
        ),
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=PROTOCOL_HASH,
        trained_at_utc=TRAINED_AT,
    )

    assert repeated == experiment
