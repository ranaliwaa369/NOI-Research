"""Tests for the final NOI v0.2 seen-item experiment."""

from datetime import datetime, timezone

import pytest

from src.evaluation.seen_item_final_experiment import (
    run_seen_item_final_experiment,
)
from src.evaluation.seen_item_memory_experiment import (
    SeenItemSystem,
)
from src.evaluation.seen_item_partition import (
    load_seen_item_partition_config,
)
from src.evaluation.synthetic_generator import (
    generate_synthetic_pilot,
)
from src.safety.policy_gate import load_policy_rules
from src.system.noi_pipeline import (
    load_noi_system_configuration,
)


TRAINED_AT = datetime(
    2026,
    8,
    22,
    19,
    5,
    tzinfo=timezone.utc,
)

PROTOCOL_HASH = (
    "e885f537b3209d7052d1517efc5be753"
    "24df39689cb4e54b38a241c95f22e512"
)


@pytest.fixture(scope="module")
def final_experiment():
    dataset = generate_synthetic_pilot(
        "configs/synthetic_data.yaml",
        "configs/research_protocol.yaml",
        event_count=10000,
    )

    return run_seen_item_final_experiment(
        dataset=dataset,
        partition_config=(
            load_seen_item_partition_config(
                "configs/seen_item_evaluation_v0.2.yaml"
            )
        ),
        system_configuration=(
            load_noi_system_configuration(
                "configs/noi_system_v0.1.yaml"
            )
        ),
        policy_configuration=load_policy_rules(
            "configs/policy_rules.yaml"
        ),
        protocol_hash=PROTOCOL_HASH,
        trained_at_utc=TRAINED_AT,
    )


def test_locked_partition_counts(final_experiment) -> None:
    assert final_experiment.training_event_count == 7000
    assert final_experiment.calibration_event_count == 493
    assert final_experiment.final_test_event_count == 507
    assert final_experiment.reachable_event_fraction == 1.0


def test_final_templates_are_not_used_for_calibration(
    final_experiment,
) -> None:
    calibration = set(
        final_experiment.calibration_template_ids
    )
    final_test = set(
        final_experiment.final_test_template_ids
    )

    assert len(calibration) == 5
    assert len(final_test) == 5
    assert calibration.isdisjoint(final_test)


def test_all_systems_use_identical_final_events(
    final_experiment,
) -> None:
    expected_event_ids = (
        final_experiment.final_test_event_ids
    )

    assert len(expected_event_ids) == 507

    assert {
        evaluation.system
        for evaluation in final_experiment.evaluations
    } == set(SeenItemSystem)

    for evaluation in final_experiment.evaluations:
        assert evaluation.event_ids == expected_event_ids
        assert len(evaluation.rankings) == 507
        assert len(evaluation.relevant_items) == 507


def test_system_alphas_are_prespecified(
    final_experiment,
) -> None:
    memory = final_experiment.for_system(
        SeenItemSystem.MEMORY_ONLY
    )
    ridge = final_experiment.for_system(
        SeenItemSystem.RIDGE_ONLY
    )
    hybrid = final_experiment.for_system(
        SeenItemSystem.HYBRID
    )

    assert memory.alpha == 0.0
    assert ridge.alpha == 1.0
    assert hybrid.alpha == (
        final_experiment.selected_hybrid_alpha
    )


def test_final_results_record_no_oracle_or_test_tuning(
    final_experiment,
) -> None:
    assert final_experiment.oracle_used is False
    assert final_experiment.final_test_tuning_used is False
    assert final_experiment.protocol_hash == PROTOCOL_HASH


def test_final_metrics_are_valid(final_experiment) -> None:
    for evaluation in final_experiment.evaluations:
        for metric in (
            evaluation.recall_at_1,
            evaluation.recall_at_10,
            evaluation.mean_reciprocal_rank,
            evaluation.ndcg_at_10,
        ):
            assert 0.0 <= metric <= 1.0
