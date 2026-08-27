"""Tests for seen-item episodic-memory evaluation."""

from datetime import datetime, timezone

import pytest

from src.evaluation.seen_item_memory_experiment import (
    SeenItemSystem,
    evaluate_seen_item_memory,
)
from src.evaluation.synthetic_generator import (
    generate_synthetic_pilot,
)
from src.safety.policy_gate import load_policy_rules
from src.system.noi_pipeline import (
    NOIPipeline,
    load_noi_system_configuration,
)


SYNTHETIC_CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"
SYSTEM_CONFIG_PATH = "configs/noi_system_v0.1.yaml"
POLICY_PATH = "configs/policy_rules.yaml"

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
        SYNTHETIC_CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )


@pytest.fixture(scope="module")
def fitted_pipeline(dataset) -> NOIPipeline:
    pipeline = NOIPipeline(
        system_configuration=load_noi_system_configuration(
            SYSTEM_CONFIG_PATH
        ),
        policy_configuration=load_policy_rules(POLICY_PATH),
        protocol_hash=PROTOCOL_HASH,
    )

    pipeline.fit(
        dataset,
        trained_at_utc=TRAINED_AT,
    )

    return pipeline


@pytest.fixture(scope="module")
def experiment(dataset, fitted_pipeline):
    return evaluate_seen_item_memory(
        fitted_pipeline,
        dataset,
        evaluated_at_utc=TRAINED_AT,
    )


def test_only_memory_reachable_validation_events_are_evaluated(
    experiment,
) -> None:
    assert experiment.total_validation_event_count == 20
    assert experiment.reachable_validation_event_count == 15
    assert experiment.reachable_event_fraction == 0.75

    assert len(experiment.reachable_event_ids) == 15
    assert len(set(experiment.reachable_event_ids)) == 15


def test_all_prespecified_systems_are_evaluated(experiment) -> None:
    assert {
        evaluation.system
        for evaluation in experiment.evaluations
    } == {
        SeenItemSystem.MEMORY_ONLY,
        SeenItemSystem.RIDGE_ONLY,
        SeenItemSystem.HYBRID,
    }


def test_system_alpha_values_are_correct(
    experiment,
    fitted_pipeline,
) -> None:
    memory = experiment.for_system(
        SeenItemSystem.MEMORY_ONLY
    )
    ridge = experiment.for_system(
        SeenItemSystem.RIDGE_ONLY
    )
    hybrid = experiment.for_system(
        SeenItemSystem.HYBRID
    )

    assert memory.alpha == 0.0
    assert ridge.alpha == 1.0
    assert hybrid.alpha == fitted_pipeline.selected_alpha


def test_each_system_uses_identical_reachable_events(
    experiment,
) -> None:
    for evaluation in experiment.evaluations:
        assert evaluation.event_ids == (
            experiment.reachable_event_ids
        )
        assert len(evaluation.rankings) == 15
        assert len(evaluation.relevant_items) == 15


def test_metrics_are_valid_probabilities(experiment) -> None:
    for evaluation in experiment.evaluations:
        for metric in (
            evaluation.recall_at_1,
            evaluation.recall_at_10,
            evaluation.mean_reciprocal_rank,
            evaluation.ndcg_at_10,
        ):
            assert 0.0 <= metric <= 1.0
