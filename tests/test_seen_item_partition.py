"""Tests for the prespecified seen-item template partition."""

import pytest

from src.evaluation.seen_item_partition import (
    create_seen_item_partition,
    load_seen_item_partition_config,
)
from src.evaluation.synthetic_generator import (
    generate_synthetic_pilot,
)
from src.evaluation.synthetic_records import SplitLabel


CONFIG_PATH = "configs/seen_item_evaluation_v0.2.yaml"
SYNTHETIC_CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"


@pytest.fixture(scope="module")
def dataset():
    return generate_synthetic_pilot(
        SYNTHETIC_CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=10000,
    )


@pytest.fixture(scope="module")
def partition_config():
    return load_seen_item_partition_config(
        CONFIG_PATH
    )


@pytest.fixture(scope="module")
def partition(dataset, partition_config):
    return create_seen_item_partition(
        dataset,
        partition_config,
    )


def test_validation_templates_are_partitioned_by_group(
    dataset,
    partition,
) -> None:
    validation_templates = {
        event.template_id
        for event in dataset.events
        if event.split is SplitLabel.VALIDATION
    }

    calibration = set(
        partition.calibration_template_ids
    )
    seen_item_test = set(
        partition.seen_item_test_template_ids
    )

    assert len(validation_templates) == 10
    assert len(calibration) == 5
    assert len(seen_item_test) == 5

    assert calibration.isdisjoint(seen_item_test)
    assert calibration | seen_item_test == validation_templates


def test_calibration_and_test_events_are_disjoint(
    partition,
) -> None:
    calibration_ids = {
        event.event_id
        for event in partition.calibration_events
    }
    test_ids = {
        event.event_id
        for event in partition.seen_item_test_events
    }

    assert calibration_ids
    assert test_ids
    assert calibration_ids.isdisjoint(test_ids)


def test_every_final_test_target_is_reachable(
    partition,
) -> None:
    training_targets = set(
        partition.training_target_ids
    )

    assert training_targets

    assert all(
        event.target_item_id in training_targets
        for event in partition.seen_item_test_events
    )


def test_unreachable_test_events_are_recorded(
    partition,
) -> None:
    retained_ids = {
        event.event_id
        for event in partition.seen_item_test_events
    }
    excluded_ids = set(
        partition.unreachable_test_event_ids
    )

    assert retained_ids.isdisjoint(excluded_ids)

    assert (
        len(retained_ids)
        + len(excluded_ids)
        == partition.raw_seen_item_test_event_count
    )


def test_fit_dataset_contains_no_final_test_events(
    partition,
) -> None:
    fit_dataset = partition.fit_dataset

    fit_ids = {
        event.event_id
        for event in fit_dataset.events
    }
    final_test_ids = {
        event.event_id
        for event in partition.seen_item_test_events
    }

    assert fit_ids.isdisjoint(final_test_ids)

    fit_validation_ids = {
        event.event_id
        for event in fit_dataset.events
        if event.split is SplitLabel.VALIDATION
    }
    calibration_ids = {
        event.event_id
        for event in partition.calibration_events
    }

    assert fit_validation_ids == calibration_ids

    assert all(
        event.split
        in {
            SplitLabel.TRAIN,
            SplitLabel.VALIDATION,
        }
        for event in fit_dataset.events
    )


def test_partition_is_deterministic(
    dataset,
    partition_config,
    partition,
) -> None:
    repeated = create_seen_item_partition(
        dataset,
        partition_config,
    )

    assert repeated == partition
