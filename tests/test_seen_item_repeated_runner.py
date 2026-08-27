"""Tests for one-at-a-time repeated Track A execution."""

from datetime import datetime, timezone
from unittest.mock import sentinel

import pytest

import src.evaluation.seen_item_repeated_runner as runner
from src.evaluation.seen_item_partition import (
    load_seen_item_partition_config,
)
from src.evaluation.seen_item_repeated_config import (
    RepeatedSeedSpec,
    SeenItemRepeatedConfigError,
    load_seen_item_repeated_config,
)


TRAINED_AT = datetime(
    2026,
    8,
    22,
    19,
    5,
    tzinfo=timezone.utc,
)


@pytest.fixture
def repeated_config():
    return load_seen_item_repeated_config(
        (
            "configs/"
            "seen_item_repeated_evaluation_v0.2.1.yaml"
        ),
        (
            "configs/"
            "seen_item_repeated_evaluation_v0.2.1.sha256"
        ),
    )


@pytest.fixture
def partition_config():
    return load_seen_item_partition_config(
        "configs/seen_item_evaluation_v0.2.yaml"
    )


def test_locked_seeds_are_forwarded_exactly(
    monkeypatch,
    repeated_config,
    partition_config,
) -> None:
    observed = {}

    def fake_generate(
        configuration_path,
        protocol_path,
        *,
        event_count,
        generator_seed,
        ood_seed,
    ):
        observed["generation"] = {
            "configuration_path": configuration_path,
            "protocol_path": protocol_path,
            "event_count": event_count,
            "generator_seed": generator_seed,
            "ood_seed": ood_seed,
        }
        return sentinel.dataset

    def fake_final_experiment(**kwargs):
        observed["final"] = kwargs
        return sentinel.experiment

    monkeypatch.setattr(
        runner,
        "generate_synthetic_pilot_with_seeds",
        fake_generate,
    )
    monkeypatch.setattr(
        runner,
        "run_seen_item_final_experiment",
        fake_final_experiment,
    )

    run_spec = repeated_config.runs[0]

    result = runner.run_one_repeated_seed(
        run_spec=run_spec,
        repeated_config=repeated_config,
        base_partition_config=partition_config,
        system_configuration={"system": "config"},
        policy_configuration={"policy": "config"},
        protocol_hash="protocol-hash",
        trained_at_utc=TRAINED_AT,
        synthetic_configuration_path=(
            "configs/synthetic_data.yaml"
        ),
        research_protocol_path=(
            "configs/research_protocol.yaml"
        ),
    )

    assert result.run_spec == run_spec
    assert result.experiment is sentinel.experiment

    assert observed["generation"] == {
        "configuration_path": (
            "configs/synthetic_data.yaml"
        ),
        "protocol_path": (
            "configs/research_protocol.yaml"
        ),
        "event_count": 10000,
        "generator_seed": 1101,
        "ood_seed": 9101,
    }

    forwarded_partition = observed["final"][
        "partition_config"
    ]

    assert forwarded_partition.partition_seed == 2201
    assert forwarded_partition.total_event_count == 10000
    assert observed["final"]["dataset"] is sentinel.dataset


def test_unregistered_run_is_rejected(
    repeated_config,
    partition_config,
) -> None:
    unregistered = RepeatedSeedSpec(
        run_id="unregistered",
        generator_seed=9991,
        ood_seed=9992,
        partition_seed=9993,
    )

    with pytest.raises(
        SeenItemRepeatedConfigError,
        match="prespecified",
    ):
        runner.run_one_repeated_seed(
            run_spec=unregistered,
            repeated_config=repeated_config,
            base_partition_config=partition_config,
            system_configuration={},
            policy_configuration={},
            protocol_hash="protocol-hash",
            trained_at_utc=TRAINED_AT,
            synthetic_configuration_path=(
                "configs/synthetic_data.yaml"
            ),
            research_protocol_path=(
                "configs/research_protocol.yaml"
            ),
        )


def test_run_result_exposes_all_seed_values(
    monkeypatch,
    repeated_config,
    partition_config,
) -> None:
    monkeypatch.setattr(
        runner,
        "generate_synthetic_pilot_with_seeds",
        lambda *args, **kwargs: sentinel.dataset,
    )
    monkeypatch.setattr(
        runner,
        "run_seen_item_final_experiment",
        lambda **kwargs: sentinel.experiment,
    )

    run_spec = repeated_config.runs[9]

    result = runner.run_one_repeated_seed(
        run_spec=run_spec,
        repeated_config=repeated_config,
        base_partition_config=partition_config,
        system_configuration={},
        policy_configuration={},
        protocol_hash="protocol-hash",
        trained_at_utc=TRAINED_AT,
        synthetic_configuration_path=(
            "configs/synthetic_data.yaml"
        ),
        research_protocol_path=(
            "configs/research_protocol.yaml"
        ),
    )

    assert result.run_id == "seed-10"
    assert result.generator_seed == 2001
    assert result.ood_seed == 10001
    assert result.partition_seed == 3101
