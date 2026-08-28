"""Tests for one final robustness seed."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from src.evaluation.final_robustness_experiment import (
    FinalRobustnessExperimentError,
    run_final_robustness_experiment,
)
from src.evaluation.paired_ood_source_generator import (
    generate_paired_graded_ood_bundle,
)
from src.evaluation.robustness_config import (
    load_robustness_configuration,
)
from src.system.noi_pipeline import (
    load_noi_system_configuration,
)


CONFIG_PATH = Path(
    "configs/"
    "robustness_evaluation_v0.2.3.yaml"
)
HASH_PATH = Path(
    "configs/"
    "robustness_evaluation_v0.2.3.sha256"
)
TRAINED_AT = datetime(
    2026,
    8,
    27,
    12,
    0,
    tzinfo=timezone.utc,
)


@pytest.fixture(scope="module")
def configuration():
    return load_robustness_configuration(
        CONFIG_PATH,
        HASH_PATH,
    )


@pytest.fixture(scope="module")
def system_configuration():
    return load_noi_system_configuration(
        "configs/noi_system_v0.1.yaml"
    )


@pytest.fixture(scope="module")
def policy_configuration():
    with Path(
        "configs/policy_rules.yaml"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def bundle():
    return generate_paired_graded_ood_bundle(
        "configs/synthetic_data.yaml",
        "configs/research_protocol.yaml",
        "configs/protocol_amendment_v0.2.yaml",
        "configs/graded_ood_generation.yaml",
        event_count=200,
        generator_seed=1101,
        ood_seed=9101,
    )


@pytest.fixture(scope="module")
def experiment(
    configuration,
    system_configuration,
    policy_configuration,
    bundle,
):
    return run_final_robustness_experiment(
        bundle=bundle,
        configuration=configuration,
        run=configuration.runs[0],
        system_configuration=(
            system_configuration
        ),
        policy_configuration=(
            policy_configuration
        ),
        protocol_hash=(
            configuration.configuration_sha256
        ),
        trained_at_utc=TRAINED_AT,
    )


def test_seed_and_count_metadata(
    experiment,
) -> None:
    assert experiment.run_id == (
        "robustness-seed-01"
    )
    assert experiment.generator_seed == 1101
    assert experiment.ood_seed == 9101
    assert experiment.training_event_count == 140
    assert experiment.validation_event_count == 20
    assert experiment.latent_event_count == 40
    assert experiment.observed_event_count == 120
    assert experiment.odor_library_size == 200


def test_complete_evaluation_grid(
    experiment,
) -> None:
    assert len(experiment.evaluations) == 144

    missing = tuple(
        item
        for item in experiment.evaluations
        if item.axis == "missing_modality"
    )
    temporal = tuple(
        item
        for item in experiment.evaluations
        if item.axis
        == "temporal_displacement"
    )

    assert len(missing) == 84
    assert len(temporal) == 60


def test_missing_modality_conditions(
    experiment,
    configuration,
) -> None:
    observed = {
        (
            item.condition_id,
            item.missing_modalities,
        )
        for item in experiment.evaluations
        if item.axis == "missing_modality"
    }
    expected = {
        (
            condition.condition_id,
            condition.missing_modalities,
        )
        for condition
        in configuration.missing_conditions
    }

    assert observed == expected

    assert {
        item.temporal_displacement_days
        for item in experiment.evaluations
        if item.axis == "missing_modality"
    } == {0}


def test_temporal_conditions(
    experiment,
    configuration,
) -> None:
    temporal = tuple(
        item
        for item in experiment.evaluations
        if item.axis
        == "temporal_displacement"
    )

    assert {
        item.temporal_displacement_days
        for item in temporal
    } == set(
        configuration.temporal_displacement_days
    )
    assert {
        item.missing_modalities
        for item in temporal
    } == {()}


def test_every_system_and_tier_is_present(
    experiment,
    configuration,
) -> None:
    assert {
        item.system
        for item in experiment.evaluations
    } == set(configuration.systems)

    assert {
        item.tier
        for item in experiment.evaluations
    } == set(configuration.severity_tiers)

    for item in experiment.evaluations:
        assert item.event_count == 40
        assert len(item.latent_event_ids) == 40
        assert len(item.reciprocal_ranks) == 40
        assert 0.0 <= item.recall_at_1 <= 1.0
        assert 0.0 <= item.recall_at_10 <= 1.0
        assert (
            0.0
            <= item.mean_reciprocal_rank
            <= 1.0
        )
        assert 0.0 <= item.ndcg_at_10 <= 1.0


def test_pairing_and_governance(
    experiment,
) -> None:
    groups = {}

    for item in experiment.evaluations:
        key = (
            item.axis,
            item.condition_id,
            item.tier,
        )
        groups.setdefault(
            key,
            set(),
        ).add(item.latent_event_ids)

    assert groups
    assert all(
        len(latent_sets) == 1
        for latent_sets in groups.values()
    )

    assert experiment.paired_analysis_unit == (
        "latent_event_id"
    )
    assert experiment.oracle_used is False
    assert experiment.ood_tuning_used is False
    assert (
        experiment.final_test_tuning_used
        is False
    )


def test_experiment_is_deterministic(
    experiment,
    configuration,
    system_configuration,
    policy_configuration,
    bundle,
) -> None:
    repeated = run_final_robustness_experiment(
        bundle=bundle,
        configuration=configuration,
        run=configuration.runs[0],
        system_configuration=(
            system_configuration
        ),
        policy_configuration=(
            policy_configuration
        ),
        protocol_hash=(
            configuration.configuration_sha256
        ),
        trained_at_utc=TRAINED_AT,
    )

    assert repeated == experiment


def test_mismatched_seed_is_rejected(
    configuration,
    system_configuration,
    policy_configuration,
    bundle,
) -> None:
    with pytest.raises(
        FinalRobustnessExperimentError,
        match="seed",
    ):
        run_final_robustness_experiment(
            bundle=bundle,
            configuration=configuration,
            run=configuration.runs[1],
            system_configuration=(
                system_configuration
            ),
            policy_configuration=(
                policy_configuration
            ),
            protocol_hash=(
                configuration
                .configuration_sha256
            ),
            trained_at_utc=TRAINED_AT,
        )
