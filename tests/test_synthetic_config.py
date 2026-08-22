"""Tests for the independent synthetic-data safeguards."""

from copy import deepcopy

import pytest
import yaml

from src.evaluation.synthetic_config import (
    SyntheticConfigurationError,
    load_synthetic_configuration,
    validate_synthetic_configuration,
)


CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"


@pytest.fixture
def configuration() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@pytest.fixture
def protocol() -> dict:
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_valid_configuration_loads() -> None:
    result = load_synthetic_configuration(
        CONFIG_PATH,
        PROTOCOL_PATH,
    )

    assert result["generator"]["model_independent"] is True


def test_model_dependent_generator_is_rejected(
    configuration: dict,
    protocol: dict,
) -> None:
    modified = deepcopy(configuration)
    modified["generator"]["model_independent"] = False

    with pytest.raises(
        SyntheticConfigurationError,
        match="model-independent",
    ):
        validate_synthetic_configuration(modified, protocol)


def test_event_count_must_match_protocol(
    configuration: dict,
    protocol: dict,
) -> None:
    modified = deepcopy(configuration)
    modified["dataset"]["total_events"] = 9999

    with pytest.raises(
        SyntheticConfigurationError,
        match="total_events",
    ):
        validate_synthetic_configuration(modified, protocol)


def test_odor_target_minimum_is_enforced(
    configuration: dict,
    protocol: dict,
) -> None:
    modified = deepcopy(configuration)
    modified["dataset"]["odor_targets"] = 100

    with pytest.raises(
        SyntheticConfigurationError,
        match="below the preregistered minimum",
    ):
        validate_synthetic_configuration(modified, protocol)


def test_training_seeds_must_match_protocol(
    configuration: dict,
    protocol: dict,
) -> None:
    modified = deepcopy(configuration)
    modified["randomness"]["training_seeds"] = [1, 2, 3, 4, 5]

    with pytest.raises(
        SyntheticConfigurationError,
        match="Training seeds",
    ):
        validate_synthetic_configuration(modified, protocol)


def test_id_and_ood_seeds_must_differ(
    configuration: dict,
    protocol: dict,
) -> None:
    modified = deepcopy(configuration)
    modified["randomness"]["independent_ood_seed"] = (
        modified["randomness"]["generator_seed"]
    )

    with pytest.raises(
        SyntheticConfigurationError,
        match="different seeds",
    ):
        validate_synthetic_configuration(modified, protocol)


def test_target_identifier_leakage_is_rejected(
    configuration: dict,
    protocol: dict,
) -> None:
    modified = deepcopy(configuration)
    modified["ground_truth"][
        "expose_target_identifier_to_features"
    ] = True

    with pytest.raises(
        SyntheticConfigurationError,
        match="Target identifiers",
    ):
        validate_synthetic_configuration(modified, protocol)


def test_split_fraction_must_match_protocol(
    configuration: dict,
    protocol: dict,
) -> None:
    modified = deepcopy(configuration)
    modified["splits"][
        "odor_family_held_out_fraction"
    ] = 0.10

    with pytest.raises(
        SyntheticConfigurationError,
        match="must match the research protocol",
    ):
        validate_synthetic_configuration(modified, protocol)


def test_ood_independence_cannot_be_disabled(
    configuration: dict,
    protocol: dict,
) -> None:
    modified = deepcopy(configuration)
    modified["ood_generation"][
        "resample_modality_transformations"
    ] = False

    with pytest.raises(
        SyntheticConfigurationError,
        match="OOD guarantee",
    ):
        validate_synthetic_configuration(modified, protocol)


def test_leakage_check_cannot_be_disabled(
    configuration: dict,
    protocol: dict,
) -> None:
    modified = deepcopy(configuration)
    modified["leakage_checks"][
        "prohibit_duplicate_events_across_splits"
    ] = False

    with pytest.raises(
        SyntheticConfigurationError,
        match="Every leakage check",
    ):
        validate_synthetic_configuration(modified, protocol)