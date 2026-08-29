"""Tests for the NOI v0.3 preimplementation protocol."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.evaluation.noi_v0_3_protocol import (
    EXPECTED_PROTOCOL_ID,
    NOIProtocolConfigurationError,
    load_noi_v0_3_protocol,
)


PROTOCOL_PATH = Path("configs/noi_v0.3_protocol.yaml")


@pytest.fixture
def valid_configuration() -> dict:
    """Return an independent copy of the v0.3 protocol."""

    with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)

    return deepcopy(configuration)


def write_configuration(
    tmp_path: Path,
    configuration: dict,
) -> Path:
    """Write one temporary v0.3 protocol configuration."""

    path = tmp_path / "noi_v0.3_protocol.yaml"

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            configuration,
            handle,
            sort_keys=False,
        )

    return path


def test_valid_protocol_loads() -> None:
    """The prespecified v0.3 protocol must load successfully."""

    configuration = load_noi_v0_3_protocol(PROTOCOL_PATH)

    assert configuration["protocol"]["id"] == EXPECTED_PROTOCOL_ID
    assert tuple(configuration["hypotheses"]) == (
        "H6",
        "H7",
        "H8",
    )
    assert len(
        configuration["synthetic_dataset"]["independent_seeds"]
    ) == 10
    assert len(
        configuration["paired_conditions"]["conditions"]
    ) == 7


def test_missing_protocol_file_is_rejected(
    tmp_path: Path,
) -> None:
    """A nonexistent v0.3 protocol cannot be loaded."""

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="does not exist",
    ):
        load_noi_v0_3_protocol(
            tmp_path / "missing.yaml"
        )


def test_invalid_yaml_is_rejected(
    tmp_path: Path,
) -> None:
    """Malformed YAML must fail before development begins."""

    path = tmp_path / "invalid.yaml"
    path.write_text(
        "protocol:\n  id: [unclosed\n",
        encoding="utf-8",
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="not valid YAML",
    ):
        load_noi_v0_3_protocol(path)


def test_nonmapping_protocol_is_rejected(
    tmp_path: Path,
) -> None:
    """The protocol root must be a mapping."""

    path = tmp_path / "list.yaml"
    path.write_text(
        "- invalid\n- protocol\n",
        encoding="utf-8",
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="must be a mapping",
    ):
        load_noi_v0_3_protocol(path)


def test_missing_required_section_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Every prespecified top-level mapping is required."""

    del valid_configuration["integrity_controls"]
    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="Missing required v0.3 sections",
    ):
        load_noi_v0_3_protocol(path)


def test_wrong_protocol_identifier_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """The protocol identity cannot be silently changed."""

    valid_configuration["protocol"]["id"] = "DIFFERENT-PROTOCOL"
    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="protocol.id",
    ):
        load_noi_v0_3_protocol(path)


def test_wrong_parent_commit_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """The immutable v0.2.0 parent commit must be preserved."""

    valid_configuration["parent_release"]["commit"] = "0" * 40
    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="parent_release.commit",
    ):
        load_noi_v0_3_protocol(path)


@pytest.mark.parametrize(
    "invalid_seeds",
    (
        list(range(1, 11)),
        list(range(1301, 1310)),
        list(range(1301, 1311)) + [1311],
        [True] + list(range(1302, 1311)),
    ),
)
def test_modified_seed_plan_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
    invalid_seeds: list,
) -> None:
    """The ten independent seeds must remain prespecified."""

    valid_configuration[
        "synthetic_dataset"
    ]["independent_seeds"] = invalid_seeds

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="seed",
    ):
        load_noi_v0_3_protocol(path)


def test_modified_support_allocation_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Final-test support regimes cannot be rebalanced later."""

    valid_configuration[
        "synthetic_dataset"
    ]["final_test_support_allocation"][
        "seen_item_events"
    ] = 801

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="support allocation",
    ):
        load_noi_v0_3_protocol(path)


def test_unseen_family_leakage_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Unseen-family test leakage can never be enabled."""

    valid_configuration[
        "synthetic_dataset"
    ]["split_controls"][
        "unseen_family_test_leakage_allowed"
    ] = True

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="unseen_family_test_leakage_allowed",
    ):
        load_noi_v0_3_protocol(path)


def test_modified_tactile_dimension_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """The simulated tactile representation remains eight-dimensional."""

    valid_configuration[
        "modalities"
    ]["tactile"]["representation_dimension"] = 9

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="representation_dimension",
    ):
        load_noi_v0_3_protocol(path)


def test_label_encoded_touch_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Target labels must remain prohibited tactile inputs."""

    valid_configuration[
        "modalities"
    ]["tactile"]["prohibited_inputs"].remove(
        "target_label"
    )

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="prohibited inputs",
    ):
        load_noi_v0_3_protocol(path)


def test_removed_stress_condition_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """All seven paired stress conditions must remain present."""

    del valid_configuration[
        "paired_conditions"
    ]["conditions"]["contradictory_modalities"]

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="seven paired conditions",
    ):
        load_noi_v0_3_protocol(path)


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("threshold", "value"),
        ("uncertainty_band", "lower"),
        ("uncertainty_band", "upper"),
    ),
)
def test_unlocked_support_threshold_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
    section: str,
    field: str,
) -> None:
    """Support thresholds cannot be filled before protocol lock."""

    valid_configuration[
        "support_gate"
    ][section][field] = 0.50

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="must remain null",
    ):
        load_noi_v0_3_protocol(path)


@pytest.mark.parametrize(
    "section",
    (
        "reliability_threshold",
        "conflict_threshold",
    ),
)
def test_unlocked_fusion_threshold_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
    section: str,
) -> None:
    """Fusion thresholds cannot be filled before protocol lock."""

    valid_configuration[
        "fusion_policy"
    ][section]["value"] = 0.50

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="must remain null",
    ):
        load_noi_v0_3_protocol(path)


def test_physical_sensor_claim_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Synthetic evidence cannot be changed into a physical claim."""

    valid_configuration["scope"][
        "physical_tactile_sensor_validated"
    ] = True

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="physical_tactile_sensor_validated",
    ):
        load_noi_v0_3_protocol(path)


def test_test_label_calibration_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Final-test labels cannot be enabled for calibration."""

    valid_configuration[
        "integrity_controls"
    ]["test_labels_for_calibration_allowed"] = True

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="test_labels_for_calibration_allowed",
    ):
        load_noi_v0_3_protocol(path)


def test_modified_artifact_path_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Versioned artifact locations must remain stable."""

    valid_configuration[
        "planned_artifacts"
    ]["final_results"] = "different.md"

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        NOIProtocolConfigurationError,
        match="artifact paths",
    ):
        load_noi_v0_3_protocol(path)
