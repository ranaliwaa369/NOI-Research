"""Tests for policy-conformance configuration validation."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from src.evaluation.policy_conformance_config import (
    PolicyConformanceConfigurationError,
    load_policy_conformance_configuration,
)


CONFIG_PATH = Path(
    "configs/policy_conformance_evaluation_v0.1.yaml"
)
CHECKSUM_PATH = Path(
    "configs/policy_conformance_evaluation_v0.1.sha256"
)


@pytest.fixture(scope="module")
def valid_configuration():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)


def write_configuration(
    tmp_path: Path,
    configuration: dict,
) -> tuple[Path, Path]:
    path = tmp_path / "configuration.yaml"
    checksum_path = tmp_path / "configuration.sha256"

    path.write_text(
        yaml.safe_dump(
            configuration,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    digest = sha256(
        path.read_bytes()
    ).hexdigest()

    checksum_path.write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )

    return path, checksum_path


def test_locked_configuration_loads() -> None:
    configuration = (
        load_policy_conformance_configuration(
            CONFIG_PATH,
            CHECKSUM_PATH,
        )
    )

    assert len(configuration["cases"]) == 26
    assert (
        configuration["metrics"][
            "violation_false_allow_target"
        ]
        == 0
    )


def test_missing_configuration_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        PolicyConformanceConfigurationError,
        match="Configuration file not found",
    ):
        load_policy_conformance_configuration(
            tmp_path / "missing.yaml",
            CHECKSUM_PATH,
        )


def test_missing_checksum_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        PolicyConformanceConfigurationError,
        match="Checksum file not found",
    ):
        load_policy_conformance_configuration(
            CONFIG_PATH,
            tmp_path / "missing.sha256",
        )


def test_tampered_file_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tampered.yaml"
    path.write_bytes(
        CONFIG_PATH.read_bytes() + b"\n"
    )

    with pytest.raises(
        PolicyConformanceConfigurationError,
        match="SHA-256",
    ):
        load_policy_conformance_configuration(
            path,
            CHECKSUM_PATH,
        )


@pytest.mark.parametrize(
    "checksum",
    ("", "invalid", "0" * 63, "g" * 64),
)
def test_invalid_checksum_is_rejected(
    tmp_path: Path,
    checksum: str,
) -> None:
    checksum_path = tmp_path / "invalid.sha256"
    checksum_path.write_text(
        checksum,
        encoding="utf-8",
    )

    with pytest.raises(
        PolicyConformanceConfigurationError,
        match="Checksum",
    ):
        load_policy_conformance_configuration(
            CONFIG_PATH,
            checksum_path,
        )


@pytest.mark.parametrize(
    ("section", "key", "invalid_value"),
    (
        (
            "metrics",
            "violation_false_allow_target",
            1,
        ),
        (
            "metrics",
            "maximum_false_block_rate",
            0.02,
        ),
        (
            "metrics",
            "exact_conformance_target",
            0.99,
        ),
        (
            "metrics",
            "policy_coverage_target",
            0.99,
        ),
        (
            "case_generation",
            "expected_case_count",
            25,
        ),
        (
            "case_generation",
            "boundary_epsilon",
            0.001,
        ),
        (
            "policy",
            "default_action_required",
            "ALLOW",
        ),
        (
            "policy",
            "physical_emission_performed",
            True,
        ),
    ),
)
def test_modified_setting_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
    section: str,
    key: str,
    invalid_value,
) -> None:
    configuration = deepcopy(
        valid_configuration
    )
    configuration[section][key] = invalid_value

    path, checksum_path = write_configuration(
        tmp_path,
        configuration,
    )

    with pytest.raises(
        PolicyConformanceConfigurationError,
        match=key,
    ):
        load_policy_conformance_configuration(
            path,
            checksum_path,
        )


def test_removed_case_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    configuration = deepcopy(
        valid_configuration
    )
    configuration["cases"].pop()

    path, checksum_path = write_configuration(
        tmp_path,
        configuration,
    )

    with pytest.raises(
        PolicyConformanceConfigurationError,
        match="26",
    ):
        load_policy_conformance_configuration(
            path,
            checksum_path,
        )


def test_duplicate_request_id_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    configuration = deepcopy(
        valid_configuration
    )
    configuration["cases"][1]["request_id"] = (
        configuration["cases"][0]["request_id"]
    )

    path, checksum_path = write_configuration(
        tmp_path,
        configuration,
    )

    with pytest.raises(
        PolicyConformanceConfigurationError,
        match="request_id",
    ):
        load_policy_conformance_configuration(
            path,
            checksum_path,
        )


def test_modified_expected_outcome_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    configuration = deepcopy(
        valid_configuration
    )
    configuration["cases"][0][
        "expected_outcome"
    ] = "BLOCK"

    path, checksum_path = write_configuration(
        tmp_path,
        configuration,
    )

    with pytest.raises(
        PolicyConformanceConfigurationError,
        match="outcome counts",
    ):
        load_policy_conformance_configuration(
            path,
            checksum_path,
        )


def test_invalid_expected_outcome_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    configuration = deepcopy(
        valid_configuration
    )
    configuration["cases"][0][
        "expected_outcome"
    ] = "UNSAFE"

    path, checksum_path = write_configuration(
        tmp_path,
        configuration,
    )

    with pytest.raises(
        PolicyConformanceConfigurationError,
        match="expected_outcome",
    ):
        load_policy_conformance_configuration(
            path,
            checksum_path,
        )


@pytest.mark.parametrize(
    "key",
    (
        "policy_configuration_hash_required",
        "expected_outcome_locked_before_execution",
        "request_order_locked",
        "no_case_removal_after_results",
        "no_physical_emission",
        "no_chemical_safety_claim",
        "no_clinical_safety_claim",
        "exact_decision_audit_retained",
        "deterministic_replay_required",
        "full_test_suite_required_before_release",
    ),
)
def test_safeguard_cannot_be_disabled(
    tmp_path: Path,
    valid_configuration: dict,
    key: str,
) -> None:
    configuration = deepcopy(
        valid_configuration
    )
    configuration["safeguards"][key] = False

    path, checksum_path = write_configuration(
        tmp_path,
        configuration,
    )

    with pytest.raises(
        PolicyConformanceConfigurationError,
        match=key,
    ):
        load_policy_conformance_configuration(
            path,
            checksum_path,
        )


def test_extra_top_level_field_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    configuration = deepcopy(
        valid_configuration
    )
    configuration["unregistered"] = True

    path, checksum_path = write_configuration(
        tmp_path,
        configuration,
    )

    with pytest.raises(
        PolicyConformanceConfigurationError,
        match="Top-level",
    ):
        load_policy_conformance_configuration(
            path,
            checksum_path,
        )
