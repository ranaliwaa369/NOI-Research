"""Tests for the preregistered NOI protocol amendment configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.evaluation.amendment_config import (
    EXPECTED_AMENDMENT_ID,
    EXPECTED_PARENT_PROTOCOL_SHA256,
    AmendmentConfigurationError,
    file_sha256,
    load_amendment_configuration,
)


AMENDMENT_PATH = Path("configs/protocol_amendment_v0.2.yaml")
PARENT_PROTOCOL_PATH = Path("configs/research_protocol.yaml")


@pytest.fixture
def valid_configuration() -> dict:
    """Return an independent copy of the validated amendment."""

    with AMENDMENT_PATH.open("r", encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)

    return deepcopy(configuration)


def write_configuration(
    tmp_path: Path,
    configuration: dict,
) -> Path:
    """Write one temporary amendment configuration."""

    path = tmp_path / "protocol_amendment.yaml"

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            configuration,
            handle,
            sort_keys=False,
        )

    return path


def test_valid_amendment_loads() -> None:
    """The locked amendment must load successfully."""

    configuration = load_amendment_configuration(
        AMENDMENT_PATH,
        PARENT_PROTOCOL_PATH,
    )

    assert configuration["amendment"]["id"] == EXPECTED_AMENDMENT_ID
    assert tuple(
        configuration["graded_ood_design"]["tiers"]
    ) == (
        "mild",
        "moderate",
        "severe",
    )


def test_parent_protocol_hash_matches_locked_value() -> None:
    """The original protocol must retain its preregistered digest."""

    assert (
        file_sha256(PARENT_PROTOCOL_PATH)
        == EXPECTED_PARENT_PROTOCOL_SHA256
    )


def test_missing_amendment_file_is_rejected(
    tmp_path: Path,
) -> None:
    """A nonexistent amendment cannot be loaded."""

    with pytest.raises(
        AmendmentConfigurationError,
        match="does not exist",
    ):
        load_amendment_configuration(
            tmp_path / "missing.yaml",
            PARENT_PROTOCOL_PATH,
        )


def test_invalid_yaml_is_rejected(
    tmp_path: Path,
) -> None:
    """Malformed YAML must fail before any evaluation begins."""

    path = tmp_path / "invalid.yaml"
    path.write_text(
        "amendment:\n  id: [unclosed\n",
        encoding="utf-8",
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match="not valid YAML",
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


def test_missing_required_section_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Every preregistered top-level section is mandatory."""

    del valid_configuration["statistical_analysis"]
    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match="Missing required amendment sections",
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


def test_wrong_amendment_identifier_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """The loader must reject an unexpected amendment identity."""

    valid_configuration["amendment"]["id"] = "UNREGISTERED-AMENDMENT"
    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match="Amendment id",
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


def test_wrong_owner_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """GUARDIANX LLC ownership cannot be silently changed."""

    valid_configuration["amendment"]["owner"] = "Different Owner"
    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match="GUARDIANX LLC",
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


def test_tampered_parent_protocol_is_rejected(
    tmp_path: Path,
) -> None:
    """A modified parent protocol must fail SHA-256 verification."""

    tampered_parent = tmp_path / "research_protocol.yaml"
    original = PARENT_PROTOCOL_PATH.read_text(encoding="utf-8")

    tampered_parent.write_text(
        original + "\n# unauthorized modification\n",
        encoding="utf-8",
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match="does not match its declared SHA-256",
    ):
        load_amendment_configuration(
            AMENDMENT_PATH,
            tampered_parent,
        )


@pytest.mark.parametrize(
    ("tier_name", "invalid_strength"),
    (
        ("mild", 0.10),
        ("moderate", 0.75),
        ("severe", 0.90),
        ("mild", float("nan")),
        ("moderate", float("inf")),
        ("severe", True),
    ),
)
def test_modified_shift_strength_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
    tier_name: str,
    invalid_strength,
) -> None:
    """OOD transformation strengths are locked by the amendment."""

    valid_configuration[
        "graded_ood_design"
    ]["tiers"][tier_name]["shift_strength"] = invalid_strength

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match="shift_strength",
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


@pytest.mark.parametrize(
    ("tier_name", "invalid_seed"),
    (
        ("mild", 1),
        ("moderate", 2),
        ("severe", 3),
        ("mild", -1),
        ("moderate", True),
        ("severe", 9001.0),
    ),
)
def test_modified_tier_seed_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
    tier_name: str,
    invalid_seed,
) -> None:
    """Each tier must retain its preregistered deterministic seed."""

    valid_configuration[
        "graded_ood_design"
    ]["tiers"][tier_name]["seed"] = invalid_seed

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match="seed",
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


def test_missing_ood_tier_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Mild, moderate, and severe tiers must all remain present."""

    del valid_configuration[
        "graded_ood_design"
    ]["tiers"]["moderate"]

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match="OOD tiers",
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


def test_target_identifier_exposure_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Ground-truth target identifiers cannot enter model features."""

    valid_configuration[
        "graded_ood_design"
    ]["transformation_rule"][
        "target_identifiers_exposed_to_features"
    ] = True

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match="Target identifiers",
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


def test_hidden_adjustment_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Unregistered clipping or hidden adjustment is prohibited."""

    valid_configuration[
        "graded_ood_design"
    ]["transformation_rule"][
        "clipping_or_hidden_adjustment_allowed"
    ] = True

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match="Hidden adjustment",
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


def test_unpaired_ood_evaluation_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Severity comparisons must remain paired by latent event."""

    valid_configuration[
        "confirmatory_evaluation"
    ]["ood_evaluation"]["paired_across_severity_tiers"] = False

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match="paired evaluation",
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


@pytest.mark.parametrize(
    ("key", "invalid_value"),
    (
        ("bootstrap_resamples", 9999),
        ("bootstrap_seed", 1234),
        ("confidence_level", 0.90),
        ("bootstrap_unit", "observed_row"),
    ),
)
def test_modified_statistical_plan_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
    key: str,
    invalid_value,
) -> None:
    """Locked statistical settings cannot change silently."""

    valid_configuration[
        "statistical_analysis"
    ][key] = invalid_value

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match=key,
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


@pytest.mark.parametrize(
    "oracle_key",
    (
        "allowed_for_diagnostics_only",
        "excluded_from_baseline_superiority_tests",
        "excluded_from_deployment_claims",
        "must_be_labeled_as_ood_calibrated",
    ),
)
def test_weakened_oracle_restriction_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
    oracle_key: str,
) -> None:
    """The OOD-calibrated oracle must remain diagnostic only."""

    valid_configuration[
        "statistical_analysis"
    ]["oracle_policy"][oracle_key] = False

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match=oracle_key,
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


@pytest.mark.parametrize(
    "lock_key",
    (
        "implementation_must_follow_amendment",
        "tests_required_before_confirmatory_run",
        "leakage_audit_required",
        "deterministic_export_required",
        "hashes_required",
        "confirmatory_results_must_not_be_inspected_before_lock",
    ),
)
def test_weakened_implementation_lock_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
    lock_key: str,
) -> None:
    """Confirmatory safeguards cannot be disabled silently."""

    valid_configuration[
        "implementation_lock"
    ][lock_key] = False

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match=lock_key,
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )


@pytest.mark.parametrize(
    "commitment_key",
    (
        "original_negative_result_retained",
        "severe_ood_stress_test_retained",
        "original_hypotheses_unchanged",
        "original_success_thresholds_unchanged",
        "locked_metrics_unchanged",
        "safety_policy_unchanged",
        "leakage_controls_unchanged",
        "synthetic_scope_unchanged",
    ),
)
def test_weakened_original_commitment_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
    commitment_key: str,
) -> None:
    """The amendment cannot erase prior scientific commitments."""

    valid_configuration[
        "unchanged_commitments"
    ][commitment_key] = False

    path = write_configuration(
        tmp_path,
        valid_configuration,
    )

    with pytest.raises(
        AmendmentConfigurationError,
        match=commitment_key,
    ):
        load_amendment_configuration(
            path,
            PARENT_PROTOCOL_PATH,
        )