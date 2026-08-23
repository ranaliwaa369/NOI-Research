"""Tests for the locked corrective-memory evaluation configuration."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from src.evaluation.corrective_memory_config import (
    CorrectiveMemoryConfigurationError,
    load_corrective_memory_configuration,
)


CONFIGURATION_PATH = Path(
    "configs/corrective_memory_evaluation_v0.1.yaml"
)
CHECKSUM_PATH = Path(
    "configs/corrective_memory_evaluation_v0.1.sha256"
)


@pytest.fixture(scope="module")
def valid_configuration() -> dict:
    """Load a mutable copy of the locked configuration."""

    with CONFIGURATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)


def write_configuration(
    tmp_path: Path,
    configuration: dict,
) -> tuple[Path, Path]:
    """Write modified YAML with a matching checksum."""

    path = tmp_path / "configuration.yaml"
    checksum_path = tmp_path / "configuration.sha256"

    text = yaml.safe_dump(
        configuration,
        sort_keys=False,
    )
    path.write_text(
        text,
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
    """The committed configuration and checksum must validate."""

    configuration = load_corrective_memory_configuration(
        CONFIGURATION_PATH,
        CHECKSUM_PATH,
    )

    assert (
        configuration["schema"]["name"]
        == "NOI Corrective Memory Evaluation"
    )
    assert configuration["schema"]["version"] == "0.1.0"
    assert (
        configuration["eligibility"][
            "expected_eligible_targets"
        ]
        == 14
    )


def test_missing_configuration_is_rejected(
    tmp_path: Path,
) -> None:
    """A missing YAML definition must fail explicitly."""

    with pytest.raises(
        CorrectiveMemoryConfigurationError,
        match="Configuration file not found",
    ):
        load_corrective_memory_configuration(
            tmp_path / "missing.yaml",
            CHECKSUM_PATH,
        )


def test_missing_checksum_is_rejected(
    tmp_path: Path,
) -> None:
    """The YAML cannot load without its checksum file."""

    with pytest.raises(
        CorrectiveMemoryConfigurationError,
        match="Checksum file not found",
    ):
        load_corrective_memory_configuration(
            CONFIGURATION_PATH,
            tmp_path / "missing.sha256",
        )


def test_tampered_configuration_is_rejected(
    tmp_path: Path,
) -> None:
    """A changed file with the old checksum must fail."""

    path = tmp_path / "tampered.yaml"
    path.write_bytes(
        CONFIGURATION_PATH.read_bytes()
        + b"\n"
    )

    with pytest.raises(
        CorrectiveMemoryConfigurationError,
        match="SHA-256",
    ):
        load_corrective_memory_configuration(
            path,
            CHECKSUM_PATH,
        )


@pytest.mark.parametrize(
    "invalid_checksum",
    (
        "",
        "not-a-hash",
        "0" * 63,
        "g" * 64,
    ),
)
def test_invalid_checksum_format_is_rejected(
    tmp_path: Path,
    invalid_checksum: str,
) -> None:
    """Malformed checksum text must not be accepted."""

    checksum_path = tmp_path / "invalid.sha256"
    checksum_path.write_text(
        invalid_checksum,
        encoding="utf-8",
    )

    with pytest.raises(
        CorrectiveMemoryConfigurationError,
        match="Checksum",
    ):
        load_corrective_memory_configuration(
            CONFIGURATION_PATH,
            checksum_path,
        )


@pytest.mark.parametrize(
    ("section", "key", "invalid_value"),
    (
        (
            "eligibility",
            "expected_eligible_validation_events",
            14,
        ),
        (
            "eligibility",
            "expected_eligible_targets",
            13,
        ),
        (
            "eligibility",
            "expected_ineligible_validation_targets",
            4,
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
            "retrieval",
            "ood_oracle_used",
            True,
        ),
        (
            "retrieval",
            "ood_tuning_used",
            True,
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
        (
            "old_memory_degradation",
            "maximum_allowed_mean_degradation",
            0.03,
        ),
    ),
)
def test_modified_locked_setting_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
    section: str,
    key: str,
    invalid_value,
) -> None:
    """Result-affecting settings cannot change silently."""

    configuration = deepcopy(
        valid_configuration
    )
    configuration[section][key] = invalid_value

    path, checksum_path = write_configuration(
        tmp_path,
        configuration,
    )

    with pytest.raises(
        CorrectiveMemoryConfigurationError,
        match=key,
    ):
        load_corrective_memory_configuration(
            path,
            checksum_path,
        )


@pytest.mark.parametrize(
    "key",
    (
        "exploratory_status",
        "negative_results_reported",
        "post_result_setting_changes_prohibited",
    ),
)
def test_governance_safeguard_cannot_be_disabled(
    tmp_path: Path,
    valid_configuration: dict,
    key: str,
) -> None:
    """Preregistration safeguards must remain enabled."""

    configuration = deepcopy(
        valid_configuration
    )
    configuration["governance"][key] = False

    path, checksum_path = write_configuration(
        tmp_path,
        configuration,
    )

    with pytest.raises(
        CorrectiveMemoryConfigurationError,
        match=key,
    ):
        load_corrective_memory_configuration(
            path,
            checksum_path,
        )


@pytest.mark.parametrize(
    "key",
    (
        "train_validation_overlap_prohibited",
        "target_truth_hidden_from_retrieval_features",
        "correction_truth_used_only_for_controlled_intervention",
        "no_physical_emission",
        "policy_gate_not_used_to_change_rankings",
        "deterministic_replay_required",
        "immutable_result_records_required",
        "full_test_suite_required_before_release",
    ),
)
def test_safeguard_cannot_be_disabled(
    tmp_path: Path,
    valid_configuration: dict,
    key: str,
) -> None:
    """Safety and reproducibility safeguards are mandatory."""

    configuration = deepcopy(
        valid_configuration
    )
    configuration["safeguards"][key] = False

    path, checksum_path = write_configuration(
        tmp_path,
        configuration,
    )

    with pytest.raises(
        CorrectiveMemoryConfigurationError,
        match=key,
    ):
        load_corrective_memory_configuration(
            path,
            checksum_path,
        )


def test_extra_top_level_field_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """Unknown top-level fields cannot alter the protocol."""

    configuration = deepcopy(
        valid_configuration
    )
    configuration["unregistered_change"] = True

    path, checksum_path = write_configuration(
        tmp_path,
        configuration,
    )

    with pytest.raises(
        CorrectiveMemoryConfigurationError,
        match="Top-level",
    ):
        load_corrective_memory_configuration(
            path,
            checksum_path,
        )


def test_empty_interpretation_limit_is_rejected(
    tmp_path: Path,
    valid_configuration: dict,
) -> None:
    """The scientific claim boundary cannot be removed."""

    configuration = deepcopy(
        valid_configuration
    )
    configuration["interpretation_limit"] = ""

    path, checksum_path = write_configuration(
        tmp_path,
        configuration,
    )

    with pytest.raises(
        CorrectiveMemoryConfigurationError,
        match="interpretation_limit",
    ):
        load_corrective_memory_configuration(
            path,
            checksum_path,
        )
