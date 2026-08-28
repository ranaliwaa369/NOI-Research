"""Tests for the final robustness configuration."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from src.evaluation.robustness_config import (
    LOCKED_CONDITIONS,
    LOCKED_TEMPORAL_DAYS,
    RobustnessConfigurationError,
    load_robustness_configuration,
)


CONFIG_PATH = Path(
    "configs/"
    "robustness_evaluation_v0.2.3.yaml"
)
HASH_PATH = Path(
    "configs/"
    "robustness_evaluation_v0.2.3.sha256"
)


def load():
    return load_robustness_configuration(
        CONFIG_PATH,
        HASH_PATH,
    )


def write_pair(
    tmp_path,
    payload,
):
    config_path = tmp_path / "config.yaml"
    hash_path = tmp_path / "config.sha256"

    config_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    digest = sha256(
        config_path.read_bytes()
    ).hexdigest()
    hash_path.write_text(
        digest + "\n",
        encoding="utf-8",
    )

    return config_path, hash_path


def test_locked_configuration_loads() -> None:
    config = load()

    assert config.version == "0.2.3"
    assert config.event_count_per_run == 10000
    assert len(config.runs) == 10
    assert config.bootstrap_resamples == 10000
    assert config.bootstrap_seed == 4245
    assert config.confidence_level == 0.95


def test_all_missing_combinations_are_locked() -> None:
    config = load()

    observed = tuple(
        (
            item.condition_id,
            item.missing_count,
            item.missing_modalities,
        )
        for item in config.missing_conditions
    )

    assert observed == LOCKED_CONDITIONS
    assert len(observed) == 7


def test_temporal_values_are_locked() -> None:
    config = load()

    assert (
        config.temporal_displacement_days
        == LOCKED_TEMPORAL_DAYS
        == (0, 1, 7, 30, 90)
    )


def test_systems_and_baselines_are_locked() -> None:
    config = load()

    assert config.systems == (
        "ridge_only",
        "memory_only",
        "hybrid_without_temporal_decay",
        "full_hybrid",
    )
    assert config.baseline_systems == (
        "ridge_only",
        "memory_only",
        "hybrid_without_temporal_decay",
    )
    assert config.full_system == "full_hybrid"


def test_hash_mismatch_is_rejected(
    tmp_path,
) -> None:
    config_path = tmp_path / "config.yaml"
    hash_path = tmp_path / "config.sha256"

    config_path.write_bytes(
        CONFIG_PATH.read_bytes()
    )
    hash_path.write_text(
        "0" * 64 + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RobustnessConfigurationError,
        match="SHA-256",
    ):
        load_robustness_configuration(
            config_path,
            hash_path,
        )


def test_changed_temporal_value_is_rejected(
    tmp_path,
) -> None:
    payload = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )
    payload[
        "temporal_displacement"
    ]["days"] = [0, 1, 7, 30]

    config_path, hash_path = write_pair(
        tmp_path,
        payload,
    )

    with pytest.raises(
        RobustnessConfigurationError,
        match="Temporal",
    ):
        load_robustness_configuration(
            config_path,
            hash_path,
        )


def test_changed_missing_condition_is_rejected(
    tmp_path,
) -> None:
    payload = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )
    payload[
        "missing_modality"
    ]["conditions"].pop()

    config_path, hash_path = write_pair(
        tmp_path,
        payload,
    )

    with pytest.raises(
        RobustnessConfigurationError,
        match="Missing-modality",
    ):
        load_robustness_configuration(
            config_path,
            hash_path,
        )


def test_configuration_is_deterministic() -> None:
    assert load() == load()
