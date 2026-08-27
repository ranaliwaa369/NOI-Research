"""Tests for the locked Track B configuration."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from src.evaluation.track_b_config import (
    TrackBConfigurationError,
    load_track_b_configuration,
)


CONFIG_PATH = Path(
    "configs/"
    "track_b_unseen_family_evaluation_v0.2.2.yaml"
)
HASH_PATH = Path(
    "configs/"
    "track_b_unseen_family_evaluation_v0.2.2.sha256"
)


def write_config_pair(
    tmp_path: Path,
    payload: dict,
) -> tuple[Path, Path]:
    """Write one temporary YAML file and matching digest."""

    config_path = tmp_path / "track_b.yaml"
    hash_path = tmp_path / "track_b.sha256"

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
        f"{digest}  {config_path}\n",
        encoding="utf-8",
    )

    return config_path, hash_path


def load_payload() -> dict:
    """Return a mutable copy of the locked YAML payload."""

    payload = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8",
        )
    )

    assert isinstance(payload, dict)
    return payload


def test_locked_configuration_loads() -> None:
    config = load_track_b_configuration(
        CONFIG_PATH,
        HASH_PATH,
    )

    assert config.version == "0.2.2"
    assert config.track == (
        "Track B: Unseen-family generalization"
    )
    assert config.independent_run_count == 10
    assert len(config.runs) == 10
    assert config.severity_tiers == (
        "mild",
        "moderate",
        "severe",
    )
    assert config.minimum_reachable_coverage == 0.95
    assert config.minimum_ood_abstention_rate == 0.80
    assert config.bootstrap_resamples == 10000
    assert config.bootstrap_seed == 4244


def test_runs_and_seeds_are_unique() -> None:
    config = load_track_b_configuration(
        CONFIG_PATH,
        HASH_PATH,
    )

    assert tuple(
        run.run_id
        for run in config.runs
    ) == tuple(
        f"track-b-seed-{index:02d}"
        for index in range(1, 11)
    )

    assert len({
        run.generator_seed
        for run in config.runs
    }) == 10

    assert len({
        run.ood_seed
        for run in config.runs
    }) == 10

    assert all(
        run.generator_seed != run.ood_seed
        for run in config.runs
    )


def test_leakage_controls_are_locked() -> None:
    config = load_track_b_configuration(
        CONFIG_PATH,
        HASH_PATH,
    )

    assert config.ood_events_used_for_threshold is False
    assert config.final_test_tuning_prohibited is True
    assert config.require_all_ood_unreachable is True
    assert config.strict_family_separation is True
    assert config.target_identifier_used is False
    assert config.family_identifier_used is False
    assert config.ood_oracle_used is False
    assert config.pool_seen_and_unseen_metrics is False


def test_incorrect_hash_is_rejected(
    tmp_path: Path,
) -> None:
    hash_path = tmp_path / "wrong.sha256"
    hash_path.write_text(
        ("0" * 64) + f"  {CONFIG_PATH}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        TrackBConfigurationError,
        match="SHA-256",
    ):
        load_track_b_configuration(
            CONFIG_PATH,
            hash_path,
        )


def test_modified_configuration_is_rejected(
    tmp_path: Path,
) -> None:
    payload = load_payload()
    payload["threshold_calibration"][
        "ood_events_used"
    ] = True

    config_path, hash_path = write_config_pair(
        tmp_path,
        payload,
    )

    with pytest.raises(
        TrackBConfigurationError,
        match="OOD events",
    ):
        load_track_b_configuration(
            config_path,
            hash_path,
        )


def test_final_test_tuning_cannot_be_enabled(
    tmp_path: Path,
) -> None:
    payload = load_payload()
    payload["governance"][
        "prohibit_final_test_tuning"
    ] = False

    config_path, hash_path = write_config_pair(
        tmp_path,
        payload,
    )

    with pytest.raises(
        TrackBConfigurationError,
        match="final-test tuning",
    ):
        load_track_b_configuration(
            config_path,
            hash_path,
        )


def test_unreachable_requirement_cannot_be_disabled(
    tmp_path: Path,
) -> None:
    payload = load_payload()
    payload["dataset"][
        "require_all_final_ood_targets_unreachable"
    ] = False

    config_path, hash_path = write_config_pair(
        tmp_path,
        payload,
    )

    with pytest.raises(
        TrackBConfigurationError,
        match="unreachable",
    ):
        load_track_b_configuration(
            config_path,
            hash_path,
        )


def test_duplicate_run_seed_is_rejected(
    tmp_path: Path,
) -> None:
    payload = load_payload()
    runs = payload["primary_runs"]["runs"]

    runs[1]["generator_seed"] = (
        runs[0]["generator_seed"]
    )

    config_path, hash_path = write_config_pair(
        tmp_path,
        payload,
    )

    with pytest.raises(
        TrackBConfigurationError,
        match="generator_seed",
    ):
        load_track_b_configuration(
            config_path,
            hash_path,
        )


def test_abstention_criterion_is_exact(
    tmp_path: Path,
) -> None:
    payload = load_payload()
    payload[
        "confirmatory_abstention_criterion"
    ][
        "unseen_family_abstention_rate_minimum"
    ] = 0.79

    config_path, hash_path = write_config_pair(
        tmp_path,
        payload,
    )

    with pytest.raises(
        TrackBConfigurationError,
        match="0.80",
    ):
        load_track_b_configuration(
            config_path,
            hash_path,
        )


def test_configuration_is_deterministic() -> None:
    first = load_track_b_configuration(
        CONFIG_PATH,
        HASH_PATH,
    )
    second = load_track_b_configuration(
        CONFIG_PATH,
        HASH_PATH,
    )

    assert first == second
