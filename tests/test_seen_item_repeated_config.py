"""Tests for the locked repeated Track A protocol."""

from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from src.evaluation.seen_item_repeated_config import (
    EXPECTED_REPEATED_RUNS,
    SeenItemRepeatedConfigError,
    load_seen_item_repeated_config,
)


CONFIG_PATH = Path(
    "configs/seen_item_repeated_evaluation_v0.2.1.yaml"
)
HASH_PATH = Path(
    "configs/seen_item_repeated_evaluation_v0.2.1.sha256"
)


def load_raw_config() -> dict:
    return yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )


def write_config_and_hash(
    tmp_path: Path,
    config: dict,
) -> tuple[Path, Path]:
    config_path = tmp_path / "repeated.yaml"
    hash_path = tmp_path / "repeated.sha256"

    text = yaml.safe_dump(
        config,
        sort_keys=False,
    )

    config_path.write_text(
        text,
        encoding="utf-8",
    )

    digest = sha256(
        text.encode("utf-8")
    ).hexdigest()

    hash_path.write_text(
        f"{digest}  {config_path}\n",
        encoding="utf-8",
    )

    return config_path, hash_path


def test_locked_repeated_configuration_loads() -> None:
    config = load_seen_item_repeated_config(
        CONFIG_PATH,
        HASH_PATH,
    )

    assert config.version == "0.2.1"
    assert config.event_count == 10000
    assert config.include_pilot_reference is False
    assert config.runs == EXPECTED_REPEATED_RUNS
    assert len(config.runs) == 10


def test_all_run_identifiers_and_seeds_are_unique() -> None:
    config = load_seen_item_repeated_config(
        CONFIG_PATH,
        HASH_PATH,
    )

    assert len(
        {run.run_id for run in config.runs}
    ) == 10
    assert len(
        {run.generator_seed for run in config.runs}
    ) == 10
    assert len(
        {run.ood_seed for run in config.runs}
    ) == 10
    assert len(
        {run.partition_seed for run in config.runs}
    ) == 10


def test_pilot_reference_cannot_enter_primary_runs(
    tmp_path: Path,
) -> None:
    raw = load_raw_config()
    raw["primary_runs"][
        "include_pilot_reference"
    ] = True

    config_path, hash_path = write_config_and_hash(
        tmp_path,
        raw,
    )

    with pytest.raises(
        SeenItemRepeatedConfigError,
        match="pilot reference",
    ):
        load_seen_item_repeated_config(
            config_path,
            hash_path,
        )


def test_modified_seed_is_rejected(
    tmp_path: Path,
) -> None:
    raw = load_raw_config()
    raw["primary_runs"]["runs"][0][
        "generator_seed"
    ] = 999999

    config_path, hash_path = write_config_and_hash(
        tmp_path,
        raw,
    )

    with pytest.raises(
        SeenItemRepeatedConfigError,
        match="locked seed schedule",
    ):
        load_seen_item_repeated_config(
            config_path,
            hash_path,
        )


def test_unexpected_section_is_rejected(
    tmp_path: Path,
) -> None:
    raw = load_raw_config()
    raw["unexpected_text"] = {
        "value": "not allowed"
    }

    config_path, hash_path = write_config_and_hash(
        tmp_path,
        raw,
    )

    with pytest.raises(
        SeenItemRepeatedConfigError,
        match="Unexpected protocol sections",
    ):
        load_seen_item_repeated_config(
            config_path,
            hash_path,
        )


def test_incorrect_sha256_is_rejected(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "repeated.yaml"
    hash_path = tmp_path / "repeated.sha256"

    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    hash_path.write_text(
        f"{'0' * 64}  {config_path}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SeenItemRepeatedConfigError,
        match="SHA-256",
    ):
        load_seen_item_repeated_config(
            config_path,
            hash_path,
        )
