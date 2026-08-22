"""Tests for the reproducible synthetic-pilot export workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.export_synthetic_pilot import (
    DEFAULT_EVENT_COUNT,
    DEFAULT_OUTPUT_DIRECTORY,
    PilotExportError,
    build_argument_parser,
    run_pilot_export,
)
from src.evaluation.dataset_export import (
    DatasetExportError,
    verify_exported_dataset,
)


SYNTHETIC_CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"


def test_pilot_workflow_exports_and_verifies(
    tmp_path: Path,
) -> None:
    """The complete pilot workflow must produce a verified export."""

    result = run_pilot_export(
        synthetic_config_path=SYNTHETIC_CONFIG_PATH,
        research_protocol_path=PROTOCOL_PATH,
        output_directory=tmp_path / "pilot",
        event_count=200,
    )

    assert result.targets_path.is_file()
    assert result.events_path.is_file()
    assert result.manifest_path.is_file()
    assert verify_exported_dataset(result) is True


def test_pilot_manifest_records_expected_counts(
    tmp_path: Path,
) -> None:
    """The exported manifest must preserve pilot record counts."""

    result = run_pilot_export(
        synthetic_config_path=SYNTHETIC_CONFIG_PATH,
        research_protocol_path=PROTOCOL_PATH,
        output_directory=tmp_path / "pilot",
        event_count=200,
    )

    manifest = json.loads(
        result.manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["counts"]["odor_targets"] == 200
    assert manifest["counts"]["events"] == 200
    assert manifest["counts"]["splits"]["train"] == 140
    assert manifest["counts"]["splits"]["validation"] == 20
    assert manifest["counts"]["splits"]["ood_test"] == 40
    assert manifest["leakage_audit"]["passed"] is True


def test_pilot_workflow_is_deterministic(
    tmp_path: Path,
) -> None:
    """Independent pilot runs must produce identical hashes."""

    first = run_pilot_export(
        synthetic_config_path=SYNTHETIC_CONFIG_PATH,
        research_protocol_path=PROTOCOL_PATH,
        output_directory=tmp_path / "first",
        event_count=200,
    )

    second = run_pilot_export(
        synthetic_config_path=SYNTHETIC_CONFIG_PATH,
        research_protocol_path=PROTOCOL_PATH,
        output_directory=tmp_path / "second",
        event_count=200,
    )

    assert first.targets_sha256 == second.targets_sha256
    assert first.events_sha256 == second.events_sha256
    assert first.manifest_sha256 == second.manifest_sha256


def test_pilot_refuses_unapproved_overwrite(
    tmp_path: Path,
) -> None:
    """Existing pilot artifacts must be protected by default."""

    output_directory = tmp_path / "pilot"

    run_pilot_export(
        synthetic_config_path=SYNTHETIC_CONFIG_PATH,
        research_protocol_path=PROTOCOL_PATH,
        output_directory=output_directory,
        event_count=200,
    )

    with pytest.raises(
        DatasetExportError,
        match="Refusing to overwrite",
    ):
        run_pilot_export(
            synthetic_config_path=SYNTHETIC_CONFIG_PATH,
            research_protocol_path=PROTOCOL_PATH,
            output_directory=output_directory,
            event_count=200,
        )


def test_pilot_allows_explicit_overwrite(
    tmp_path: Path,
) -> None:
    """Explicit overwrite authorization must remain deterministic."""

    output_directory = tmp_path / "pilot"

    first = run_pilot_export(
        synthetic_config_path=SYNTHETIC_CONFIG_PATH,
        research_protocol_path=PROTOCOL_PATH,
        output_directory=output_directory,
        event_count=200,
    )

    second = run_pilot_export(
        synthetic_config_path=SYNTHETIC_CONFIG_PATH,
        research_protocol_path=PROTOCOL_PATH,
        output_directory=output_directory,
        event_count=200,
        overwrite=True,
    )

    assert first.targets_sha256 == second.targets_sha256
    assert first.events_sha256 == second.events_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert verify_exported_dataset(second) is True


@pytest.mark.parametrize(
    "invalid_event_count",
    (
        0,
        -1,
        True,
        1.5,
    ),
)
def test_invalid_event_count_is_rejected(
    tmp_path: Path,
    invalid_event_count,
) -> None:
    """The workflow must reject invalid event-count values."""

    with pytest.raises(
        PilotExportError,
        match="positive integer",
    ):
        run_pilot_export(
            synthetic_config_path=SYNTHETIC_CONFIG_PATH,
            research_protocol_path=PROTOCOL_PATH,
            output_directory=tmp_path / "pilot",
            event_count=invalid_event_count,
        )


def test_command_line_defaults_are_locked() -> None:
    """The CLI defaults must match the documented pilot protocol."""

    parser = build_argument_parser()
    arguments = parser.parse_args([])

    assert arguments.event_count == DEFAULT_EVENT_COUNT
    assert arguments.output_directory == DEFAULT_OUTPUT_DIRECTORY
    assert arguments.overwrite is False