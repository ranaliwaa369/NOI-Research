"""Tests for deterministic synthetic-dataset export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.dataset_export import (
    DatasetExportError,
    export_synthetic_dataset,
    verify_exported_dataset,
)
from src.evaluation.synthetic_generator import generate_synthetic_pilot


SYNTHETIC_CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"


@pytest.fixture(scope="module")
def pilot_dataset():
    """Create one deterministic pilot dataset for exporter tests."""

    return generate_synthetic_pilot(
        SYNTHETIC_CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )


def test_export_creates_required_files(
    pilot_dataset,
    tmp_path: Path,
) -> None:
    """A successful export must create all declared artifacts."""

    result = export_synthetic_dataset(
        pilot_dataset,
        tmp_path / "export",
    )

    assert result.targets_path.is_file()
    assert result.events_path.is_file()
    assert result.manifest_path.is_file()


def test_exported_record_counts_are_correct(
    pilot_dataset,
    tmp_path: Path,
) -> None:
    """JSONL files must contain exactly one line per record."""

    result = export_synthetic_dataset(
        pilot_dataset,
        tmp_path / "export",
    )

    target_lines = result.targets_path.read_text(
        encoding="utf-8"
    ).splitlines()

    event_lines = result.events_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(target_lines) == 200
    assert len(event_lines) == 200


def test_manifest_contains_provenance_and_hashes(
    pilot_dataset,
    tmp_path: Path,
) -> None:
    """The manifest must preserve generator and ownership metadata."""

    result = export_synthetic_dataset(
        pilot_dataset,
        tmp_path / "export",
    )

    manifest = json.loads(
        result.manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == "1.0.0"
    assert manifest["project"] == "Neuro-Olfactive Interface"
    assert manifest["owner"] == "GUARDIANX LLC"
    assert manifest["generator"]["version"] == (
        pilot_dataset.generator_version
    )
    assert manifest["generator"]["generator_seed"] == (
        pilot_dataset.generator_seed
    )
    assert manifest["generator"]["ood_seed"] == (
        pilot_dataset.ood_seed
    )
    assert manifest["leakage_audit"]["passed"] is True

    assert (
        manifest["files"]["odor_targets.jsonl"]["sha256"]
        == result.targets_sha256
    )
    assert (
        manifest["files"]["events.jsonl"]["sha256"]
        == result.events_sha256
    )


def test_exported_hashes_verify_successfully(
    pilot_dataset,
    tmp_path: Path,
) -> None:
    """Untampered export artifacts must pass hash verification."""

    result = export_synthetic_dataset(
        pilot_dataset,
        tmp_path / "export",
    )

    assert verify_exported_dataset(result) is True


def test_tampering_is_detected(
    pilot_dataset,
    tmp_path: Path,
) -> None:
    """A post-export modification must invalidate verification."""

    result = export_synthetic_dataset(
        pilot_dataset,
        tmp_path / "export",
    )

    with result.events_path.open(
        "a",
        encoding="utf-8",
    ) as file_handle:
        file_handle.write('{"tampered":true}\n')

    assert verify_exported_dataset(result) is False


def test_existing_files_are_not_overwritten_by_default(
    pilot_dataset,
    tmp_path: Path,
) -> None:
    """A second export must fail unless overwrite is authorized."""

    output_directory = tmp_path / "export"

    export_synthetic_dataset(
        pilot_dataset,
        output_directory,
    )

    with pytest.raises(
        DatasetExportError,
        match="Refusing to overwrite",
    ):
        export_synthetic_dataset(
            pilot_dataset,
            output_directory,
        )


def test_explicit_overwrite_is_supported(
    pilot_dataset,
    tmp_path: Path,
) -> None:
    """An explicitly authorized deterministic overwrite must succeed."""

    output_directory = tmp_path / "export"

    first_result = export_synthetic_dataset(
        pilot_dataset,
        output_directory,
    )

    second_result = export_synthetic_dataset(
        pilot_dataset,
        output_directory,
        overwrite=True,
    )

    assert (
        first_result.targets_sha256
        == second_result.targets_sha256
    )
    assert (
        first_result.events_sha256
        == second_result.events_sha256
    )
    assert (
        first_result.manifest_sha256
        == second_result.manifest_sha256
    )
    assert verify_exported_dataset(second_result) is True


def test_exports_are_deterministic_across_directories(
    pilot_dataset,
    tmp_path: Path,
) -> None:
    """The same dataset must produce identical bytes and hashes."""

    first_result = export_synthetic_dataset(
        pilot_dataset,
        tmp_path / "first",
    )

    second_result = export_synthetic_dataset(
        pilot_dataset,
        tmp_path / "second",
    )

    assert (
        first_result.targets_path.read_bytes()
        == second_result.targets_path.read_bytes()
    )
    assert (
        first_result.events_path.read_bytes()
        == second_result.events_path.read_bytes()
    )
    assert (
        first_result.manifest_path.read_bytes()
        == second_result.manifest_path.read_bytes()
    )


def test_non_directory_output_is_rejected(
    pilot_dataset,
    tmp_path: Path,
) -> None:
    """An existing ordinary file cannot serve as an output directory."""

    invalid_output = tmp_path / "ordinary-file"
    invalid_output.write_text(
        "not a directory",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetExportError,
        match="not a directory",
    ):
        export_synthetic_dataset(
            pilot_dataset,
            invalid_output,
        )