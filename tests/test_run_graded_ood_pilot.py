"""Tests for the reproducible graded-OOD pilot workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.run_graded_ood_pilot import (
    GradedOODPilotWorkflowError,
    run_graded_ood_pilot,
    verify_graded_ood_pilot_export,
)
from src.evaluation.amendment_config import file_sha256


@pytest.fixture(scope="module")
def export_directory(
    tmp_path_factory,
) -> Path:
    """Create one complete temporary pilot export."""

    path = tmp_path_factory.mktemp(
        "graded-ood-export"
    )

    result = run_graded_ood_pilot(path)

    assert result["verification_passed"] is True

    return path


def test_workflow_creates_three_files(
    export_directory: Path,
) -> None:
    """The workflow must create results, statistics, and manifest."""

    assert {
        path.name
        for path in export_directory.iterdir()
    } == {
        "graded_ood_results.json",
        "paired_statistics.json",
        "run_manifest.json",
    }


def test_results_document_has_twelve_evaluations(
    export_directory: Path,
) -> None:
    """Four baselines by three tiers must be exported."""

    document = json.loads(
        (
            export_directory
            / "graded_ood_results.json"
        ).read_text(encoding="utf-8")
    )

    assert document["owner"] == "GUARDIANX LLC"
    assert document["latent_event_count"] == 40
    assert document["odor_library_size"] == 200
    assert document["oracle_used"] is False
    assert len(document["evaluations"]) == 12


def test_statistics_document_has_twelve_comparisons(
    export_directory: Path,
) -> None:
    """All locked paired contrasts must be exported."""

    document = json.loads(
        (
            export_directory
            / "paired_statistics.json"
        ).read_text(encoding="utf-8")
    )

    assert document["bootstrap_seed"] == 4242
    assert document["bootstrap_resamples"] == 10_000
    assert document["confidence_level"] == 0.95
    assert document["oracle_used"] is False
    assert len(document["comparisons"]) == 12


def test_manifest_preserves_locked_counts(
    export_directory: Path,
) -> None:
    """The manifest must distinguish latent units from observed rows."""

    manifest = json.loads(
        (
            export_directory
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["counts"] == {
        "original_events": 200,
        "odor_targets": 200,
        "latent_ood_events": 40,
        "observed_graded_ood_rows": 120,
        "tier_rows": {
            "mild": 40,
            "moderate": 40,
            "severe": 40,
        },
        "baseline_tier_evaluations": 12,
        "paired_statistical_comparisons": 12,
    }


def test_manifest_records_safety_limits(
    export_directory: Path,
) -> None:
    """Export metadata must prohibit unsupported conclusions."""

    manifest = json.loads(
        (
            export_directory
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["status"] == "exploratory"
    assert (
        manifest["verification"]["severe_reference_replay"]
        == "PASSED"
    )
    assert manifest["verification"]["oracle_used"] is False
    assert (
        manifest["verification"]["analysis_unit"]
        == "latent_event_id"
    )
    assert len(manifest["scope_limitations"]) >= 5


def test_manifest_file_hashes_are_correct(
    export_directory: Path,
) -> None:
    """Every recorded output digest must match its file."""

    manifest = json.loads(
        (
            export_directory
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )

    for filename, metadata in manifest[
        "output_files"
    ].items():
        assert file_sha256(
            export_directory / filename
        ) == metadata["sha256"]


def test_export_verification_passes(
    export_directory: Path,
) -> None:
    """An unchanged export must pass independent verification."""

    result = verify_graded_ood_pilot_export(
        export_directory
    )

    assert result == {
        "passed": True,
        "failures": [],
    }


def test_tampered_result_is_detected(
    export_directory: Path,
) -> None:
    """Post-export modification must fail hash verification."""

    results_path = (
        export_directory
        / "graded_ood_results.json"
    )
    original = results_path.read_text(
        encoding="utf-8"
    )

    try:
        results_path.write_text(
            original + " ",
            encoding="utf-8",
        )

        verification = verify_graded_ood_pilot_export(
            export_directory
        )

        assert verification["passed"] is False
        assert any(
            "SHA-256 mismatch" in failure
            for failure in verification["failures"]
        )
    finally:
        results_path.write_text(
            original,
            encoding="utf-8",
        )


def test_existing_export_requires_overwrite(
    export_directory: Path,
) -> None:
    """Existing results cannot be overwritten accidentally."""

    with pytest.raises(
        GradedOODPilotWorkflowError,
        match="already exist",
    ):
        run_graded_ood_pilot(
            export_directory
        )


def test_intentional_overwrite_is_deterministic(
    export_directory: Path,
) -> None:
    """An explicit rerun must reproduce identical output hashes."""

    before = {
        path.name: file_sha256(path)
        for path in export_directory.iterdir()
    }

    result = run_graded_ood_pilot(
        export_directory,
        overwrite=True,
    )

    after = {
        path.name: file_sha256(path)
        for path in export_directory.iterdir()
    }

    assert result["verification_passed"] is True
    assert after == before


def test_separate_exports_are_deterministic(
    tmp_path: Path,
    export_directory: Path,
) -> None:
    """Two independent directories must contain identical bytes."""

    second = tmp_path / "second"
    run_graded_ood_pilot(second)

    for filename in (
        "graded_ood_results.json",
        "paired_statistics.json",
        "run_manifest.json",
    ):
        assert file_sha256(
            second / filename
        ) == file_sha256(
            export_directory / filename
        )


@pytest.mark.parametrize(
    "invalid_overwrite",
    (
        1,
        0,
        "yes",
        None,
    ),
)
def test_invalid_overwrite_value_is_rejected(
    tmp_path: Path,
    invalid_overwrite,
) -> None:
    """Overwrite authorization must be explicitly boolean."""

    with pytest.raises(
        GradedOODPilotWorkflowError,
        match="boolean",
    ):
        run_graded_ood_pilot(
            tmp_path / "invalid",
            overwrite=invalid_overwrite,
        )


def test_non_directory_output_is_rejected(
    tmp_path: Path,
) -> None:
    """An ordinary file cannot serve as the output directory."""

    path = tmp_path / "ordinary-file"
    path.write_text(
        "not a directory",
        encoding="utf-8",
    )

    with pytest.raises(
        GradedOODPilotWorkflowError,
        match="not a directory",
    ):
        run_graded_ood_pilot(path)
