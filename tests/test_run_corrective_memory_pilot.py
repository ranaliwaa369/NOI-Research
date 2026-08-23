"""Tests for the corrective-memory pilot workflow."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.run_corrective_memory_pilot import (
    CorrectiveMemoryPilotWorkflowError,
    run_corrective_memory_pilot,
    verify_corrective_memory_pilot_export,
)


@pytest.fixture(scope="module")
def exported_pilot(tmp_path_factory):
    output_path = (
        tmp_path_factory.mktemp("corrective-memory")
        / "pilot"
    )
    result = run_corrective_memory_pilot(
        output_path
    )
    return output_path, result


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def test_expected_files_are_exported(
    exported_pilot,
) -> None:
    output_path, result = exported_pilot

    assert (
        output_path / "corrective_memory_results.json"
    ).is_file()
    assert (
        output_path / "corrective_memory_summary.json"
    ).is_file()
    assert (
        output_path / "run_manifest.json"
    ).is_file()
    assert result["verification_passed"] is True


def test_results_preserve_locked_counts(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot
    document = load_json(
        output_path / "corrective_memory_results.json"
    )

    assert document["training_event_count"] == 140
    assert document["validation_event_count"] == 20
    assert document["eligible_target_count"] == 14
    assert (
        document["eligible_validation_event_count"]
        == 15
    )
    assert len(document["target_results"]) == 14


def test_results_preserve_primary_runtime(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot
    document = load_json(
        output_path / "corrective_memory_results.json"
    )

    assert document["alpha"] == 0.0
    assert document["apply_temporal_decay"] is False
    assert document["top_k"] == 10
    assert document["oracle_used"] is False
    assert document["ood_tuning_used"] is False


def test_all_restorations_are_audited(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot
    document = load_json(
        output_path / "corrective_memory_results.json"
    )

    for result in document["target_results"]:
        assert result["corrupted_memory_ids"]
        assert (
            result["restoration_audit_count"]
            == len(result["corrupted_memory_ids"])
        )


def test_summary_reports_controlled_scope(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot
    summary = load_json(
        output_path / "corrective_memory_summary.json"
    )
    interpretation = summary["interpretation"]

    assert (
        interpretation[
            "controlled_mechanism_restoration_supported"
        ]
        is True
    )
    assert (
        interpretation[
            "automatic_error_detection_tested"
        ]
        is False
    )
    assert (
        interpretation[
            "unseen_target_discovery_tested"
        ]
        is False
    )
    assert (
        interpretation[
            "human_perceptual_validity_tested"
        ]
        is False
    )
    assert (
        interpretation["physical_device_tested"]
        is False
    )


def test_summary_records_success_rules(
    exported_pilot,
) -> None:
    output_path, result = exported_pilot
    summary = load_json(
        output_path / "corrective_memory_summary.json"
    )

    assert (
        summary["correction_success_rule_passed"]
        is True
    )
    assert (
        summary["old_memory_degradation_rule_passed"]
        is True
    )
    assert summary["bootstrap_ci_lower"] > 0.0
    assert (
        summary["mean_old_memory_degradation"]
        <= 0.02
    )
    assert (
        result["correction_success_rule_passed"]
        is True
    )
    assert result["old_memory_rule_passed"] is True


def test_manifest_records_preregistration(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot
    manifest = load_json(
        output_path / "run_manifest.json"
    )

    assert (
        manifest["registration"]["tag"]
        == "corrective-memory-v0.1-preimplementation"
    )
    assert (
        manifest["registration"][
            "experiment_implemented_after_registration"
        ]
        is True
    )


def test_manifest_records_all_configuration_hashes(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot
    manifest = load_json(
        output_path / "run_manifest.json"
    )

    assert set(
        manifest["configuration_hashes"]
    ) == {
        "research_protocol.yaml",
        "synthetic_data.yaml",
        "corrective_memory_evaluation_v0.1.yaml",
        "noi_system_v0.1.yaml",
        "policy_rules.yaml",
    }

    for digest in manifest[
        "configuration_hashes"
    ].values():
        assert len(digest) == 64
        int(digest, 16)


def test_export_verification_passes(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot

    assert verify_corrective_memory_pilot_export(
        output_path
    ) == {
        "passed": True,
        "failures": [],
    }


def test_existing_output_requires_overwrite(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot

    with pytest.raises(
        CorrectiveMemoryPilotWorkflowError,
        match="already exist",
    ):
        run_corrective_memory_pilot(
            output_path
        )


def test_intentional_rerun_is_deterministic(
    exported_pilot,
) -> None:
    output_path, original = exported_pilot

    repeated = run_corrective_memory_pilot(
        output_path,
        overwrite=True,
    )

    assert (
        repeated["results_sha256"]
        == original["results_sha256"]
    )
    assert (
        repeated["summary_sha256"]
        == original["summary_sha256"]
    )
    assert (
        repeated["manifest_sha256"]
        == original["manifest_sha256"]
    )


@pytest.mark.parametrize(
    "invalid_overwrite",
    (1, "yes", None),
)
def test_invalid_overwrite_is_rejected(
    tmp_path: Path,
    invalid_overwrite,
) -> None:
    with pytest.raises(
        CorrectiveMemoryPilotWorkflowError,
        match="boolean",
    ):
        run_corrective_memory_pilot(
            tmp_path / "output",
            overwrite=invalid_overwrite,
        )


def test_file_output_path_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "file"
    path.write_text("occupied", encoding="utf-8")

    with pytest.raises(
        CorrectiveMemoryPilotWorkflowError,
        match="not a directory",
    ):
        run_corrective_memory_pilot(path)


def test_missing_manifest_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        CorrectiveMemoryPilotWorkflowError,
        match="run_manifest.json is missing",
    ):
        verify_corrective_memory_pilot_export(
            tmp_path
        )


def test_tampered_results_fail_verification(
    exported_pilot,
    tmp_path: Path,
) -> None:
    original_path, _ = exported_pilot
    copied_path = tmp_path / "tampered"

    shutil.copytree(
        original_path,
        copied_path,
    )

    results_path = (
        copied_path / "corrective_memory_results.json"
    )
    results_path.write_text(
        results_path.read_text(encoding="utf-8")
        + "\n",
        encoding="utf-8",
    )

    verification = (
        verify_corrective_memory_pilot_export(
            copied_path
        )
    )

    assert verification["passed"] is False
    assert (
        "SHA-256 mismatch: corrective_memory_results.json"
        in verification["failures"]
    )


def test_invalid_manifest_json_is_rejected(
    exported_pilot,
    tmp_path: Path,
) -> None:
    original_path, _ = exported_pilot
    copied_path = tmp_path / "invalid-json"

    shutil.copytree(
        original_path,
        copied_path,
    )

    (
        copied_path / "run_manifest.json"
    ).write_text(
        "{invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        CorrectiveMemoryPilotWorkflowError,
        match="not valid JSON",
    ):
        verify_corrective_memory_pilot_export(
            copied_path
        )
