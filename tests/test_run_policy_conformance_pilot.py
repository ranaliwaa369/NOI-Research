"""Tests for the policy-conformance pilot workflow."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.run_policy_conformance_pilot import (
    PolicyConformancePilotWorkflowError,
    run_policy_conformance_pilot,
    verify_policy_conformance_pilot_export,
)


@pytest.fixture(scope="module")
def exported_pilot(tmp_path_factory):
    output_path = (
        tmp_path_factory.mktemp("policy-conformance")
        / "pilot"
    )
    result = run_policy_conformance_pilot(
        output_path
    )
    return output_path, result


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def test_expected_files_are_created(
    exported_pilot,
) -> None:
    output_path, result = exported_pilot

    assert (
        output_path / "policy_conformance_results.json"
    ).is_file()
    assert (
        output_path / "policy_conformance_summary.json"
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
        output_path / "policy_conformance_results.json"
    )

    assert document["total_case_count"] == 26
    assert document["expected_counts"] == {
        "allow": 4,
        "block": 15,
        "require_missing_information": 7,
    }
    assert len(document["case_results"]) == 26


def test_every_case_retains_decision_audit(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot
    document = load_json(
        output_path / "policy_conformance_results.json"
    )

    for result in document["case_results"]:
        assert result["request_id"]
        assert result["rule_ids"]
        assert result["explanation"]
        assert len(result["protocol_hash"]) == 64


def test_summary_records_exact_conformance(
    exported_pilot,
) -> None:
    output_path, result = exported_pilot
    summary = load_json(
        output_path / "policy_conformance_summary.json"
    )

    assert summary["exact_match_count"] == 26
    assert summary["false_allow_count"] == 0
    assert summary["false_allow_rate"] == 0.0
    assert summary["false_block_count"] == 0
    assert summary["false_block_rate"] == 0.0
    assert summary["exact_conformance_rate"] == 1.0
    assert summary["policy_coverage"] == 1.0
    assert summary["h3_success_rule_passed"] is True
    assert result["h3_success_rule_passed"] is True


def test_summary_preserves_claim_limits(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot
    summary = load_json(
        output_path / "policy_conformance_summary.json"
    )
    interpretation = summary["interpretation"]

    assert (
        interpretation[
            "locked_case_conformance_supported"
        ]
        is True
    )

    for key in (
        "comprehensive_safety_demonstrated",
        "chemical_exposure_safety_demonstrated",
        "clinical_safety_demonstrated",
        "physical_device_safety_demonstrated",
        "adversarial_robustness_demonstrated",
        "legal_compliance_demonstrated",
        "deployment_readiness_demonstrated",
    ):
        assert interpretation[key] is False


def test_no_physical_emission_is_recorded(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot
    results = load_json(
        output_path / "policy_conformance_results.json"
    )
    summary = load_json(
        output_path / "policy_conformance_summary.json"
    )

    assert results["physical_emission_performed"] is False
    assert summary["physical_emission_performed"] is False


def test_manifest_records_preregistration(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot
    manifest = load_json(
        output_path / "run_manifest.json"
    )

    assert (
        manifest["registration"]["tag"]
        == "policy-conformance-v0.1-preimplementation"
    )
    assert (
        manifest["registration"][
            "experiment_implemented_after_registration"
        ]
        is True
    )


def test_manifest_records_configuration_hashes(
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
        "policy_rules.yaml",
        "policy_conformance_evaluation_v0.1.yaml",
    }

    for digest in manifest[
        "configuration_hashes"
    ].values():
        assert len(digest) == 64
        int(digest, 16)


def test_verification_passes(
    exported_pilot,
) -> None:
    output_path, _ = exported_pilot

    assert verify_policy_conformance_pilot_export(
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
        PolicyConformancePilotWorkflowError,
        match="already exist",
    ):
        run_policy_conformance_pilot(
            output_path
        )


def test_intentional_rerun_is_deterministic(
    exported_pilot,
) -> None:
    output_path, original = exported_pilot

    repeated = run_policy_conformance_pilot(
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
        PolicyConformancePilotWorkflowError,
        match="boolean",
    ):
        run_policy_conformance_pilot(
            tmp_path / "output",
            overwrite=invalid_overwrite,
        )


def test_file_output_path_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "file"
    path.write_text("occupied", encoding="utf-8")

    with pytest.raises(
        PolicyConformancePilotWorkflowError,
        match="not a directory",
    ):
        run_policy_conformance_pilot(path)


def test_missing_manifest_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        PolicyConformancePilotWorkflowError,
        match="run_manifest.json is missing",
    ):
        verify_policy_conformance_pilot_export(
            tmp_path
        )


def test_tampering_is_detected(
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
        copied_path / "policy_conformance_results.json"
    )
    results_path.write_text(
        results_path.read_text(encoding="utf-8")
        + "\n",
        encoding="utf-8",
    )

    verification = (
        verify_policy_conformance_pilot_export(
            copied_path
        )
    )

    assert verification["passed"] is False
    assert (
        "SHA-256 mismatch: policy_conformance_results.json"
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
        PolicyConformancePilotWorkflowError,
        match="not valid JSON",
    ):
        verify_policy_conformance_pilot_export(
            copied_path
        )
