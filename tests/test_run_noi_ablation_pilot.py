"""Tests for the reproducible NOI ablation pilot workflow."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.run_noi_ablation_pilot import (
    NOIAblationPilotWorkflowError,
    run_noi_ablation_pilot,
    verify_noi_ablation_pilot_export,
)


@pytest.fixture(scope="module")
def exported_pilot(
    tmp_path_factory,
):
    """Run one complete export for read-only tests."""

    output_path = (
        tmp_path_factory.mktemp("noi-ablation")
        / "pilot"
    )

    result = run_noi_ablation_pilot(
        output_path
    )

    return output_path, result


def load_json(path: Path) -> dict:
    """Load one exported JSON document."""

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def test_workflow_exports_expected_files(
    exported_pilot,
) -> None:
    """The workflow must create all three locked files."""

    output_path, result = exported_pilot

    assert (
        output_path / "noi_ablation_results.json"
    ).is_file()
    assert (
        output_path / "noi_ablation_summary.json"
    ).is_file()
    assert (
        output_path / "run_manifest.json"
    ).is_file()
    assert result["verification_passed"] is True


def test_export_preserves_locked_counts(
    exported_pilot,
) -> None:
    """The results document must preserve experimental counts."""

    output_path, result = exported_pilot
    document = load_json(
        output_path / "noi_ablation_results.json"
    )

    assert document["latent_event_count"] == 40
    assert document["odor_library_size"] == 200
    assert document["training_event_count"] == 140
    assert document["validation_event_count"] == 20
    assert len(document["evaluations"]) == 60
    assert result["evaluation_count"] == 60


def test_export_records_no_oracle_or_ood_tuning(
    exported_pilot,
) -> None:
    """Prohibited OOD adaptation must remain absent."""

    output_path, _ = exported_pilot
    document = load_json(
        output_path / "noi_ablation_results.json"
    )

    assert document["oracle_used"] is False
    assert document["ood_tuning_used"] is False
    assert (
        document["paired_analysis_unit"]
        == "latent_event_id"
    )


def test_summary_reports_negative_memory_result(
    exported_pilot,
) -> None:
    """The workflow cannot hide the observed negative ablation."""

    output_path, result = exported_pilot
    summary = load_json(
        output_path / "noi_ablation_summary.json"
    )

    assert (
        summary[
            "full_hybrid_equals_ridge_rankings_all_conditions"
        ]
        is True
    )
    assert summary["memory_only_any_nonzero_mrr"] is False
    assert (
        summary["interpretation"][
            "associative_memory_incremental_benefit_detected"
        ]
        is False
    )
    assert (
        summary["interpretation"][
            "negative_result_reported"
        ]
        is True
    )
    assert result["full_hybrid_equals_ridge"] is True
    assert (
        result["memory_incremental_benefit_detected"]
        is False
    )


def test_summary_contains_all_paired_mrr_differences(
    exported_pilot,
) -> None:
    """Three tiers across five times require 15 comparisons."""

    output_path, _ = exported_pilot
    summary = load_json(
        output_path / "noi_ablation_summary.json"
    )
    comparisons = summary[
        "full_hybrid_minus_ridge_only"
    ]

    assert len(comparisons) == 15

    for comparison in comparisons:
        assert comparison["difference"] == pytest.approx(
            comparison["full_hybrid_mrr"]
            - comparison["ridge_only_mrr"],
            abs=1e-15,
        )


def test_manifest_records_preregistration(
    exported_pilot,
) -> None:
    """The manifest must identify the preimplementation tag."""

    output_path, _ = exported_pilot
    manifest = load_json(
        output_path / "run_manifest.json"
    )

    assert (
        manifest["registration"]["tag"]
        == "noi-ablation-v0.1-preimplementation"
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
    """Every locked input must have a SHA-256 digest."""

    output_path, _ = exported_pilot
    manifest = load_json(
        output_path / "run_manifest.json"
    )

    expected = {
        "research_protocol.yaml",
        "protocol_amendment_v0.2.yaml",
        "graded_ood_generation.yaml",
        "synthetic_data.yaml",
        "noi_system_v0.1.yaml",
        "policy_rules.yaml",
    }

    assert (
        set(manifest["configuration_hashes"])
        == expected
    )

    for digest in manifest[
        "configuration_hashes"
    ].values():
        assert len(digest) == 64
        int(digest, 16)


def test_export_verification_passes(
    exported_pilot,
) -> None:
    """Fresh output must pass independent verification."""

    output_path, _ = exported_pilot
    verification = (
        verify_noi_ablation_pilot_export(
            output_path
        )
    )

    assert verification == {
        "passed": True,
        "failures": [],
    }


def test_existing_output_requires_explicit_overwrite(
    exported_pilot,
) -> None:
    """Existing results cannot be silently overwritten."""

    output_path, _ = exported_pilot

    with pytest.raises(
        NOIAblationPilotWorkflowError,
        match="already exist",
    ):
        run_noi_ablation_pilot(output_path)


def test_intentional_rerun_is_deterministic(
    exported_pilot,
) -> None:
    """An explicit rerun must reproduce every output hash."""

    output_path, original = exported_pilot

    repeated = run_noi_ablation_pilot(
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
    """The overwrite safeguard must require a boolean."""

    with pytest.raises(
        NOIAblationPilotWorkflowError,
        match="boolean",
    ):
        run_noi_ablation_pilot(
            tmp_path / "output",
            overwrite=invalid_overwrite,
        )


def test_output_path_that_is_file_is_rejected(
    tmp_path: Path,
) -> None:
    """A regular file cannot be used as an output directory."""

    path = tmp_path / "not-a-directory"
    path.write_text("occupied", encoding="utf-8")

    with pytest.raises(
        NOIAblationPilotWorkflowError,
        match="not a directory",
    ):
        run_noi_ablation_pilot(path)


def test_missing_manifest_is_rejected(
    tmp_path: Path,
) -> None:
    """Verification requires a manifest."""

    with pytest.raises(
        NOIAblationPilotWorkflowError,
        match="run_manifest.json is missing",
    ):
        verify_noi_ablation_pilot_export(
            tmp_path
        )


def test_tampered_result_file_fails_verification(
    exported_pilot,
    tmp_path: Path,
) -> None:
    """Hash verification must detect changed output content."""

    original_path, _ = exported_pilot
    copied_path = tmp_path / "tampered"
    shutil.copytree(
        original_path,
        copied_path,
    )

    results_path = (
        copied_path / "noi_ablation_results.json"
    )
    results_path.write_text(
        results_path.read_text(encoding="utf-8")
        + "\n",
        encoding="utf-8",
    )

    verification = (
        verify_noi_ablation_pilot_export(
            copied_path
        )
    )

    assert verification["passed"] is False
    assert (
        "SHA-256 mismatch: noi_ablation_results.json"
        in verification["failures"]
    )


def test_invalid_manifest_json_is_rejected(
    exported_pilot,
    tmp_path: Path,
) -> None:
    """Malformed manifest JSON must fail explicitly."""

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
        NOIAblationPilotWorkflowError,
        match="not valid JSON",
    ):
        verify_noi_ablation_pilot_export(
            copied_path
        )
