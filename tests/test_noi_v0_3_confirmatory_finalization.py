"""Tests for secure NOI v0.3 confirmatory finalization."""

import hashlib
import json
from pathlib import Path

import pytest

import experiments.finalize_noi_v0_3_confirmatory as finalizer
from experiments.finalize_noi_v0_3_confirmatory import (
    ConfirmatoryFinalizationError,
    build_environment_manifest,
    finalize_confirmatory_results,
    load_verified_seed_payloads,
    render_findings_markdown,
    sha256_file,
    verify_sha256_sidecar,
)


REGISTERED_SEEDS = tuple(range(1301, 1311))


def write_seed(directory: Path, seed: int) -> Path:
    payload = {
        "schema_version": "noi-v0.3-confirmatory-seed-v1",
        "study_phase": "confirmatory",
        "confirmatory_execution": True,
        "seed": seed,
        "final_test_event_count": 2000,
        "integrity": {
            "final_test_tuning_used": False,
            "final_test_labels_used_for_inference": False,
            "thresholds_changed": False,
        },
        "view_results": [],
    }

    path = directory / f"seed-{seed}.json"
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )

    return path


def test_sha256_file_and_sidecar_verification(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"value": 1}\n', encoding="utf-8")

    digest = sha256_file(path)
    sidecar = Path(f"{path}.sha256")
    sidecar.write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )

    assert len(digest) == 64
    assert verify_sha256_sidecar(path) == digest


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "seed-1301.json"
    path.write_text("{}\n", encoding="utf-8")
    Path(f"{path}.sha256").write_text(
        f"{'0' * 64}  {path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfirmatoryFinalizationError,
        match="SHA-256",
    ):
        verify_sha256_sidecar(path)


def test_all_registered_seed_files_are_required(
    tmp_path: Path,
) -> None:
    for seed in REGISTERED_SEEDS:
        write_seed(tmp_path, seed)

    payloads = load_verified_seed_payloads(tmp_path)

    assert tuple(
        payload["seed"] for payload in payloads
    ) == REGISTERED_SEEDS

    (tmp_path / "seed-1310.json").unlink()

    with pytest.raises(
        ConfirmatoryFinalizationError,
        match="seed-1310",
    ):
        load_verified_seed_payloads(tmp_path)


def test_environment_manifest_has_no_runtime_timestamp() -> None:
    manifest = build_environment_manifest()

    assert manifest["schema_version"] == (
        "noi-v0.3-environment-manifest-v1"
    )
    assert manifest["study_phase"] == "confirmatory"
    assert manifest["python_version"]
    assert manifest["platform"]
    assert manifest["numpy_version"]
    assert "timestamp" not in manifest
    assert "generated_at" not in manifest


def test_findings_report_includes_every_hypothesis() -> None:
    aggregate = {
        "schema_version": (
            "noi-v0.3-confirmatory-aggregate-v1"
        ),
        "registered_seeds": list(REGISTERED_SEEDS),
        "hypotheses": {
            "H6": {
                "role": "primary",
                "status": "supported",
            },
            "H7": {
                "role": "secondary",
                "status": "not_supported",
            },
            "H8": {
                "role": "secondary",
                "status": "supported",
            },
        },
        "integrity": {
            "all_registered_seeds_retained": True,
            "final_test_tuning_used": False,
        },
    }

    report = render_findings_markdown(aggregate)

    assert "# NOI v0.3 Confirmatory Findings" in report
    assert "H6: Supported" in report
    assert "H7: Not supported" in report
    assert "H8: Supported" in report
    assert "synthetic" in report.lower()
    assert "physical sensor" in report.lower()
    assert "clinical" in report.lower()


def test_invalid_aggregate_cannot_be_rendered() -> None:
    with pytest.raises(
        ConfirmatoryFinalizationError,
        match="schema",
    ):
        render_findings_markdown({})

def test_finalization_writes_and_hashes_all_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for seed in REGISTERED_SEEDS:
        write_seed(tmp_path, seed)

    aggregate = {
        "schema_version": (
            "noi-v0.3-confirmatory-aggregate-v1"
        ),
        "registered_seeds": list(REGISTERED_SEEDS),
        "hypotheses": {
            "H6": {
                "role": "primary",
                "status": "supported",
            },
            "H7": {
                "role": "secondary",
                "status": "not_supported",
            },
            "H8": {
                "role": "secondary",
                "status": "supported",
            },
        },
        "integrity": {
            "all_registered_seeds_retained": True,
            "final_test_tuning_used": False,
        },
    }

    monkeypatch.setattr(
        finalizer,
        "analyze_confirmatory_payloads",
        lambda payloads: aggregate,
    )

    outputs = finalize_confirmatory_results(tmp_path)

    assert set(outputs) == {
        "aggregate",
        "aggregate_sha256",
        "environment_manifest",
        "environment_manifest_sha256",
        "findings",
        "findings_sha256",
    }
    assert all(path.is_file() for path in outputs.values())

    verify_sha256_sidecar(outputs["aggregate"])
    verify_sha256_sidecar(
        outputs["environment_manifest"]
    )
    verify_sha256_sidecar(outputs["findings"])

    with pytest.raises(
        ConfirmatoryFinalizationError,
        match="already exists",
    ):
        finalize_confirmatory_results(tmp_path)

