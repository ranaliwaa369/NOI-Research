"""Tests for the post-confirmatory read-only trace audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.audit_noi_v0_3_confirmatory_trace import (
    EXPECTED_INTEGRITY,
    TraceAuditError,
    _parser,
    _reciprocal_rank,
    _seed_lock,
    _verify_sidecar,
    audit_seed_payload,
    export_trace_audit,
)


def test_default_output_uses_release_artifact_directory() -> None:
    args = _parser().parse_args([])

    assert args.output == Path(
        "artifacts/noi_v0.3_confirmatory/"
        "trace_audit.json"
    )


def test_reciprocal_rank_reproduces_scoring_rule() -> None:
    ranking = [
        "target-a",
        "target-b",
        "target-c",
    ]

    assert _reciprocal_rank(
        ranking,
        "target-a",
    ) == 1.0
    assert _reciprocal_rank(
        ranking,
        "target-b",
    ) == 0.5
    assert _reciprocal_rank(
        ranking,
        "missing",
    ) == 0.0


def test_seed_lock_accepts_string_seed_keys() -> None:
    lock = {
        "values_by_seed": {
            "1301": {
                "support_threshold": -2.0,
                "support_uncertainty_lower": -2.5,
                "support_uncertainty_upper": -1.5,
                "reliability_threshold": 0.2,
                "conflict_threshold": 0.3,
            }
        }
    }

    values = _seed_lock(lock, 1301)

    assert values["support_threshold"] == -2.0
    assert values["conflict_threshold"] == 0.3


def test_replaced_suffix_sidecar_is_accepted(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "validation_lock.yaml"
    artifact.write_text(
        "locked: true\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    artifact.with_suffix(
        ".sha256"
    ).write_text(
        f"{digest}  validation_lock.yaml\n",
        encoding="utf-8",
    )

    assert _verify_sidecar(
        artifact
    ) == digest


def test_hash_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        '{"value": 1}\n',
        encoding="utf-8",
    )
    Path(f"{artifact}.sha256").write_text(
        f"{'0' * 64}  artifact.json\n",
        encoding="utf-8",
    )

    with pytest.raises(
        TraceAuditError,
        match="SHA-256 mismatch",
    ):
        _verify_sidecar(artifact)


def test_injected_integrity_violation_is_rejected() -> None:
    payload = {
        "schema_version": (
            "noi-v0.3-confirmatory-seed-v1"
        ),
        "study_phase": "confirmatory",
        "confirmatory_execution": True,
        "seed": 1301,
        "final_latent_event_count": 2000,
        "condition_view_count": 14000,
        "system_count": 9,
        "system_evaluation_count": 126000,
        "integrity": {
            **EXPECTED_INTEGRITY,
            "final_test_labels_used_for_training": True,
        },
    }

    with pytest.raises(
        TraceAuditError,
        match="Integrity record mismatch",
    ):
        audit_seed_payload(
            payload=payload,
            raw_sha256="a" * 64,
            expected_seed=1301,
            locked_values={
                "support_threshold": -2.0,
                "support_uncertainty_lower": -2.0,
                "support_uncertainty_upper": -2.0,
                "reliability_threshold": 0.2,
                "conflict_threshold": 0.3,
            },
        )


def test_injected_locked_value_change_is_rejected() -> None:
    payload = {
        "schema_version": (
            "noi-v0.3-confirmatory-seed-v1"
        ),
        "study_phase": "confirmatory",
        "confirmatory_execution": True,
        "seed": 1301,
        "final_latent_event_count": 2000,
        "condition_view_count": 14000,
        "system_count": 9,
        "system_evaluation_count": 126000,
        "integrity": dict(EXPECTED_INTEGRITY),
        "locked_values": {
            "support_threshold": -1.9,
            "support_uncertainty_lower": -2.0,
            "support_uncertainty_upper": -2.0,
            "reliability_threshold": 0.2,
            "conflict_threshold": 0.3,
        },
    }

    with pytest.raises(
        TraceAuditError,
        match="Lock mismatch",
    ):
        audit_seed_payload(
            payload=payload,
            raw_sha256="a" * 64,
            expected_seed=1301,
            locked_values={
                "support_threshold": -2.0,
                "support_uncertainty_lower": -2.0,
                "support_uncertainty_upper": -2.0,
                "reliability_threshold": 0.2,
                "conflict_threshold": 0.3,
            },
        )


def test_export_is_hashed_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trace.json"
    payload = {
        "classification": (
            "post_hoc_read_only_integrity_audit"
        )
    }

    digest = export_trace_audit(
        payload,
        output,
        overwrite=False,
    )

    observed = hashlib.sha256(
        output.read_bytes()
    ).hexdigest()

    assert digest == observed
    assert json.loads(
        output.read_text(encoding="utf-8")
    ) == payload
    assert Path(
        f"{output}.sha256"
    ).read_text(
        encoding="utf-8"
    ).split()[0] == observed

    with pytest.raises(
        TraceAuditError,
        match="Refusing to overwrite",
    ):
        export_trace_audit(
            payload,
            output,
            overwrite=False,
        )
