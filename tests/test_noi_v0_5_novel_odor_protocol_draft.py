"""Guardrails for the NOI v0.5 novel-odor draft protocol."""

from __future__ import annotations

from pathlib import Path

import yaml


PROTOCOL_PATH = Path("configs/noi_v0.5_novel_odor_protocol_draft.yaml")


def load_protocol() -> dict[str, object]:
    """Load the tracked v0.5 draft from the repository root."""

    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_protocol_is_unrun_and_traces_completed_v0_4_milestone() -> None:
    payload = load_protocol()

    assert payload["protocol"]["id"] == "NOI-PROTOCOL-0.5-NOVEL-ODOR"
    assert payload["protocol"]["status"] == "draft_no_final_test_access"
    assert payload["protocol"]["parent_milestone"] == {
        "version": "0.4.0",
        "branch": "development-v0.4-sensor-grounding",
        "commit": "dca0fae",
        "completed_stage": "real_sensor_ingestion_and_integrity_audit",
    }
    assert all(
        hypothesis["final_status"] == "not_run"
        for hypothesis in payload["hypotheses"].values()
    )


def test_h10_h11_h12_have_locked_distinct_roles() -> None:
    payload = load_protocol()

    assert tuple(payload["hypotheses"]) == ("H10", "H11", "H12")
    assert payload["hypotheses"]["H10"]["role"] == "primary"
    assert payload["hypotheses"]["H11"]["role"] == "secondary"
    assert payload["hypotheses"]["H12"]["role"] == "secondary"


def test_family_rotation_and_final_test_boundaries_are_explicit() -> None:
    payload = load_protocol()

    assert payload["family_holdout"]["locked_order"] == [
        "fruits",
        "herbs",
        "nuts",
        "spices",
        "vegetables",
    ]
    rules = payload["integrity_rules"]
    assert rules["recording_identity_may_cross_roles"] is False
    assert rules["labels_may_enter_model_features"] is False
    assert rules["official_test_may_tune_models_or_thresholds"] is False
    assert rules["final_unknown_family_may_enter_fold_development"] is False
    assert rules["final_metrics_may_be_viewed_before_lock"] is False


def test_repeat_sensing_requires_independent_temporal_evidence() -> None:
    payload = load_protocol()
    evidence = payload["temporal_evidence"]

    assert evidence["windows_nonoverlapping"] is True
    assert evidence["partial_tail_dropped"] is True
    assert evidence["repeat_pairs_disjoint"] is True
    assert evidence["window_size_must_be_locked_before_final_execution"] is True


def test_claim_boundary_prohibits_neural_and_deployment_overclaims() -> None:
    boundary = load_protocol()["claim_boundary"]

    assert "direct olfactory nerve interface" in boundary
    assert "human-equivalent smell" in boundary
    assert "deployment readiness" in boundary
