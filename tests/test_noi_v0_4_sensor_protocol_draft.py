"""Guardrails for the NOI v0.4 sensor-grounding draft protocol."""

from __future__ import annotations

from pathlib import Path

import yaml


PROTOCOL_PATH = Path("configs/noi_v0.4_sensor_protocol_draft.yaml")


def load_protocol() -> dict[str, object]:
    """Load the tracked protocol from the repository root."""

    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_protocol_is_explicitly_draft_and_traces_parent_release() -> None:
    payload = load_protocol()

    assert payload["protocol"]["id"] == "NOI-PROTOCOL-0.4-SENSOR-GROUNDING"
    assert payload["protocol"]["status"] == "draft_not_validation_locked"
    assert payload["parent_release"] == {
        "version": "0.3.0",
        "branch": "development-v0.3",
        "commit": "be0a316bf31941cb66da558fb41b167151351f66",
        "doi": "10.5281/zenodo.22180610",
        "supported_result": (
            "Validation-locked support gating rejected registered synthetic "
            "unseen-family queries without clean seen-item MRR loss."
        ),
        "unsupported_results": [
            "conditional tactile synergy",
            "practically sufficient conflict-aware improvement",
        ],
    }


def test_protocol_locks_sensor_to_bounded_decision_architecture() -> None:
    payload = load_protocol()
    architecture = payload["architectural_contribution"]

    assert architecture["name"] == "sensor_to_context_epistemic_bridge"
    assert architecture["stages"] == [
        "physical_sensor_time_series",
        "temporal_olfactory_representation",
        "associative_evidence_memory",
        "open_set_support_estimation",
        "reliability_and_drift_assessment",
        "recognize_repeat_sense_or_abstain_policy",
    ]
    assert set(payload["draft_hypotheses"]) == {"H9", "H10", "H11"}
    assert payload["draft_hypotheses"]["H9"]["role"] == "primary"


def test_protocol_preserves_feature_and_final_test_integrity() -> None:
    payload = load_protocol()

    assert payload["dataset"]["sensor_columns"] == [
        "NO2",
        "C2H5OH",
        "VOC",
        "CO",
        "Alcohol",
        "LPG",
    ]
    assert payload["dataset"]["raw_data_committed_to_noi_repository"] is False
    assert payload["integrity_controls"]["labels_appended_to_sensor_features"] is False
    assert payload["integrity_controls"]["final_test_threshold_selection_allowed"] is False
    assert payload["development_stages"]["stage_0"]["final_test_metrics_allowed"] is False
    assert payload["development_stages"]["stage_2"]["final_test_tuning_allowed"] is False


def test_protocol_prohibits_neural_hardware_and_deployment_overclaims() -> None:
    payload = load_protocol()
    scope = payload["scope"]
    prohibited = payload["architectural_contribution"]["prohibited_interpretations"]

    assert scope["physical_sensor_signal_used"] is True
    assert scope["robot_hardware_integrated"] is False
    assert scope["biological_equivalence_claims_allowed"] is False
    assert scope["deployment_claims_allowed"] is False
    assert "direct_olfactory_nerve_interface" in prohibited
    assert "brain_signal_decoding" in prohibited
    assert "human_neural_equivalence" in prohibited

