"""Integrity tests for the NOI v0.5 physical-sensor validation lock."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import yaml


LOCK_PATH = Path("configs/noi_v0.5_validation_lock.yaml")
HASH_PATH = Path("configs/noi_v0.5_validation_lock.sha256")


def load_lock() -> dict[str, object]:
    return yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8"))


def test_validation_lock_sidecar_matches_exact_bytes() -> None:
    expected = HASH_PATH.read_text(encoding="utf-8").split()[0]
    assert sha256(LOCK_PATH.read_bytes()).hexdigest() == expected


def test_validation_lock_precedes_final_access() -> None:
    payload = load_lock()
    lock = payload["validation_lock"]
    boundary = payload["data_boundary"]
    assert lock["status"] == "validation_locked"
    assert lock["final_test_executed"] is False
    assert boundary["official_test_used_for_development"] is False
    assert boundary["official_test_labels_viewed_for_model_selection"] is False


def test_implementation_and_development_artifacts_are_pinned() -> None:
    provenance = load_lock()["provenance"]
    assert provenance["implementation_commit"] == (
        "528cb537300d5359718090ae6bb81997123ca717"
    )
    for key in (
        "dataset_integrity_sha256",
        "development_summary_sha256",
        "development_evidence_sha256",
    ):
        assert len(provenance[key]) == 64
        int(provenance[key], 16)


def test_sensor_representation_and_policy_are_locked() -> None:
    payload = load_lock()
    representation = payload["locked_representation"]
    calibration = payload["locked_calibration"]
    repeat = payload["locked_repeat_controller"]
    assert representation["window_size"] == 128
    assert representation["feature_dimension"] == 66
    assert len(representation["sensor_channels"]) == 6
    assert len(representation["per_channel_statistics"]) == 11
    assert calibration["required_known_acceptance"] == 0.80
    assert calibration["repeat_margin"] == 0.06
    assert repeat["primary"] == "asymmetric_safety_veto"
    assert repeat["initial_abstention_may_be_promoted_to_recognition"] is False


def test_hypotheses_and_final_rules_are_complete() -> None:
    payload = load_lock()
    assert set(payload["hypotheses"]) == {"H10", "H11", "H12"}
    assert all(
        item["final_status"] == "not_run"
        for item in payload["hypotheses"].values()
    )
    rules = payload["final_execution_rules"]
    assert rules["execute_each_outer_fold_once"] is True
    assert rules["no_model_reselection_after_final_access"] is True
    assert rules["no_threshold_recalibration_after_final_access"] is True
    assert rules["preserve_negative_results"] is True
