"""Integrity tests for the pre-lock NOI v0.3 amendment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


PROTOCOL_PATH = Path("configs/noi_v0.3_protocol.yaml")
AMENDMENT_PATH = Path(
    "configs/protocol_amendment_v0.3.yaml"
)


def load_amendment() -> dict:
    """Load the versioned amendment."""

    return yaml.safe_load(
        AMENDMENT_PATH.read_text(encoding="utf-8")
    )


def load_protocol() -> dict:
    """Load the parent protocol."""

    return yaml.safe_load(
        PROTOCOL_PATH.read_text(encoding="utf-8")
    )


def test_amendment_identity_and_stage() -> None:
    """The correction must remain pre-lock and pre-confirmatory."""

    data = load_amendment()

    assert data["amendment"]["id"] == (
        "NOI-PROTOCOL-AMENDMENT-0.3"
    )
    assert data["amendment"]["status"] == "preimplementation"
    assert data["classification"]["stage"] == (
        "post-feasibility and pre-validation-lock"
    )
    assert data["classification"][
        "confirmatory_results_inspected"
    ] is False
    assert data["classification"][
        "confirmatory_hypotheses_tested"
    ] is False
    assert data["classification"][
        "final_test_labels_inspected"
    ] is False


def test_parent_protocol_hash_matches() -> None:
    """The amendment must identify the exact parent protocol bytes."""

    data = load_amendment()
    observed = hashlib.sha256(
        PROTOCOL_PATH.read_bytes()
    ).hexdigest()

    assert data["parent_protocol"]["sha256"] == observed


def test_support_threshold_uses_registered_constraint() -> None:
    """Support calibration must enforce the registered false-known rate."""

    correction = load_amendment()[
        "locked_corrections"
    ]["support_threshold"]

    assert correction["fitting_source"] == "training_only"
    assert correction["calibration_source"] == "validation_only"
    assert correction["selection_constraint"][
        "maximum_validation_false_known_rate"
    ] == 0.05
    assert correction["final_test_labels_allowed"] is False


def test_uncertainty_interval_is_validation_bootstrapped() -> None:
    """The uncertainty interval must not retain a pilot constant."""

    correction = load_amendment()[
        "locked_corrections"
    ]["uncertainty_band"]

    assert correction["source"] == "validation_only"
    assert correction["bootstrap_seed"] == 4242
    assert correction["bootstrap_resamples"] == 10000
    assert correction["confidence_level"] == 0.95
    assert correction["final_test_labels_allowed"] is False


def test_direct_condition_metadata_is_prohibited() -> None:
    """The proposed model cannot consume stress-condition answers."""

    corrections = load_amendment()["locked_corrections"]

    reliability_prohibited = set(
        corrections["reliability_estimation"][
            "prohibited_inputs"
        ]
    )
    conflict_prohibited = set(
        corrections["conflict_detection"][
            "prohibited_inputs"
        ]
    )

    assert "condition_label" in reliability_prohibited
    assert "modality_conflict_flag" in reliability_prohibited
    assert "final_test_label" in reliability_prohibited

    assert "condition_label" in conflict_prohibited
    assert "modality_conflict_flag" in conflict_prohibited
    assert "target labels" in conflict_prohibited
    assert "final-test labels" in conflict_prohibited


def test_only_training_and_validation_are_permitted_before_lock() -> None:
    """Final-test records must remain inaccessible during derivation."""

    policy = load_amendment()["validation_data_policy"]

    assert policy["permitted_splits"]["fitting"] == ["train"]
    assert policy["permitted_splits"][
        "threshold_derivation"
    ] == ["validation"]
    assert policy["prohibited_splits_before_lock"] == [
        "final_test"
    ]


def test_all_registered_seeds_are_preserved() -> None:
    """No registered seed may be silently removed."""

    data = load_amendment()

    assert data["validation_data_policy"][
        "registered_seeds"
    ] == list(range(1301, 1311))
    assert data["validation_data_policy"][
        "silent_seed_removal_allowed"
    ] is False


def test_original_confirmatory_allocation_is_unchanged() -> None:
    """The amendment cannot alter test size or support strata."""

    unchanged = load_amendment()["unchanged_commitments"]

    assert unchanged["final_test_events_per_seed"] == 2000
    assert unchanged["final_test_support_allocation"] == {
        "seen_item": 800,
        "known_family_unseen_item": 600,
        "unseen_family": 600,
    }


def test_parent_protocol_remains_unlocked() -> None:
    """The amendment itself must not silently fill lock values."""

    protocol = load_protocol()

    assert protocol["protocol"]["status"] == "preimplementation"
    assert protocol["support_gate"]["threshold"]["value"] is None
    assert protocol["support_gate"]["uncertainty_band"][
        "lower"
    ] is None
    assert protocol["support_gate"]["uncertainty_band"][
        "upper"
    ] is None
    assert protocol["fusion_policy"][
        "reliability_threshold"
    ]["value"] is None
    assert protocol["fusion_policy"][
        "conflict_threshold"
    ]["value"] is None


def test_post_lock_threshold_changes_are_prohibited() -> None:
    """The amendment must require a separate irreversible lock step."""

    locking = load_amendment()["locking"]

    assert locking[
        "separate_validation_lock_commit_required"
    ] is True
    assert locking[
        "separate_validation_lock_tag_required"
    ] is True
    assert locking[
        "hashes_required_before_confirmatory_evaluation"
    ] is True
    assert locking[
        "threshold_changes_after_lock_allowed"
    ] is False
