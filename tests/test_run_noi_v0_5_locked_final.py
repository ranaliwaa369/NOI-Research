from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.run_noi_v0_5_locked_final import (
    CONFIRMATION,
    LockedFinalError,
    _write_exclusive_state,
    load_and_verify_lock,
    score_hypotheses,
    verify_locked_file,
)


def test_locked_file_hash_accepts_exact_content(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("locked\n", encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    verify_locked_file(path, expected, label="artifact")


def test_locked_file_hash_rejects_changed_content(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(LockedFinalError, match="SHA-256 mismatch"):
        verify_locked_file(path, "0" * 64, label="artifact")


def test_load_lock_rejects_nonlocked_status(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.yaml"
    digest_path = tmp_path / "lock.sha256"
    lock_path.write_text(
        yaml.safe_dump(
            {
                "validation_lock": {
                    "status": "draft",
                    "final_test_executed": False,
                }
            }
        ),
        encoding="utf-8",
    )
    digest_path.write_text(
        hashlib.sha256(lock_path.read_bytes()).hexdigest() + "  lock.yaml\n",
        encoding="utf-8",
    )
    with pytest.raises(LockedFinalError, match="not validation-locked"):
        load_and_verify_lock(lock_path, digest_path)


def test_execution_state_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    payload = {"status": "started", "confirmation": CONFIRMATION}
    _write_exclusive_state(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        _write_exclusive_state(path, payload)


def test_hypotheses_pass_at_locked_boundaries() -> None:
    aggregate = {
        "closed_set_false_known_rate": 1.0,
        "confidence_baseline_false_known_rate": 0.75,
        "sentinel_false_known_rate": 0.50,
        "sentinel_known_acceptance_rate": 0.80,
        "pair_initial_false_known_rate": 0.52,
        "pair_initial_known_acceptance_rate": 0.81,
        "asymmetric_veto_false_known_rate": 0.50,
        "asymmetric_veto_known_acceptance_rate": 0.79,
        "asymmetric_veto_utility_change": 0.01,
        "memory_top1_change": 0.10,
    }
    results = score_hypotheses(
        aggregate,
        required_known_acceptance=0.80,
        maximum_known_acceptance_loss=0.02,
    )
    assert all(result["passed"] for result in results.values())


def test_h11_fails_if_repeat_promotes_risk() -> None:
    aggregate = {
        "closed_set_false_known_rate": 1.0,
        "confidence_baseline_false_known_rate": 0.75,
        "sentinel_false_known_rate": 0.50,
        "sentinel_known_acceptance_rate": 0.82,
        "pair_initial_false_known_rate": 0.50,
        "pair_initial_known_acceptance_rate": 0.82,
        "asymmetric_veto_false_known_rate": 0.51,
        "asymmetric_veto_known_acceptance_rate": 0.80,
        "asymmetric_veto_utility_change": 0.02,
        "memory_top1_change": 0.10,
    }
    results = score_hypotheses(
        aggregate,
        required_known_acceptance=0.80,
        maximum_known_acceptance_loss=0.02,
    )
    assert results["H11"]["passed"] is False


def test_repository_validation_lock_is_hash_valid() -> None:
    lock = load_and_verify_lock()
    assert lock["validation_lock"]["status"] == "validation_locked"
