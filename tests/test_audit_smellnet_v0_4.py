"""Tests for the reproducible NOI v0.4 Stage 0 audit command."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.audit_smellnet_v0_4 import build_audit_payload, main
from src.evaluation.smellnet_adapter import SMELLNET_SENSOR_COLUMNS


def write_recording(path: Path, values: list[int]) -> None:
    """Create one compact official-layout recording fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([values], columns=SMELLNET_SENSOR_COLUMNS).to_csv(
        path,
        index=False,
    )


def test_audit_payload_is_claim_bounded_and_serializable(
    tmp_path: Path,
) -> None:
    write_recording(tmp_path / "training" / "cashew" / "a.csv", [1, 2, 3, 4, 5, 6])
    write_recording(tmp_path / "testing" / "lemon" / "b.csv", [7, 8, 9, 10, 11, 12])

    payload = build_audit_payload(tmp_path)

    assert payload["schema"] == "noi-v0.4-smellnet-integrity-audit-v1"
    assert payload["audit"]["passed"] is True
    assert payload["audit"]["recording_count"] == 2
    assert "does not establish recognition" in payload["claim_boundary"]
    json.dumps(payload)


def test_command_writes_reproducible_json(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_path = tmp_path / "results" / "audit.json"
    write_recording(dataset_root / "training" / "cashew" / "a.csv", [1, 2, 3, 4, 5, 6])
    write_recording(dataset_root / "testing" / "lemon" / "b.csv", [7, 8, 9, 10, 11, 12])

    exit_code = main([str(dataset_root), "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["audit"]["training_count"] == 1
    assert payload["audit"]["testing_count"] == 1


def test_command_returns_nonzero_for_cross_split_duplicate(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    values = [1, 2, 3, 4, 5, 6]
    write_recording(dataset_root / "training" / "cashew" / "a.csv", values)
    write_recording(dataset_root / "testing" / "cashew" / "b.csv", values)

    assert main([str(dataset_root)]) == 1
