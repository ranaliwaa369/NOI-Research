"""Tests for the claim-bounded NOI v0.4 SmellNet adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.smellnet_adapter import (
    SMELLNET_SENSOR_COLUMNS,
    SmellNetAdapterError,
    SmellNetSplit,
    audit_smellnet_recordings,
    discover_smellnet_base,
    load_smellnet_recording,
)


def write_recording(
    path: Path,
    *,
    rows: list[list[object]] | None = None,
    columns: list[str] | None = None,
) -> None:
    """Write a compact sensor fixture using the official folder shape."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows
        or [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        ],
        columns=columns or list(SMELLNET_SENSOR_COLUMNS),
    ).to_csv(path, index=False)


def test_load_recording_keeps_labels_outside_features(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training" / "cashew" / "sample.csv"
    write_recording(path)

    recording = load_smellnet_recording(
        path,
        split=SmellNetSplit.TRAINING,
    )

    assert recording.recording_id == "training/cashew/sample.csv"
    assert recording.ingredient == "cashew"
    assert recording.family == "nuts"
    assert recording.sensor_columns == SMELLNET_SENSOR_COLUMNS
    assert recording.sensor_rows[0] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert len(recording.source_sha256) == 64


def test_extra_environment_columns_are_not_model_features(
    tmp_path: Path,
) -> None:
    columns = [*SMELLNET_SENSOR_COLUMNS, "Temperature", "Humidity"]
    path = tmp_path / "training" / "lemon" / "sample.csv"
    write_recording(
        path,
        columns=columns,
        rows=[[1, 2, 3, 4, 5, 6, 22.0, 45.0]],
    )

    recording = load_smellnet_recording(
        path,
        split=SmellNetSplit.TRAINING,
    )

    assert recording.sensor_columns == SMELLNET_SENSOR_COLUMNS
    assert recording.sensor_rows == ((1.0, 2.0, 3.0, 4.0, 5.0, 6.0),)


def test_missing_locked_sensor_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "training" / "cashew" / "sample.csv"
    write_recording(
        path,
        columns=list(SMELLNET_SENSOR_COLUMNS[:-1]),
        rows=[[1, 2, 3, 4, 5]],
    )

    with pytest.raises(SmellNetAdapterError, match="Missing locked"):
        load_smellnet_recording(path, split=SmellNetSplit.TRAINING)


@pytest.mark.parametrize("invalid_value", ("not-a-number", float("nan")))
def test_invalid_sensor_value_is_rejected(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    path = tmp_path / "training" / "cashew" / "sample.csv"
    write_recording(
        path,
        rows=[[invalid_value, 2, 3, 4, 5, 6]],
    )

    with pytest.raises(SmellNetAdapterError, match="numeric|finite"):
        load_smellnet_recording(path, split=SmellNetSplit.TRAINING)


def test_unknown_ingredient_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "training" / "unknown_food" / "sample.csv"
    write_recording(path)

    with pytest.raises(SmellNetAdapterError, match="Unknown SmellNet"):
        load_smellnet_recording(path, split=SmellNetSplit.TRAINING)


def test_discovery_is_deterministic_and_preserves_official_splits(
    tmp_path: Path,
) -> None:
    write_recording(tmp_path / "training" / "lemon" / "b.csv")
    write_recording(tmp_path / "training" / "cashew" / "a.csv")
    write_recording(
        tmp_path / "testing" / "lemon" / "c.csv",
        rows=[[11, 12, 13, 14, 15, 16]],
    )

    recordings = discover_smellnet_base(tmp_path)

    assert tuple(value.recording_id for value in recordings) == (
        "training/cashew/a.csv",
        "training/lemon/b.csv",
        "testing/lemon/c.csv",
    )


def test_missing_official_split_is_rejected(tmp_path: Path) -> None:
    write_recording(tmp_path / "training" / "cashew" / "a.csv")

    with pytest.raises(SmellNetAdapterError, match="split directory"):
        discover_smellnet_base(tmp_path)


def test_audit_passes_distinct_recordings(tmp_path: Path) -> None:
    write_recording(tmp_path / "training" / "cashew" / "a.csv")
    write_recording(
        tmp_path / "testing" / "cashew" / "b.csv",
        rows=[[7, 8, 9, 10, 11, 12]],
    )
    recordings = discover_smellnet_base(tmp_path)

    report = audit_smellnet_recordings(recordings, raise_on_failure=True)

    assert report.passed is True
    assert report.recording_count == 2
    assert report.training_count == 1
    assert report.testing_count == 1
    assert report.ingredient_count == 1
    assert report.family_count == 1


def test_audit_rejects_byte_identical_cross_split_files(
    tmp_path: Path,
) -> None:
    write_recording(tmp_path / "training" / "cashew" / "a.csv")
    write_recording(tmp_path / "testing" / "cashew" / "b.csv")
    recordings = discover_smellnet_base(tmp_path)

    report = audit_smellnet_recordings(recordings)

    assert report.passed is False
    assert len(report.cross_split_content_duplicates) == 1
    with pytest.raises(SmellNetAdapterError, match="audit failed"):
        audit_smellnet_recordings(recordings, raise_on_failure=True)


def test_duplicate_recording_identity_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "training" / "cashew" / "a.csv"
    write_recording(path)
    recording = load_smellnet_recording(
        path,
        split=SmellNetSplit.TRAINING,
    )

    report = audit_smellnet_recordings((recording, recording))

    assert report.passed is False
    assert report.duplicate_recording_ids == (
        "training/cashew/a.csv",
    )
