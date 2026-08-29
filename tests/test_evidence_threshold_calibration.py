"""Tests for validation-only evidence-threshold calibration."""

from __future__ import annotations

from dataclasses import replace
import inspect
import math

import pytest

from src.evaluation.evidence_conflict import (
    ConflictCalibrationObservation,
    EvidenceConflictError,
    ReliabilityCalibrationObservation,
    calibrate_conflict_threshold,
    calibrate_reliability_threshold,
)
from src.evaluation.multisensory_records import (
    MultisensorySplit,
)


def reliability_observations():
    """Return perfectly separated validation reliability examples."""

    return (
        ReliabilityCalibrationObservation(
            source_split=MultisensorySplit.VALIDATION,
            reliability=0.90,
            prediction_correct=True,
        ),
        ReliabilityCalibrationObservation(
            source_split=MultisensorySplit.VALIDATION,
            reliability=0.80,
            prediction_correct=True,
        ),
        ReliabilityCalibrationObservation(
            source_split=MultisensorySplit.VALIDATION,
            reliability=0.40,
            prediction_correct=False,
        ),
        ReliabilityCalibrationObservation(
            source_split=MultisensorySplit.VALIDATION,
            reliability=0.20,
            prediction_correct=False,
        ),
    )


def conflict_observations():
    """Return validation conflicts with one permitted false conflict."""

    negatives = tuple(
        ConflictCalibrationObservation(
            source_split=MultisensorySplit.VALIDATION,
            conflict_score=(
                0.85
                if index == 0
                else 0.40 - (index * 0.01)
            ),
            conflict_present=False,
        )
        for index in range(20)
    )
    positives = (
        ConflictCalibrationObservation(
            source_split=MultisensorySplit.VALIDATION,
            conflict_score=0.90,
            conflict_present=True,
        ),
        ConflictCalibrationObservation(
            source_split=MultisensorySplit.VALIDATION,
            conflict_score=0.80,
            conflict_present=True,
        ),
    )

    return positives + negatives


def test_reliability_calibration_interface_is_minimal() -> None:
    """Reliability calibration accepts validation observations only."""

    signature = inspect.signature(
        calibrate_reliability_threshold
    )

    assert set(signature.parameters) == {"observations"}


def test_conflict_calibration_interface_is_minimal() -> None:
    """Conflict calibration exposes only observations and locked cap."""

    signature = inspect.signature(
        calibrate_conflict_threshold
    )

    assert set(signature.parameters) == {
        "observations",
        "maximum_false_conflict_rate",
    }


def test_reliability_threshold_maximizes_balanced_accuracy() -> None:
    """The registered reliability objective must be applied exactly."""

    report = calibrate_reliability_threshold(
        reliability_observations()
    )

    assert 0.40 < report.threshold < 0.80
    assert report.balanced_accuracy == 1.0
    assert report.validation_observation_count == 4
    assert report.source_split is MultisensorySplit.VALIDATION
    assert report.final_test_labels_used is False


def test_conflict_threshold_enforces_false_conflict_cap() -> None:
    """Conflict detection must remain inside the five-percent cap."""

    report = calibrate_conflict_threshold(
        conflict_observations(),
        maximum_false_conflict_rate=0.05,
    )

    assert report.validation_false_conflict_rate <= 0.05
    assert report.maximum_false_conflict_rate == 0.05
    assert report.conflict_true_positive_rate == 1.0
    assert report.source_split is MultisensorySplit.VALIDATION
    assert report.final_test_labels_used is False


def test_calibration_is_deterministic() -> None:
    """Identical observations produce exactly identical reports."""

    first_reliability = calibrate_reliability_threshold(
        reliability_observations()
    )
    second_reliability = calibrate_reliability_threshold(
        reliability_observations()
    )
    first_conflict = calibrate_conflict_threshold(
        conflict_observations(),
        maximum_false_conflict_rate=0.05,
    )
    second_conflict = calibrate_conflict_threshold(
        conflict_observations(),
        maximum_false_conflict_rate=0.05,
    )

    assert first_reliability == second_reliability
    assert first_conflict == second_conflict


@pytest.mark.parametrize(
    "split",
    (
        MultisensorySplit.TRAIN,
        MultisensorySplit.FINAL_TEST,
    ),
)
def test_reliability_calibration_rejects_nonvalidation_split(
    split: MultisensorySplit,
) -> None:
    """Training and final labels cannot enter reliability calibration."""

    observations = list(reliability_observations())
    observations[0] = replace(
        observations[0],
        source_split=split,
    )

    with pytest.raises(
        EvidenceConflictError,
        match="validation observations only",
    ):
        calibrate_reliability_threshold(tuple(observations))


@pytest.mark.parametrize(
    "split",
    (
        MultisensorySplit.TRAIN,
        MultisensorySplit.FINAL_TEST,
    ),
)
def test_conflict_calibration_rejects_nonvalidation_split(
    split: MultisensorySplit,
) -> None:
    """Training and final labels cannot enter conflict calibration."""

    observations = list(conflict_observations())
    observations[0] = replace(
        observations[0],
        source_split=split,
    )

    with pytest.raises(
        EvidenceConflictError,
        match="validation observations only",
    ):
        calibrate_conflict_threshold(
            tuple(observations),
            maximum_false_conflict_rate=0.05,
        )


@pytest.mark.parametrize(
    "value",
    (
        -0.01,
        1.01,
        float("nan"),
    ),
)
def test_invalid_false_conflict_cap_is_rejected(
    value: float,
) -> None:
    """The registered conflict cap must remain finite in [0, 1]."""

    with pytest.raises(
        EvidenceConflictError,
        match="false-conflict",
    ):
        calibrate_conflict_threshold(
            conflict_observations(),
            maximum_false_conflict_rate=value,
        )


def test_observation_values_must_be_unit_interval() -> None:
    """Reliability and conflict inputs must remain finite probabilities."""

    with pytest.raises(
        EvidenceConflictError,
        match="between 0 and 1",
    ):
        ReliabilityCalibrationObservation(
            source_split=MultisensorySplit.VALIDATION,
            reliability=float("nan"),
            prediction_correct=True,
        )

    with pytest.raises(
        EvidenceConflictError,
        match="between 0 and 1",
    ):
        ConflictCalibrationObservation(
            source_split=MultisensorySplit.VALIDATION,
            conflict_score=1.1,
            conflict_present=True,
        )


def test_reports_contain_finite_thresholds() -> None:
    """Both validation-derived lock values must be finite."""

    reliability = calibrate_reliability_threshold(
        reliability_observations()
    )
    conflict = calibrate_conflict_threshold(
        conflict_observations(),
        maximum_false_conflict_rate=0.05,
    )

    assert math.isfinite(reliability.threshold)
    assert math.isfinite(conflict.threshold)
