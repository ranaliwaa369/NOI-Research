"""Tests for synthetic-dataset leakage auditing."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.evaluation.leakage_audit import (
    LeakageAuditError,
    LeakageAuditReport,
    audit_synthetic_dataset,
)
from src.evaluation.synthetic_generator import generate_synthetic_pilot
from src.evaluation.synthetic_records import SplitLabel


SYNTHETIC_CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"


@pytest.fixture
def pilot_dataset():
    """Return a deterministic synthetic pilot dataset."""

    return generate_synthetic_pilot(
        SYNTHETIC_CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )


def test_generated_pilot_passes_leakage_audit(
    pilot_dataset,
) -> None:
    """The independent pilot generator must satisfy all safeguards."""

    report = audit_synthetic_dataset(pilot_dataset)

    assert report.passed is True
    assert report.event_count == 200
    assert report.target_count == 200


def test_generated_pilot_has_expected_split_counts(
    pilot_dataset,
) -> None:
    """The report must preserve the preregistered pilot split sizes."""

    report = audit_synthetic_dataset(pilot_dataset)

    counts = dict(report.split_counts)

    assert counts["train"] == 140
    assert counts["validation"] == 20
    assert counts["ood_test"] == 40


def test_generated_pilot_has_no_detected_leakage(
    pilot_dataset,
) -> None:
    """Every reported leakage collection must be empty."""

    report = audit_synthetic_dataset(pilot_dataset)

    assert report.duplicate_event_ids == ()
    assert report.cross_split_feature_duplicates == ()
    assert report.ood_family_overlap == ()
    assert report.template_overlap == ()
    assert report.inconsistent_target_families == ()
    assert report.missing_splits == ()


def test_report_is_immutable(
    pilot_dataset,
) -> None:
    """Audit results must not be silently modified after creation."""

    report = audit_synthetic_dataset(pilot_dataset)

    with pytest.raises(FrozenInstanceError):
        report.passed = False  # type: ignore[misc]


def test_empty_required_splits_are_rejected(
    pilot_dataset,
) -> None:
    """An audit must define at least one required evaluation split."""

    with pytest.raises(
        LeakageAuditError,
        match="At least one required split",
    ):
        audit_synthetic_dataset(
            pilot_dataset,
            required_splits=(),
        )


def test_duplicate_required_splits_are_rejected(
    pilot_dataset,
) -> None:
    """Repeated split requirements must be rejected."""

    with pytest.raises(
        LeakageAuditError,
        match="cannot contain duplicates",
    ):
        audit_synthetic_dataset(
            pilot_dataset,
            required_splits=(
                SplitLabel.TRAIN,
                SplitLabel.TRAIN,
            ),
        )


def test_report_require_pass_accepts_clean_report(
    pilot_dataset,
) -> None:
    """A passing report must not raise an exception."""

    report = audit_synthetic_dataset(
        pilot_dataset,
        raise_on_failure=False,
    )

    report.require_pass()


def test_failed_report_raises_descriptive_error() -> None:
    """A failed report must identify the violated safeguard."""

    report = LeakageAuditReport(
        event_count=1,
        target_count=1,
        split_counts=(("train", 1),),
        duplicate_event_ids=("event-001",),
        cross_split_feature_duplicates=(),
        ood_family_overlap=(),
        template_overlap=(),
        inconsistent_target_families=(),
        missing_splits=("ood_test",),
        passed=False,
    )

    with pytest.raises(
        LeakageAuditError,
        match="duplicate event identifiers",
    ):
        report.require_pass()