"""Tests for synthetic NOI distribution-shift diagnostics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.evaluation.distribution_diagnostics import (
    DistributionDiagnosticError,
    DistributionShiftReport,
    analyze_distribution_shift,
)
from src.evaluation.synthetic_generator import generate_synthetic_pilot


SYNTHETIC_CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"


@pytest.fixture(scope="module")
def pilot_dataset():
    """Create the deterministic 200-event pilot dataset."""

    return generate_synthetic_pilot(
        SYNTHETIC_CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )


@pytest.fixture(scope="module")
def diagnostic_report(
    pilot_dataset,
) -> DistributionShiftReport:
    """Compute the locked diagnostic report once."""

    return analyze_distribution_shift(
        pilot_dataset,
        oracle_alpha=1.0,
    )


def test_report_preserves_locked_split_counts(
    diagnostic_report: DistributionShiftReport,
) -> None:
    """The diagnostic must use 20 validation and 40 OOD events."""

    assert diagnostic_report.validation_count == 20
    assert diagnostic_report.ood_count == 40
    assert diagnostic_report.feature_dimension == 16


def test_ood_oracle_uses_disjoint_halves(
    diagnostic_report: DistributionShiftReport,
) -> None:
    """OOD calibration and evaluation must each contain 20 events."""

    assert diagnostic_report.ood_oracle_calibration_count == 20
    assert diagnostic_report.ood_oracle_evaluation_count == 20


def test_shift_statistics_are_in_valid_ranges(
    diagnostic_report: DistributionShiftReport,
) -> None:
    """All distribution-shift statistics must be valid."""

    assert diagnostic_report.mean_shift_l2 >= 0.0
    assert diagnostic_report.covariance_shift_frobenius >= 0.0
    assert diagnostic_report.rbf_mmd_squared >= 0.0

    assert (
        -1.0
        <= diagnostic_report.centroid_cosine_similarity
        <= 1.0
    )

    assert (
        0.0
        <= diagnostic_report.ood_oracle_mrr
        <= 1.0
    )


def test_diagnostic_is_deterministic(
    pilot_dataset,
    diagnostic_report: DistributionShiftReport,
) -> None:
    """Repeated diagnostics must produce identical results."""

    repeated = analyze_distribution_shift(
        pilot_dataset,
        oracle_alpha=1.0,
    )

    assert repeated == diagnostic_report


def test_report_is_immutable(
    diagnostic_report: DistributionShiftReport,
) -> None:
    """A completed diagnostic cannot be silently modified."""

    with pytest.raises(FrozenInstanceError):
        diagnostic_report.ood_oracle_mrr = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "invalid_alpha",
    (
        -1.0,
        float("inf"),
        float("nan"),
        True,
    ),
)
def test_invalid_oracle_alpha_is_rejected(
    pilot_dataset,
    invalid_alpha,
) -> None:
    """The oracle regularization value must be valid."""

    with pytest.raises(
        DistributionDiagnosticError,
        match="finite and nonnegative",
    ):
        analyze_distribution_shift(
            pilot_dataset,
            oracle_alpha=invalid_alpha,
        )


def test_invalid_dataset_type_is_rejected() -> None:
    """Diagnostics require a validated SyntheticDataset."""

    with pytest.raises(
        DistributionDiagnosticError,
        match="SyntheticDataset",
    ):
        analyze_distribution_shift(
            "invalid"  # type: ignore[arg-type]
        )


def test_invalid_report_metric_is_rejected() -> None:
    """A report cannot contain an impossible oracle MRR."""

    with pytest.raises(
        DistributionDiagnosticError,
        match="Oracle MRR",
    ):
        DistributionShiftReport(
            validation_count=20,
            ood_count=40,
            feature_dimension=16,
            mean_shift_l2=0.1,
            covariance_shift_frobenius=0.2,
            centroid_cosine_similarity=0.5,
            rbf_mmd_squared=0.1,
            ood_oracle_calibration_count=20,
            ood_oracle_evaluation_count=20,
            ood_oracle_mrr=1.5,
        )