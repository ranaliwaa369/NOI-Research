"""Tests for repeated OOD-oracle diagnostics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.evaluation.repeated_ood_diagnostics import (
    RepeatedOODDiagnosticError,
    RepeatedOODReport,
    run_repeated_ood_diagnostics,
)
from src.evaluation.synthetic_generator import generate_synthetic_pilot


SYNTHETIC_CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"


@pytest.fixture(scope="module")
def pilot_dataset():
    return generate_synthetic_pilot(
        SYNTHETIC_CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )


@pytest.fixture(scope="module")
def report(pilot_dataset) -> RepeatedOODReport:
    return run_repeated_ood_diagnostics(
        pilot_dataset,
        split_seeds=range(10),
        bootstrap_seed=4242,
        bootstrap_resamples=1000,
    )


def test_report_contains_ten_runs(
    report: RepeatedOODReport,
) -> None:
    assert len(report.runs) == 10
    assert tuple(run.seed for run in report.runs) == tuple(range(10))


def test_each_run_uses_disjoint_equal_halves(
    report: RepeatedOODReport,
) -> None:
    for run in report.runs:
        calibration = set(run.calibration_event_ids)
        evaluation = set(run.evaluation_event_ids)

        assert len(calibration) == 20
        assert len(evaluation) == 20
        assert calibration.isdisjoint(evaluation)
        assert len(calibration | evaluation) == 40


def test_aggregate_statistics_are_valid(
    report: RepeatedOODReport,
) -> None:
    assert 0.0 <= report.minimum_mrr <= report.mean_mrr
    assert report.mean_mrr <= report.maximum_mrr <= 1.0
    assert report.standard_deviation_mrr >= 0.0
    assert report.bootstrap_ci_lower <= report.mean_mrr
    assert report.mean_mrr <= report.bootstrap_ci_upper


def test_repeated_diagnostic_is_deterministic(
    pilot_dataset,
    report: RepeatedOODReport,
) -> None:
    repeated = run_repeated_ood_diagnostics(
        pilot_dataset,
        split_seeds=range(10),
        bootstrap_seed=4242,
        bootstrap_resamples=1000,
    )

    assert repeated == report


def test_report_is_immutable(
    report: RepeatedOODReport,
) -> None:
    with pytest.raises(FrozenInstanceError):
        report.mean_mrr = 1.0  # type: ignore[misc]


def test_different_split_seeds_change_runs(
    pilot_dataset,
    report: RepeatedOODReport,
) -> None:
    different = run_repeated_ood_diagnostics(
        pilot_dataset,
        split_seeds=range(10, 20),
        bootstrap_seed=4242,
        bootstrap_resamples=1000,
    )

    assert different.runs != report.runs


def test_duplicate_split_seeds_are_rejected(
    pilot_dataset,
) -> None:
    with pytest.raises(
        RepeatedOODDiagnosticError,
        match="unique split seeds",
    ):
        run_repeated_ood_diagnostics(
            pilot_dataset,
            split_seeds=(1, 1),
            bootstrap_resamples=100,
        )


@pytest.mark.parametrize(
    "invalid_fraction",
    (0.0, 1.0, -0.5, 1.5),
)
def test_invalid_calibration_fraction_is_rejected(
    pilot_dataset,
    invalid_fraction,
) -> None:
    with pytest.raises(
        RepeatedOODDiagnosticError,
        match="between 0 and 1",
    ):
        run_repeated_ood_diagnostics(
            pilot_dataset,
            calibration_fraction=invalid_fraction,
            bootstrap_resamples=100,
        )


def test_too_few_bootstrap_resamples_are_rejected(
    pilot_dataset,
) -> None:
    with pytest.raises(
        RepeatedOODDiagnosticError,
        match="at least 100",
    ):
        run_repeated_ood_diagnostics(
            pilot_dataset,
            bootstrap_resamples=99,
        )