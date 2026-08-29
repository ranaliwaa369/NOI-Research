"""Tests for NOI v0.3 retrieval, safety, and touch metrics."""

from __future__ import annotations

import math

import pytest

from src.evaluation.noi_v0_3_metrics import (
    CalibrationMetrics,
    NOIV03MetricError,
    OpenSetMetrics,
    RetrievalSummary,
    RiskCoveragePoint,
    RiskCoverageSummary,
    TouchUtilityMetrics,
    calibration_metrics,
    open_set_metrics,
    retrieval_summary,
    risk_coverage_summary,
    touch_utility_metrics,
)


def test_retrieval_summary_reuses_standard_rank_metrics() -> None:
    """Recall, MRR, and nDCG retain the established project definitions."""

    rankings = (
        ("a", "b", "c"),
        ("x", "b", "a"),
        ("x", "y", "c"),
    )
    relevant = (
        frozenset(("a",)),
        frozenset(("a",)),
        frozenset(("c",)),
    )

    result = retrieval_summary(
        rankings,
        relevant,
    )

    assert isinstance(result, RetrievalSummary)
    assert result.event_count == 3
    assert result.recall_at_1 == pytest.approx(1 / 3)
    assert result.recall_at_10 == 1.0
    assert result.mean_reciprocal_rank == pytest.approx(
        (1.0 + 1 / 3 + 1 / 3) / 3,
    )
    assert 0.0 <= result.ndcg_at_10 <= 1.0


def test_retrieval_summary_rejects_misaligned_inputs() -> None:
    """Rankings and relevance sets remain event-aligned."""

    with pytest.raises(NOIV03MetricError):
        retrieval_summary(
            (("a",),),
            (
                frozenset(("a",)),
                frozenset(("b",)),
            ),
        )


def test_open_set_metrics_are_exact_for_perfect_predictions() -> None:
    """A perfect support detector receives ideal metrics."""

    result = open_set_metrics(
        true_supported=(True, True, False, False),
        predicted_supported=(True, True, False, False),
        support_scores=(0.9, 0.8, 0.2, 0.1),
    )

    assert isinstance(result, OpenSetMetrics)
    assert result.event_count == 4
    assert result.supported_count == 2
    assert result.unsupported_count == 2
    assert result.true_supported_rate == 1.0
    assert result.true_unsupported_rate == 1.0
    assert result.false_known_rate == 0.0
    assert result.false_unknown_rate == 0.0
    assert result.balanced_accuracy == 1.0
    assert result.auroc == 1.0


def test_open_set_false_known_and_false_unknown_rates() -> None:
    """Both directions of support error are reported separately."""

    result = open_set_metrics(
        true_supported=(True, True, False, False),
        predicted_supported=(True, False, True, False),
        support_scores=(0.8, 0.4, 0.7, 0.2),
    )

    assert result.true_supported_rate == 0.5
    assert result.true_unsupported_rate == 0.5
    assert result.false_known_rate == 0.5
    assert result.false_unknown_rate == 0.5
    assert result.balanced_accuracy == 0.5
    assert result.auroc == pytest.approx(0.75)


def test_open_set_requires_both_support_classes() -> None:
    """AUROC and balanced accuracy need supported and unsupported events."""

    with pytest.raises(
        NOIV03MetricError,
        match="both supported and unsupported",
    ):
        open_set_metrics(
            true_supported=(True, True),
            predicted_supported=(True, False),
            support_scores=(0.9, 0.4),
        )


def test_open_set_rejects_nonfinite_scores() -> None:
    """Support scores used in AUROC must be finite."""

    with pytest.raises(
        NOIV03MetricError,
        match="finite",
    ):
        open_set_metrics(
            true_supported=(True, False),
            predicted_supported=(True, False),
            support_scores=(float("nan"), 0.1),
        )


def test_calibration_metrics_are_exact_for_perfect_confidence() -> None:
    """Perfect binary predictions have zero Brier score and ECE."""

    result = calibration_metrics(
        confidences=(1.0, 0.0, 1.0, 0.0),
        correctness=(True, False, True, False),
        bin_count=4,
    )

    assert isinstance(result, CalibrationMetrics)
    assert result.event_count == 4
    assert result.bin_count == 4
    assert result.brier_score == 0.0
    assert result.expected_calibration_error == 0.0


def test_calibration_metrics_match_simple_manual_case() -> None:
    """Brier score follows mean squared probability error."""

    result = calibration_metrics(
        confidences=(0.8, 0.6, 0.4, 0.2),
        correctness=(True, False, True, False),
        bin_count=2,
    )

    expected_brier = (
        (0.8 - 1.0) ** 2
        + (0.6 - 0.0) ** 2
        + (0.4 - 1.0) ** 2
        + (0.2 - 0.0) ** 2
    ) / 4

    assert result.brier_score == pytest.approx(expected_brier)
    assert 0.0 <= result.expected_calibration_error <= 1.0


@pytest.mark.parametrize(
    "confidence",
    (
        -0.1,
        1.1,
        float("nan"),
    ),
)
def test_invalid_confidence_is_rejected(
    confidence: float,
) -> None:
    """Confidence probabilities must remain finite in [0, 1]."""

    with pytest.raises(
        NOIV03MetricError,
        match="between 0 and 1",
    ):
        calibration_metrics(
            confidences=(confidence,),
            correctness=(True,),
            bin_count=2,
        )


@pytest.mark.parametrize(
    "bin_count",
    (
        0,
        -1,
        True,
    ),
)
def test_invalid_bin_count_is_rejected(
    bin_count: object,
) -> None:
    """Calibration bin count must be a positive integer."""

    with pytest.raises(NOIV03MetricError):
        calibration_metrics(
            confidences=(0.5,),
            correctness=(True,),
            bin_count=bin_count,  # type: ignore[arg-type]
        )


def test_risk_coverage_curve_is_confidence_ordered() -> None:
    """Selective risk is computed as lower-confidence cases are removed."""

    result = risk_coverage_summary(
        confidences=(0.9, 0.8, 0.2, 0.1),
        correctness=(True, True, False, False),
    )

    assert isinstance(result, RiskCoverageSummary)
    assert result.event_count == 4
    assert len(result.points) == 4
    assert all(
        isinstance(point, RiskCoveragePoint)
        for point in result.points
    )

    assert result.points[0].coverage == pytest.approx(0.25)
    assert result.points[0].risk == 0.0
    assert result.points[1].coverage == pytest.approx(0.50)
    assert result.points[1].risk == 0.0
    assert result.points[-1].coverage == 1.0
    assert result.points[-1].risk == 0.5
    assert 0.0 <= result.area_under_risk_coverage <= 1.0


def test_risk_coverage_is_deterministic_under_ties() -> None:
    """Equal confidences retain stable original-index ordering."""

    first = risk_coverage_summary(
        confidences=(0.5, 0.5, 0.5),
        correctness=(True, False, True),
    )
    second = risk_coverage_summary(
        confidences=(0.5, 0.5, 0.5),
        correctness=(True, False, True),
    )

    assert first == second


def test_touch_metrics_report_request_and_synergy() -> None:
    """Requested touch utility is measured relative to odor-only retrieval."""

    result = touch_utility_metrics(
        touch_requested=(True, True, False, False),
        odor_only_reciprocal_rank=(0.2, 0.8, 1.0, 0.5),
        fused_reciprocal_rank=(0.7, 0.4, 1.0, 0.5),
        final_correct=(True, False, True, True),
        final_confidence=(0.9, 0.9, 0.9, 0.9),
        abstained=(False, False, False, False),
        false_confident_threshold=0.8,
    )

    assert isinstance(result, TouchUtilityMetrics)
    assert result.event_count == 4
    assert result.touch_request_count == 2
    assert result.touch_request_rate == 0.5
    assert result.useful_touch_count == 1
    assert result.useful_touch_rate == 0.5
    assert result.mean_touch_synergy == pytest.approx(0.025)
    assert result.requested_touch_synergy == pytest.approx(0.05)
    assert result.harmful_fusion_count == 1
    assert result.harmful_fusion_rate == 0.25
    assert result.false_confident_count == 1
    assert result.false_confident_rate == 0.25


def test_false_confident_excludes_abstentions() -> None:
    """An abstained event is not counted as a confident wrong prediction."""

    result = touch_utility_metrics(
        touch_requested=(False, False),
        odor_only_reciprocal_rank=(0.0, 0.0),
        fused_reciprocal_rank=(0.0, 0.0),
        final_correct=(False, False),
        final_confidence=(0.95, 0.95),
        abstained=(True, False),
        false_confident_threshold=0.8,
    )

    assert result.false_confident_count == 1
    assert result.false_confident_rate == 0.5


def test_no_touch_requests_have_zero_requested_synergy() -> None:
    """An empty requested-touch subset has a defined zero summary."""

    result = touch_utility_metrics(
        touch_requested=(False, False),
        odor_only_reciprocal_rank=(0.2, 0.5),
        fused_reciprocal_rank=(0.2, 0.5),
        final_correct=(True, True),
        final_confidence=(0.8, 0.8),
        abstained=(False, False),
        false_confident_threshold=0.8,
    )

    assert result.touch_request_count == 0
    assert result.touch_request_rate == 0.0
    assert result.useful_touch_count == 0
    assert result.useful_touch_rate == 0.0
    assert result.requested_touch_synergy == 0.0


def test_touch_metric_sequences_must_align() -> None:
    """Every metric input must refer to the same event count."""

    with pytest.raises(
        NOIV03MetricError,
        match="same length",
    ):
        touch_utility_metrics(
            touch_requested=(True,),
            odor_only_reciprocal_rank=(0.2, 0.3),
            fused_reciprocal_rank=(0.4,),
            final_correct=(True,),
            final_confidence=(0.8,),
            abstained=(False,),
            false_confident_threshold=0.8,
        )


@pytest.mark.parametrize(
    "value",
    (
        -0.1,
        1.1,
        float("nan"),
        True,
    ),
)
def test_invalid_false_confident_threshold_is_rejected(
    value: object,
) -> None:
    """The confidence cutoff must be finite in [0, 1]."""

    with pytest.raises(
        NOIV03MetricError,
        match="false_confident_threshold",
    ):
        touch_utility_metrics(
            touch_requested=(False,),
            odor_only_reciprocal_rank=(0.0,),
            fused_reciprocal_rank=(0.0,),
            final_correct=(False,),
            final_confidence=(0.5,),
            abstained=(False,),
            false_confident_threshold=value,  # type: ignore[arg-type]
        )


def test_all_metric_outputs_are_finite() -> None:
    """Published summaries cannot contain NaN or infinity."""

    retrieval = retrieval_summary(
        (("a",),),
        (frozenset(("a",)),),
    )
    open_set = open_set_metrics(
        true_supported=(True, False),
        predicted_supported=(True, False),
        support_scores=(0.8, 0.2),
    )
    calibration = calibration_metrics(
        confidences=(0.8, 0.2),
        correctness=(True, False),
        bin_count=2,
    )
    risk = risk_coverage_summary(
        confidences=(0.8, 0.2),
        correctness=(True, False),
    )

    values = (
        retrieval.recall_at_1,
        retrieval.recall_at_10,
        retrieval.mean_reciprocal_rank,
        retrieval.ndcg_at_10,
        open_set.auroc,
        open_set.balanced_accuracy,
        calibration.brier_score,
        calibration.expected_calibration_error,
        risk.area_under_risk_coverage,
    )

    assert all(math.isfinite(value) for value in values)
