"""Retrieval, open-set, calibration, risk, and touch metrics for NOI v0.3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

from src.evaluation.retrieval_metrics import (
    MetricInputError,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


class NOIV03MetricError(ValueError):
    """Raised when v0.3 metric inputs are invalid or misaligned."""


@dataclass(frozen=True, slots=True)
class RetrievalSummary:
    """Standard retrieval metrics reused from the existing evaluation layer."""

    event_count: int
    recall_at_1: float
    recall_at_10: float
    mean_reciprocal_rank: float
    ndcg_at_10: float


@dataclass(frozen=True, slots=True)
class OpenSetMetrics:
    """Binary supported-versus-unsupported evaluation metrics."""

    event_count: int
    supported_count: int
    unsupported_count: int
    true_supported_rate: float
    true_unsupported_rate: float
    false_known_rate: float
    false_unknown_rate: float
    balanced_accuracy: float
    auroc: float


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """Binary confidence calibration summary."""

    event_count: int
    bin_count: int
    brier_score: float
    expected_calibration_error: float


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    """Selective risk after retaining the highest-confidence events."""

    retained_count: int
    coverage: float
    risk: float
    minimum_retained_confidence: float


@dataclass(frozen=True, slots=True)
class RiskCoverageSummary:
    """Full deterministic risk–coverage curve and its mean area."""

    event_count: int
    points: tuple[RiskCoveragePoint, ...]
    area_under_risk_coverage: float


@dataclass(frozen=True, slots=True)
class TouchUtilityMetrics:
    """Touch request, utility, harm, and false-confidence metrics."""

    event_count: int
    touch_request_count: int
    touch_request_rate: float
    useful_touch_count: int
    useful_touch_rate: float
    mean_touch_synergy: float
    requested_touch_synergy: float
    harmful_fusion_count: int
    harmful_fusion_rate: float
    false_confident_count: int
    false_confident_rate: float


def _require_nonempty_equal_lengths(
    *sequences: Sequence[object],
) -> int:
    """Require aligned nonempty event sequences."""

    if not sequences:
        raise NOIV03MetricError(
            "At least one metric sequence is required."
        )

    lengths = tuple(len(sequence) for sequence in sequences)

    if not lengths or lengths[0] == 0:
        raise NOIV03MetricError(
            "Metric sequences cannot be empty."
        )

    if len(set(lengths)) != 1:
        raise NOIV03MetricError(
            "All metric sequences must have the same length."
        )

    return lengths[0]


def _validate_boolean_sequence(
    name: str,
    values: Sequence[object],
) -> tuple[bool, ...]:
    """Require explicit boolean labels."""

    if not all(isinstance(value, bool) for value in values):
        raise NOIV03MetricError(
            f"{name} must contain only boolean values."
        )

    return tuple(values)


def _validate_finite_sequence(
    name: str,
    values: Sequence[object],
) -> tuple[float, ...]:
    """Require finite numeric values."""

    converted: list[float] = []

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise NOIV03MetricError(
                f"{name} must contain only finite numeric values."
            )

        converted.append(float(value))

    return tuple(converted)


def _validate_probability_sequence(
    name: str,
    values: Sequence[object],
) -> tuple[float, ...]:
    """Require finite probabilities in [0, 1]."""

    converted: list[float] = []

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise NOIV03MetricError(
                f"{name} values must be finite and between 0 and 1."
            )

        converted.append(float(value))

    return tuple(converted)


def retrieval_summary(
    rankings: Sequence[Sequence[str]],
    relevant_items: Sequence[frozenset[str]],
) -> RetrievalSummary:
    """Compute the established project retrieval metrics."""

    try:
        event_count = _require_nonempty_equal_lengths(
            rankings,
            relevant_items,
        )

        recall_1 = recall_at_k(
            rankings,
            relevant_items,
            k=1,
        )
        recall_10 = recall_at_k(
            rankings,
            relevant_items,
            k=10,
        )
        mrr = mean_reciprocal_rank(
            rankings,
            relevant_items,
        )
        ndcg_10 = ndcg_at_k(
            rankings,
            relevant_items,
            k=10,
        )
    except (MetricInputError, TypeError, ValueError) as error:
        raise NOIV03MetricError(
            f"Invalid retrieval metric inputs: {error}"
        ) from error

    return RetrievalSummary(
        event_count=event_count,
        recall_at_1=float(recall_1),
        recall_at_10=float(recall_10),
        mean_reciprocal_rank=float(mrr),
        ndcg_at_10=float(ndcg_10),
    )


def _binary_auroc(
    labels: tuple[bool, ...],
    scores: tuple[float, ...],
) -> float:
    """Compute AUROC using all positive-negative score pairs."""

    positive_scores = tuple(
        score
        for label, score in zip(
            labels,
            scores,
            strict=True,
        )
        if label
    )
    negative_scores = tuple(
        score
        for label, score in zip(
            labels,
            scores,
            strict=True,
        )
        if not label
    )

    wins = 0.0

    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5

    return wins / (
        len(positive_scores)
        * len(negative_scores)
    )


def open_set_metrics(
    *,
    true_supported: Sequence[bool],
    predicted_supported: Sequence[bool],
    support_scores: Sequence[float],
) -> OpenSetMetrics:
    """Evaluate supported-versus-unsupported decisions."""

    event_count = _require_nonempty_equal_lengths(
        true_supported,
        predicted_supported,
        support_scores,
    )
    truth = _validate_boolean_sequence(
        "true_supported",
        true_supported,
    )
    predictions = _validate_boolean_sequence(
        "predicted_supported",
        predicted_supported,
    )
    scores = _validate_finite_sequence(
        "support_scores",
        support_scores,
    )

    supported_count = sum(truth)
    unsupported_count = event_count - supported_count

    if supported_count == 0 or unsupported_count == 0:
        raise NOIV03MetricError(
            "Open-set metrics require both supported and unsupported "
            "events."
        )

    true_supported_count = sum(
        actual and predicted
        for actual, predicted in zip(
            truth,
            predictions,
            strict=True,
        )
    )
    true_unsupported_count = sum(
        (not actual) and (not predicted)
        for actual, predicted in zip(
            truth,
            predictions,
            strict=True,
        )
    )
    false_known_count = sum(
        (not actual) and predicted
        for actual, predicted in zip(
            truth,
            predictions,
            strict=True,
        )
    )
    false_unknown_count = sum(
        actual and (not predicted)
        for actual, predicted in zip(
            truth,
            predictions,
            strict=True,
        )
    )

    true_supported_rate = (
        true_supported_count / supported_count
    )
    true_unsupported_rate = (
        true_unsupported_count / unsupported_count
    )

    return OpenSetMetrics(
        event_count=event_count,
        supported_count=supported_count,
        unsupported_count=unsupported_count,
        true_supported_rate=true_supported_rate,
        true_unsupported_rate=true_unsupported_rate,
        false_known_rate=(
            false_known_count / unsupported_count
        ),
        false_unknown_rate=(
            false_unknown_count / supported_count
        ),
        balanced_accuracy=(
            true_supported_rate + true_unsupported_rate
        ) / 2.0,
        auroc=_binary_auroc(
            truth,
            scores,
        ),
    )


def calibration_metrics(
    *,
    confidences: Sequence[float],
    correctness: Sequence[bool],
    bin_count: int,
) -> CalibrationMetrics:
    """Compute Brier score and equal-width expected calibration error."""

    event_count = _require_nonempty_equal_lengths(
        confidences,
        correctness,
    )

    if (
        isinstance(bin_count, bool)
        or not isinstance(bin_count, int)
        or bin_count <= 0
    ):
        raise NOIV03MetricError(
            "bin_count must be a positive integer."
        )

    probabilities = _validate_probability_sequence(
        "confidence",
        confidences,
    )
    labels = _validate_boolean_sequence(
        "correctness",
        correctness,
    )

    targets = tuple(
        1.0 if label else 0.0
        for label in labels
    )
    brier_score = sum(
        (probability - target) ** 2
        for probability, target in zip(
            probabilities,
            targets,
            strict=True,
        )
    ) / event_count

    bins: list[list[int]] = [
        []
        for _ in range(bin_count)
    ]

    for index, probability in enumerate(probabilities):
        bin_index = min(
            int(probability * bin_count),
            bin_count - 1,
        )
        bins[bin_index].append(index)

    expected_calibration_error = 0.0

    for indices in bins:
        if not indices:
            continue

        mean_confidence = sum(
            probabilities[index]
            for index in indices
        ) / len(indices)
        mean_accuracy = sum(
            targets[index]
            for index in indices
        ) / len(indices)

        expected_calibration_error += (
            len(indices)
            / event_count
            * abs(mean_confidence - mean_accuracy)
        )

    return CalibrationMetrics(
        event_count=event_count,
        bin_count=bin_count,
        brier_score=brier_score,
        expected_calibration_error=(
            expected_calibration_error
        ),
    )


def risk_coverage_summary(
    *,
    confidences: Sequence[float],
    correctness: Sequence[bool],
) -> RiskCoverageSummary:
    """Compute selective risk while retaining highest-confidence events."""

    event_count = _require_nonempty_equal_lengths(
        confidences,
        correctness,
    )
    probabilities = _validate_probability_sequence(
        "confidence",
        confidences,
    )
    labels = _validate_boolean_sequence(
        "correctness",
        correctness,
    )

    ordered_indices = tuple(
        sorted(
            range(event_count),
            key=lambda index: (
                -probabilities[index],
                index,
            ),
        ),
    )

    points: list[RiskCoveragePoint] = []
    error_count = 0

    for retained_count, index in enumerate(
        ordered_indices,
        start=1,
    ):
        if not labels[index]:
            error_count += 1

        points.append(
            RiskCoveragePoint(
                retained_count=retained_count,
                coverage=retained_count / event_count,
                risk=error_count / retained_count,
                minimum_retained_confidence=(
                    probabilities[index]
                ),
            ),
        )

    area = sum(
        point.risk
        for point in points
    ) / len(points)

    return RiskCoverageSummary(
        event_count=event_count,
        points=tuple(points),
        area_under_risk_coverage=area,
    )


def touch_utility_metrics(
    *,
    touch_requested: Sequence[bool],
    odor_only_reciprocal_rank: Sequence[float],
    fused_reciprocal_rank: Sequence[float],
    final_correct: Sequence[bool],
    final_confidence: Sequence[float],
    abstained: Sequence[bool],
    false_confident_threshold: float,
) -> TouchUtilityMetrics:
    """Compute touch utility, harmful fusion, and false-confidence rates."""

    event_count = _require_nonempty_equal_lengths(
        touch_requested,
        odor_only_reciprocal_rank,
        fused_reciprocal_rank,
        final_correct,
        final_confidence,
        abstained,
    )

    requests = _validate_boolean_sequence(
        "touch_requested",
        touch_requested,
    )
    odor_scores = _validate_probability_sequence(
        "odor_only_reciprocal_rank",
        odor_only_reciprocal_rank,
    )
    fused_scores = _validate_probability_sequence(
        "fused_reciprocal_rank",
        fused_reciprocal_rank,
    )
    correctness = _validate_boolean_sequence(
        "final_correct",
        final_correct,
    )
    confidences = _validate_probability_sequence(
        "final_confidence",
        final_confidence,
    )
    abstentions = _validate_boolean_sequence(
        "abstained",
        abstained,
    )

    if (
        isinstance(false_confident_threshold, bool)
        or not isinstance(false_confident_threshold, Real)
        or not math.isfinite(
            float(false_confident_threshold),
        )
        or not 0.0 <= float(false_confident_threshold) <= 1.0
    ):
        raise NOIV03MetricError(
            "false_confident_threshold must be finite and "
            "between 0 and 1."
        )

    threshold = float(false_confident_threshold)
    differences = tuple(
        fused - odor
        for odor, fused in zip(
            odor_scores,
            fused_scores,
            strict=True,
        )
    )
    request_indices = tuple(
        index
        for index, requested in enumerate(requests)
        if requested
    )

    touch_request_count = len(request_indices)
    useful_touch_count = sum(
        differences[index] > 0.0
        for index in request_indices
    )
    harmful_fusion_count = sum(
        difference < 0.0
        for difference in differences
    )
    false_confident_count = sum(
        (not correct)
        and confidence >= threshold
        and (not abstain)
        for correct, confidence, abstain in zip(
            correctness,
            confidences,
            abstentions,
            strict=True,
        )
    )

    requested_synergy = (
        sum(
            differences[index]
            for index in request_indices
        )
        / touch_request_count
        if touch_request_count
        else 0.0
    )

    return TouchUtilityMetrics(
        event_count=event_count,
        touch_request_count=touch_request_count,
        touch_request_rate=(
            touch_request_count / event_count
        ),
        useful_touch_count=useful_touch_count,
        useful_touch_rate=(
            useful_touch_count / touch_request_count
            if touch_request_count
            else 0.0
        ),
        mean_touch_synergy=(
            sum(differences) / event_count
        ),
        requested_touch_synergy=requested_synergy,
        harmful_fusion_count=harmful_fusion_count,
        harmful_fusion_rate=(
            harmful_fusion_count / event_count
        ),
        false_confident_count=false_confident_count,
        false_confident_rate=(
            false_confident_count / event_count
        ),
    )
