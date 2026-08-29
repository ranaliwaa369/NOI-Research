"""Tests for applying frozen support thresholds at inference."""

import inspect
import math

import pytest

from src.evaluation.support_gate import (
    SupportGateError,
    SupportMethod,
    UncertaintyStatus,
    apply_locked_support_threshold,
)


def test_score_below_lower_is_certainly_unsupported() -> None:
    decision = apply_locked_support_threshold(
        event_id="event-low",
        method=SupportMethod.MAHALANOBIS,
        support_score=-3.0,
        threshold=-2.0,
        uncertainty_lower=-2.5,
        uncertainty_upper=-1.5,
    )

    assert decision.is_supported is False
    assert decision.uncertainty_status is (
        UncertaintyStatus.CERTAIN_UNSUPPORTED
    )
    assert decision.request_touch is False


def test_score_inside_band_is_uncertain() -> None:
    decision = apply_locked_support_threshold(
        event_id="event-middle",
        method=SupportMethod.MAHALANOBIS,
        support_score=-2.0,
        threshold=-2.0,
        uncertainty_lower=-2.5,
        uncertainty_upper=-1.5,
    )

    assert decision.is_supported is True
    assert decision.uncertainty_status is (
        UncertaintyStatus.UNCERTAIN
    )
    assert decision.request_touch is True


def test_score_above_upper_is_certainly_supported() -> None:
    decision = apply_locked_support_threshold(
        event_id="event-high",
        method=SupportMethod.MAHALANOBIS,
        support_score=-1.0,
        threshold=-2.0,
        uncertainty_lower=-2.5,
        uncertainty_upper=-1.5,
    )

    assert decision.is_supported is True
    assert decision.uncertainty_status is (
        UncertaintyStatus.CERTAIN_SUPPORTED
    )
    assert decision.request_touch is False


def test_inclusive_band_boundaries_are_uncertain() -> None:
    for score in (-2.5, -1.5):
        decision = apply_locked_support_threshold(
            event_id=f"event-{score}",
            method=SupportMethod.MAHALANOBIS,
            support_score=score,
            threshold=-2.0,
            uncertainty_lower=-2.5,
            uncertainty_upper=-1.5,
        )

        assert decision.uncertainty_status is (
            UncertaintyStatus.UNCERTAIN
        )
        assert decision.request_touch is True


@pytest.mark.parametrize(
    "field",
    (
        "support_score",
        "threshold",
        "uncertainty_lower",
        "uncertainty_upper",
    ),
)
def test_nonfinite_lock_value_is_rejected(
    field: str,
) -> None:
    arguments = {
        "event_id": "bad",
        "method": SupportMethod.MAHALANOBIS,
        "support_score": -2.0,
        "threshold": -2.0,
        "uncertainty_lower": -2.5,
        "uncertainty_upper": -1.5,
    }
    arguments[field] = float("nan")

    with pytest.raises(
        SupportGateError,
        match="finite",
    ):
        apply_locked_support_threshold(**arguments)


def test_invalid_band_order_is_rejected() -> None:
    with pytest.raises(
        SupportGateError,
        match="lower.*threshold.*upper",
    ):
        apply_locked_support_threshold(
            event_id="bad-order",
            method=SupportMethod.MAHALANOBIS,
            support_score=0.0,
            threshold=0.0,
            uncertainty_lower=1.0,
            uncertainty_upper=-1.0,
        )


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(
        SupportGateError,
        match="SupportMethod",
    ):
        apply_locked_support_threshold(
            event_id="bad-method",
            method="mahalanobis",  # type: ignore[arg-type]
            support_score=0.0,
            threshold=0.0,
            uncertainty_lower=0.0,
            uncertainty_upper=0.0,
        )


def test_application_contains_no_labels_or_conditions() -> None:
    parameters = set(
        inspect.signature(
            apply_locked_support_threshold
        ).parameters
    )

    assert "target_item_id" not in parameters
    assert "target_family_id" not in parameters
    assert "support_regime" not in parameters
    assert "condition" not in parameters
    assert "final_test_label" not in parameters


def test_output_values_are_finite() -> None:
    decision = apply_locked_support_threshold(
        event_id="finite",
        method=SupportMethod.MAHALANOBIS,
        support_score=-2.0,
        threshold=-2.0,
        uncertainty_lower=-2.0,
        uncertainty_upper=-2.0,
    )

    assert math.isfinite(decision.support_score)
    assert math.isfinite(decision.threshold)
