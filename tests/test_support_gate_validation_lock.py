"""Tests for validation-only support-gate locking."""

from __future__ import annotations

import math

import pytest

from src.evaluation.multisensory_records import (
    MultisensorySplit,
)
from src.evaluation.noi_v0_3_generator import (
    NOIV03GenerationConfig,
    generate_noi_v0_3_events,
)
from src.evaluation.support_gate import (
    SupportGate,
    SupportGateError,
    SupportMethod,
)


def generated():
    """Return deterministic reduced data for lock mechanics."""

    return generate_noi_v0_3_events(
        NOIV03GenerationConfig(
            seed=1301,
            train_event_count=70,
            validation_event_count=100,
            final_test_event_count=20,
            validation_seen_item_count=40,
            validation_known_family_unseen_item_count=30,
            validation_unseen_family_count=30,
            final_seen_item_count=8,
            final_known_family_unseen_item_count=6,
            final_unseen_family_count=6,
            known_family_count=4,
            training_items_per_family=4,
            withheld_items_per_known_family=2,
            validation_unknown_family_count=2,
            final_unknown_family_count=2,
            items_per_unknown_family=3,
            generator_version="0.3.1-lock-test",
            feasibility_only=True,
        )
    )


def events_for(split: MultisensorySplit):
    """Return one split without exposing another split to calibration."""

    return tuple(
        event
        for event in generated().latent_events
        if event.split is split
    )


def fitted_gate(
    method: SupportMethod = SupportMethod.MAHALANOBIS,
) -> SupportGate:
    """Fit one gate using training records only."""

    gate = SupportGate(method=method)
    gate.fit(events_for(MultisensorySplit.TRAIN))

    return gate


@pytest.mark.parametrize(
    "method",
    tuple(SupportMethod),
)
def test_lock_calibration_enforces_false_known_constraint(
    method: SupportMethod,
) -> None:
    """The locked threshold must satisfy the registered five-percent cap."""

    gate = fitted_gate(method)

    report = gate.calibrate_for_lock(
        events_for(MultisensorySplit.VALIDATION),
        maximum_false_known_rate=0.05,
        bootstrap_seed=4242,
        bootstrap_resamples=200,
        confidence_level=0.95,
    )

    assert report.validation_false_known_rate <= 0.05
    assert report.maximum_false_known_rate == 0.05
    assert report.source_split is MultisensorySplit.VALIDATION
    assert report.final_test_labels_used is False


def test_lock_calibration_produces_ordered_finite_interval() -> None:
    """Bootstrap bounds must be finite, ordered, and contain the threshold."""

    gate = fitted_gate()

    report = gate.calibrate_for_lock(
        events_for(MultisensorySplit.VALIDATION),
        maximum_false_known_rate=0.05,
        bootstrap_seed=4242,
        bootstrap_resamples=200,
        confidence_level=0.95,
    )

    assert math.isfinite(report.threshold)
    assert math.isfinite(report.uncertainty_lower)
    assert math.isfinite(report.uncertainty_upper)
    assert (
        report.uncertainty_lower
        <= report.threshold
        <= report.uncertainty_upper
    )


def test_lock_calibration_is_deterministic() -> None:
    """Identical validation records and bootstrap controls reproduce exactly."""

    validation = events_for(MultisensorySplit.VALIDATION)
    first = fitted_gate().calibrate_for_lock(
        validation,
        maximum_false_known_rate=0.05,
        bootstrap_seed=4242,
        bootstrap_resamples=200,
        confidence_level=0.95,
    )
    second = fitted_gate().calibrate_for_lock(
        validation,
        maximum_false_known_rate=0.05,
        bootstrap_seed=4242,
        bootstrap_resamples=200,
        confidence_level=0.95,
    )

    assert first == second


def test_lock_calibration_rejects_final_test_events() -> None:
    """Final-test labels cannot enter threshold or interval derivation."""

    gate = fitted_gate()

    with pytest.raises(
        SupportGateError,
        match="validation data only",
    ):
        gate.calibrate_for_lock(
            events_for(MultisensorySplit.FINAL_TEST),
            maximum_false_known_rate=0.05,
            bootstrap_seed=4242,
            bootstrap_resamples=200,
            confidence_level=0.95,
        )


def test_lock_decision_uses_direct_bootstrap_bounds() -> None:
    """Touch requests use the locked interval rather than a pilot width."""

    gate = fitted_gate()
    report = gate.calibrate_for_lock(
        events_for(MultisensorySplit.VALIDATION),
        maximum_false_known_rate=0.05,
        bootstrap_seed=4242,
        bootstrap_resamples=200,
        confidence_level=0.95,
    )

    inside = gate.decide_from_score(
        event_id="inside",
        support_score=report.threshold,
    )
    below = gate.decide_from_score(
        event_id="below",
        support_score=(
            report.uncertainty_lower
            - max(1.0, abs(report.uncertainty_lower))
        ),
    )
    above = gate.decide_from_score(
        event_id="above",
        support_score=(
            report.uncertainty_upper
            + max(1.0, abs(report.uncertainty_upper))
        ),
    )

    assert inside.request_touch is True
    assert below.request_touch is False
    assert above.request_touch is False


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("maximum_false_known_rate", -0.01, "false-known"),
        ("maximum_false_known_rate", 1.01, "false-known"),
        ("bootstrap_seed", -1, "bootstrap_seed"),
        ("bootstrap_resamples", 0, "bootstrap_resamples"),
        ("confidence_level", 1.0, "confidence_level"),
    ),
)
def test_invalid_lock_controls_are_rejected(
    keyword: str,
    value: float | int,
    message: str,
) -> None:
    """Lock controls must remain finite and inside registered domains."""

    arguments = {
        "maximum_false_known_rate": 0.05,
        "bootstrap_seed": 4242,
        "bootstrap_resamples": 200,
        "confidence_level": 0.95,
    }
    arguments[keyword] = value

    with pytest.raises(
        SupportGateError,
        match=message,
    ):
        fitted_gate().calibrate_for_lock(
            events_for(MultisensorySplit.VALIDATION),
            **arguments,
        )
