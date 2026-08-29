"""Tests for the preregistered NOI v0.3 support gate."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from src.evaluation.multisensory_records import (
    LatentMultisensoryEvent,
    MultisensorySplit,
    SupportRegime,
)
from src.evaluation.noi_v0_3_generator import (
    NOIV03GenerationConfig,
    generate_noi_v0_3_events,
)
from src.evaluation.support_gate import (
    SupportCalibrationReport,
    SupportDecision,
    SupportGate,
    SupportGateError,
    SupportMethod,
    UncertaintyStatus,
)


def make_generated():
    """Return one feasibility dataset with all support regimes."""

    return generate_noi_v0_3_events(
        NOIV03GenerationConfig(
            seed=1301,
            train_event_count=70,
            validation_event_count=10,
            final_test_event_count=20,
            validation_seen_item_count=4,
            validation_known_family_unseen_item_count=3,
            validation_unseen_family_count=3,
            final_seen_item_count=8,
            final_known_family_unseen_item_count=6,
            final_unseen_family_count=6,
            known_family_count=4,
            training_items_per_family=4,
            withheld_items_per_known_family=2,
            validation_unknown_family_count=2,
            final_unknown_family_count=2,
            items_per_unknown_family=3,
            generator_version="0.3.0-feasibility",
            feasibility_only=True,
        ),
    )


def events_for(
    split: MultisensorySplit,
) -> tuple[LatentMultisensoryEvent, ...]:
    """Return generated latent events from one split."""

    return tuple(
        event
        for event in make_generated().latent_events
        if event.split is split
    )


@pytest.mark.parametrize(
    "method",
    tuple(SupportMethod),
)
def test_each_prespecified_method_fits_training_only(
    method: SupportMethod,
) -> None:
    """All three registered support methods fit the development records."""

    gate = SupportGate(method=method)
    training_events = events_for(MultisensorySplit.TRAIN)

    returned = gate.fit(training_events)

    assert returned is gate
    assert gate.is_fitted is True
    assert gate.training_event_count == 70
    assert gate.training_family_count == 4
    assert gate.threshold is None


def test_methods_are_exactly_the_preregistered_three() -> None:
    """No unregistered support method silently enters the experiment."""

    assert tuple(SupportMethod) == (
        SupportMethod.MAHALANOBIS,
        SupportMethod.COSINE_MARGIN,
        SupportMethod.NEAREST_PROTOTYPE_DISTANCE,
    )


@pytest.mark.parametrize(
    "method",
    tuple(SupportMethod),
)
def test_scores_are_finite_after_fit(
    method: SupportMethod,
) -> None:
    """Every method returns one finite higher-is-supported score."""

    gate = SupportGate(method=method)
    training_events = events_for(MultisensorySplit.TRAIN)
    gate.fit(training_events)

    score = gate.score(training_events[0].olfactory_vector)

    assert math.isfinite(score)


def test_fit_rejects_validation_events() -> None:
    """Validation observations cannot enter model fitting."""

    gate = SupportGate(
        method=SupportMethod.MAHALANOBIS,
    )

    with pytest.raises(
        SupportGateError,
        match="training data only",
    ):
        gate.fit(
            events_for(MultisensorySplit.VALIDATION),
        )


def test_fit_rejects_final_test_events() -> None:
    """Final-test observations cannot enter model fitting."""

    gate = SupportGate(
        method=SupportMethod.COSINE_MARGIN,
    )

    with pytest.raises(
        SupportGateError,
        match="training data only",
    ):
        gate.fit(
            events_for(MultisensorySplit.FINAL_TEST),
        )


def test_fit_rejects_non_development_support_labels() -> None:
    """Training fitting cannot consume evaluation support labels."""

    event = events_for(MultisensorySplit.TRAIN)[0]

    # Simulate a corrupted externally deserialized record. Normal record
    # construction already rejects this state; the gate must still defend
    # its own training boundary independently.
    object.__setattr__(
        event,
        "support_regime",
        SupportRegime.SEEN_ITEM,
    )

    gate = SupportGate(
        method=SupportMethod.NEAREST_PROTOTYPE_DISTANCE,
    )

    with pytest.raises(
        SupportGateError,
        match="development support",
    ):
        gate.fit((event,))


def test_score_before_fit_is_rejected() -> None:
    """Support scores cannot be produced by an unfitted model."""

    gate = SupportGate(
        method=SupportMethod.MAHALANOBIS,
    )

    with pytest.raises(
        SupportGateError,
        match="fitted",
    ):
        gate.score((1.0,) * 16)


def test_zero_query_is_rejected() -> None:
    """L2 normalization requires a nonzero query vector."""

    gate = SupportGate(
        method=SupportMethod.MAHALANOBIS,
    )
    gate.fit(events_for(MultisensorySplit.TRAIN))

    with pytest.raises(
        SupportGateError,
        match="nonzero",
    ):
        gate.score((0.0,) * 16)


def test_wrong_query_dimension_is_rejected() -> None:
    """Scoring must use the fitted olfactory dimension."""

    gate = SupportGate(
        method=SupportMethod.COSINE_MARGIN,
    )
    gate.fit(events_for(MultisensorySplit.TRAIN))

    with pytest.raises(
        SupportGateError,
        match="dimension",
    ):
        gate.score((1.0,) * 8)


@pytest.mark.parametrize(
    "method",
    tuple(SupportMethod),
)
def test_calibration_uses_validation_only(
    method: SupportMethod,
) -> None:
    """Threshold selection records validation as its sole source."""

    gate = SupportGate(method=method)
    gate.fit(events_for(MultisensorySplit.TRAIN))

    report = gate.calibrate(
        events_for(MultisensorySplit.VALIDATION),
        uncertainty_width=0.05,
    )

    assert isinstance(report, SupportCalibrationReport)
    assert report.source_split is MultisensorySplit.VALIDATION
    assert report.validation_event_count == 10
    assert report.supported_count == 7
    assert report.unsupported_count == 3
    assert math.isfinite(report.threshold)
    assert report.threshold == gate.threshold
    assert report.uncertainty_width == 0.05
    assert report.final_test_labels_used is False


def test_calibration_rejects_training_events() -> None:
    """Training records cannot substitute for held-out validation."""

    gate = SupportGate(
        method=SupportMethod.MAHALANOBIS,
    )
    gate.fit(events_for(MultisensorySplit.TRAIN))

    with pytest.raises(
        SupportGateError,
        match="validation data only",
    ):
        gate.calibrate(
            events_for(MultisensorySplit.TRAIN),
            uncertainty_width=0.05,
        )


def test_calibration_rejects_final_test_events() -> None:
    """The final test can never select the support threshold."""

    gate = SupportGate(
        method=SupportMethod.MAHALANOBIS,
    )
    gate.fit(events_for(MultisensorySplit.TRAIN))

    with pytest.raises(
        SupportGateError,
        match="validation data only",
    ):
        gate.calibrate(
            events_for(MultisensorySplit.FINAL_TEST),
            uncertainty_width=0.05,
        )


def test_calibration_requires_supported_and_unsupported_examples() -> None:
    """Validation must contain both sides of the support decision."""

    validation = tuple(
        event
        for event in events_for(MultisensorySplit.VALIDATION)
        if event.support_regime is not SupportRegime.UNSEEN_FAMILY
    )
    gate = SupportGate(
        method=SupportMethod.COSINE_MARGIN,
    )
    gate.fit(events_for(MultisensorySplit.TRAIN))

    with pytest.raises(
        SupportGateError,
        match="supported and unsupported",
    ):
        gate.calibrate(
            validation,
            uncertainty_width=0.05,
        )


@pytest.mark.parametrize(
    "value",
    (
        0.0,
        -0.1,
        1.1,
        float("nan"),
        True,
    ),
)
def test_invalid_uncertainty_width_is_rejected(
    value: object,
) -> None:
    """The uncertainty band must be finite and strictly between 0 and 1."""

    gate = SupportGate(
        method=SupportMethod.MAHALANOBIS,
    )
    gate.fit(events_for(MultisensorySplit.TRAIN))

    with pytest.raises(
        SupportGateError,
        match="uncertainty_width",
    ):
        gate.calibrate(
            events_for(MultisensorySplit.VALIDATION),
            uncertainty_width=value,  # type: ignore[arg-type]
        )


def test_decision_before_calibration_is_rejected() -> None:
    """A threshold-free gate cannot issue operational decisions."""

    gate = SupportGate(
        method=SupportMethod.MAHALANOBIS,
    )
    gate.fit(events_for(MultisensorySplit.TRAIN))

    with pytest.raises(
        SupportGateError,
        match="calibrated",
    ):
        gate.decide(
            event_id="query-001",
            query_vector=events_for(
                MultisensorySplit.TRAIN,
            )[0].olfactory_vector,
        )


def make_calibrated_gate() -> SupportGate:
    """Return one fitted and validation-calibrated gate."""

    gate = SupportGate(
        method=SupportMethod.MAHALANOBIS,
    )
    gate.fit(events_for(MultisensorySplit.TRAIN))
    gate.calibrate(
        events_for(MultisensorySplit.VALIDATION),
        uncertainty_width=0.05,
    )

    return gate


def test_decision_contains_score_support_and_uncertainty() -> None:
    """The gate emits every output required by the protocol."""

    gate = make_calibrated_gate()
    event = events_for(MultisensorySplit.FINAL_TEST)[0]

    decision = gate.decide(
        event_id=event.latent_event_id,
        query_vector=event.olfactory_vector,
    )

    assert isinstance(decision, SupportDecision)
    assert decision.event_id == event.latent_event_id
    assert decision.method is SupportMethod.MAHALANOBIS
    assert math.isfinite(decision.support_score)
    assert isinstance(decision.is_supported, bool)
    assert isinstance(
        decision.uncertainty_status,
        UncertaintyStatus,
    )
    assert decision.request_touch == (
        decision.uncertainty_status
        is UncertaintyStatus.UNCERTAIN
    )


def test_touch_is_requested_only_inside_uncertainty_band() -> None:
    """Conditional touch follows the validation-defined policy exactly."""

    gate = make_calibrated_gate()
    threshold = gate.threshold

    assert threshold is not None

    below = gate.decide_from_score(
        event_id="below",
        support_score=threshold - 0.06,
    )
    inside = gate.decide_from_score(
        event_id="inside",
        support_score=threshold,
    )
    above = gate.decide_from_score(
        event_id="above",
        support_score=threshold + 0.06,
    )

    assert below.uncertainty_status is UncertaintyStatus.CERTAIN_UNSUPPORTED
    assert below.is_supported is False
    assert below.request_touch is False

    assert inside.uncertainty_status is UncertaintyStatus.UNCERTAIN
    assert inside.request_touch is True

    assert above.uncertainty_status is UncertaintyStatus.CERTAIN_SUPPORTED
    assert above.is_supported is True
    assert above.request_touch is False


def test_threshold_is_deterministic() -> None:
    """Repeated validation calibration selects the same threshold."""

    first = make_calibrated_gate()
    second = make_calibrated_gate()

    assert first.threshold == second.threshold
    assert first.calibration_report == second.calibration_report


def test_calibration_report_is_immutable() -> None:
    """The validation lock cannot be silently edited."""

    gate = make_calibrated_gate()
    report = gate.calibration_report

    assert report is not None

    with pytest.raises(AttributeError):
        report.threshold = 0.0  # type: ignore[misc]


def test_mahalanobis_regularization_must_be_positive() -> None:
    """Covariance inversion uses explicit positive regularization."""

    with pytest.raises(
        SupportGateError,
        match="covariance_regularization",
    ):
        SupportGate(
            method=SupportMethod.MAHALANOBIS,
            covariance_regularization=0.0,
        )


def test_unknown_method_is_rejected() -> None:
    """The implementation cannot accept an unregistered method name."""

    with pytest.raises(
        SupportGateError,
        match="SupportMethod",
    ):
        SupportGate(method="invented")  # type: ignore[arg-type]
