"""Tests for evidence-derived reliability and conflict scoring."""

from __future__ import annotations

import inspect
import math

import pytest

from src.evaluation.evidence_conflict import (
    EvidenceConflictDetector,
    EvidenceConflictError,
)
from src.evaluation.multisensory_records import (
    MultisensorySplit,
)
from src.evaluation.noi_v0_3_generator import (
    NOIV03GenerationConfig,
    generate_noi_v0_3_events,
)


def generated():
    """Return deterministic feasibility records."""

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
            generator_version="0.3.1-evidence-test",
            feasibility_only=True,
        )
    )


def events_for(split: MultisensorySplit):
    """Return one split."""

    return tuple(
        event
        for event in generated().latent_events
        if event.split is split
    )


def fitted_detector() -> EvidenceConflictDetector:
    """Return one training-only fitted detector."""

    detector = EvidenceConflictDetector()
    detector.fit(events_for(MultisensorySplit.TRAIN))

    return detector


def different_family_pair():
    """Return olfactory and tactile evidence from different families."""

    training = events_for(MultisensorySplit.TRAIN)
    odor_event = training[0]
    touch_event = next(
        event
        for event in training
        if event.target_family_id != odor_event.target_family_id
    )

    return odor_event, touch_event


def test_public_assessment_has_evidence_inputs_only() -> None:
    """Inference cannot receive condition or ground-truth metadata."""

    signature = inspect.signature(
        EvidenceConflictDetector.assess
    )

    assert set(signature.parameters) == {
        "self",
        "olfactory_vector",
        "tactile_vector",
    }


def test_fit_accepts_training_only() -> None:
    """Family prototypes are derived only from training records."""

    detector = EvidenceConflictDetector()
    returned = detector.fit(
        events_for(MultisensorySplit.TRAIN)
    )

    assert returned is detector
    assert detector.is_fitted is True
    assert detector.training_event_count == 70
    assert detector.training_family_count == 4


@pytest.mark.parametrize(
    "split",
    (
        MultisensorySplit.VALIDATION,
        MultisensorySplit.FINAL_TEST,
    ),
)
def test_fit_rejects_evaluation_splits(
    split: MultisensorySplit,
) -> None:
    """Validation and final-test records cannot shape prototypes."""

    with pytest.raises(
        EvidenceConflictError,
        match="training data only",
    ):
        EvidenceConflictDetector().fit(events_for(split))


def test_assessment_before_fit_is_rejected() -> None:
    """Evidence cannot be assessed before training-only fitting."""

    with pytest.raises(
        EvidenceConflictError,
        match="fitted",
    ):
        EvidenceConflictDetector().assess(
            olfactory_vector=(1.0,) * 16,
            tactile_vector=(1.0,) * 8,
        )


def test_assessment_outputs_finite_unit_interval_values() -> None:
    """Reliabilities and conflict score must remain finite in [0, 1]."""

    event = events_for(MultisensorySplit.TRAIN)[0]
    assessment = fitted_detector().assess(
        olfactory_vector=event.olfactory_vector,
        tactile_vector=event.tactile_vector,
    )

    values = (
        assessment.odor_reliability,
        assessment.touch_reliability,
        assessment.conflict_score,
    )

    assert all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in values
    )


def test_family_distributions_are_normalized() -> None:
    """Each available modality produces an auditable distribution."""

    event = events_for(MultisensorySplit.TRAIN)[0]
    assessment = fitted_detector().assess(
        olfactory_vector=event.olfactory_vector,
        tactile_vector=event.tactile_vector,
    )

    assert sum(assessment.odor_family_distribution) == pytest.approx(
        1.0
    )
    assert sum(assessment.touch_family_distribution) == pytest.approx(
        1.0
    )
    assert len(assessment.odor_family_distribution) == 4
    assert len(assessment.touch_family_distribution) == 4


def test_cross_family_evidence_increases_conflict() -> None:
    """Mismatched-family evidence must score above matched evidence."""

    odor_event, touch_event = different_family_pair()
    detector = fitted_detector()

    matched = detector.assess(
        olfactory_vector=odor_event.olfactory_vector,
        tactile_vector=odor_event.tactile_vector,
    )
    mismatched = detector.assess(
        olfactory_vector=odor_event.olfactory_vector,
        tactile_vector=touch_event.tactile_vector,
    )

    assert mismatched.conflict_score > matched.conflict_score


def test_missing_touch_receives_zero_reliability() -> None:
    """Unavailable touch cannot contribute reliability or conflict."""

    event = events_for(MultisensorySplit.TRAIN)[0]
    assessment = fitted_detector().assess(
        olfactory_vector=event.olfactory_vector,
        tactile_vector=None,
    )

    assert assessment.odor_available is True
    assert assessment.touch_available is False
    assert assessment.touch_reliability == 0.0
    assert assessment.conflict_available is False
    assert assessment.conflict_score == 0.0


def test_missing_odor_receives_zero_reliability() -> None:
    """Unavailable odor cannot contribute reliability or conflict."""

    event = events_for(MultisensorySplit.TRAIN)[0]
    assessment = fitted_detector().assess(
        olfactory_vector=None,
        tactile_vector=event.tactile_vector,
    )

    assert assessment.odor_available is False
    assert assessment.touch_available is True
    assert assessment.odor_reliability == 0.0
    assert assessment.conflict_available is False
    assert assessment.conflict_score == 0.0


def test_both_modalities_missing_is_rejected() -> None:
    """At least one evidence stream is required."""

    with pytest.raises(
        EvidenceConflictError,
        match="At least one modality",
    ):
        fitted_detector().assess(
            olfactory_vector=None,
            tactile_vector=None,
        )


def test_assessment_is_deterministic() -> None:
    """Repeated assessment of identical vectors is exact."""

    event = events_for(MultisensorySplit.TRAIN)[0]
    detector = fitted_detector()

    first = detector.assess(
        olfactory_vector=event.olfactory_vector,
        tactile_vector=event.tactile_vector,
    )
    second = detector.assess(
        olfactory_vector=event.olfactory_vector,
        tactile_vector=event.tactile_vector,
    )

    assert first == second


def test_assessment_contains_no_target_ground_truth() -> None:
    """The inference result cannot expose or depend on target answers."""

    event = events_for(MultisensorySplit.TRAIN)[0]
    assessment = fitted_detector().assess(
        olfactory_vector=event.olfactory_vector,
        tactile_vector=event.tactile_vector,
    )

    fields = set(assessment.__dataclass_fields__)

    assert "target_item_id" not in fields
    assert "target_family_id" not in fields
    assert "condition" not in fields
    assert "modality_conflict" not in fields
