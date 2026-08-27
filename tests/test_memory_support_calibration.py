"""Tests for leakage-free memory-support calibration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.evaluation.memory_support_calibration import (
    MemorySupportCalibrationError,
    calibrate_memory_support,
)
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticEvent,
)


def make_event(
    event_id: str,
    *,
    split: SplitLabel,
    target_item_id: str,
    vector: tuple[float, ...],
) -> SyntheticEvent:
    """Create an event whose three modalities agree."""

    return SyntheticEvent(
        event_id=event_id,
        split=split,
        template_id=1,
        target_item_id=target_item_id,
        target_family_id=1,
        text_vector=vector,
        image_vector=vector,
        audio_vector=vector,
    )


def training_event(
    event_id: str = "train-1",
    vector: tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
    ),
) -> SyntheticEvent:
    return make_event(
        event_id,
        split=SplitLabel.TRAIN,
        target_item_id="odor-known",
        vector=vector,
    )


def validation_event(
    event_id: str,
    *,
    target_item_id: str = "odor-known",
    vector: tuple[float, ...],
) -> SyntheticEvent:
    return make_event(
        event_id,
        split=SplitLabel.VALIDATION,
        target_item_id=target_item_id,
        vector=vector,
    )


def ood_event(
    event_id: str,
    *,
    target_item_id: str,
    vector: tuple[float, ...],
) -> SyntheticEvent:
    return make_event(
        event_id,
        split=SplitLabel.OOD_TEST,
        target_item_id=target_item_id,
        vector=vector,
    )


def test_threshold_is_largest_value_preserving_coverage() -> None:
    calibration = calibrate_memory_support(
        training_events=(training_event(),),
        validation_events=(
            validation_event(
                "validation-1",
                vector=(1.0, 0.0, 0.0),
            ),
            validation_event(
                "validation-2",
                vector=(0.8, 0.6, 0.0),
            ),
            validation_event(
                "validation-3",
                vector=(0.0, 1.0, 0.0),
            ),
        ),
        minimum_reachable_coverage=2.0 / 3.0,
    )

    assert calibration.threshold == pytest.approx(0.8)
    assert calibration.calibration_event_count == 3
    assert (
        calibration.achieved_reachable_coverage
        == pytest.approx(2.0 / 3.0)
    )
    assert calibration.minimum_reachable_coverage == (
        pytest.approx(2.0 / 3.0)
    )


def test_unreachable_validation_events_are_excluded() -> None:
    calibration = calibrate_memory_support(
        training_events=(training_event(),),
        validation_events=(
            validation_event(
                "reachable",
                vector=(1.0, 0.0, 0.0),
            ),
            validation_event(
                "unreachable",
                target_item_id="odor-unseen",
                vector=(0.0, 1.0, 0.0),
            ),
        ),
        minimum_reachable_coverage=0.95,
    )

    assert calibration.calibration_event_ids == (
        "reachable",
    )
    assert calibration.excluded_event_ids == (
        "unreachable",
    )
    assert calibration.calibration_event_count == 1
    assert calibration.excluded_event_count == 1


def test_score_is_clipped_cosine_similarity() -> None:
    calibration = calibrate_memory_support(
        training_events=(training_event(),),
        validation_events=(
            validation_event(
                "validation-1",
                vector=(1.0, 0.0, 0.0),
            ),
        ),
        minimum_reachable_coverage=0.95,
    )

    matching = calibration.score_event(
        ood_event(
            "matching",
            target_item_id="odor-new-a",
            vector=(1.0, 0.0, 0.0),
        )
    )
    orthogonal = calibration.score_event(
        ood_event(
            "orthogonal",
            target_item_id="odor-new-b",
            vector=(0.0, 1.0, 0.0),
        )
    )
    opposite = calibration.score_event(
        ood_event(
            "opposite",
            target_item_id="odor-new-c",
            vector=(-1.0, 0.0, 0.0),
        )
    )

    assert matching.support_score == pytest.approx(1.0)
    assert orthogonal.support_score == pytest.approx(0.0)
    assert opposite.support_score == pytest.approx(0.0)


def test_threshold_ties_are_supported() -> None:
    calibration = calibrate_memory_support(
        training_events=(training_event(),),
        validation_events=(
            validation_event(
                "validation-1",
                vector=(0.8, 0.6, 0.0),
            ),
        ),
        minimum_reachable_coverage=0.95,
    )

    decision = calibration.score_event(
        ood_event(
            "tie",
            target_item_id="odor-new",
            vector=(0.8, 0.6, 0.0),
        )
    )

    assert decision.support_score == pytest.approx(
        calibration.threshold
    )
    assert decision.supported is True
    assert decision.abstained is False


def test_target_identity_does_not_change_score() -> None:
    calibration = calibrate_memory_support(
        training_events=(training_event(),),
        validation_events=(
            validation_event(
                "validation-1",
                vector=(1.0, 0.0, 0.0),
            ),
        ),
        minimum_reachable_coverage=0.95,
    )

    first = calibration.score_event(
        ood_event(
            "same-features-a",
            target_item_id="odor-new-a",
            vector=(0.8, 0.6, 0.0),
        )
    )
    second = calibration.score_event(
        ood_event(
            "same-features-b",
            target_item_id="odor-new-b",
            vector=(0.8, 0.6, 0.0),
        )
    )

    assert first.support_score == second.support_score
    assert first.supported == second.supported
    assert calibration.target_identifier_used is False
    assert calibration.family_identifier_used is False
    assert calibration.ood_oracle_used is False


def test_batch_scoring_preserves_input_order() -> None:
    calibration = calibrate_memory_support(
        training_events=(training_event(),),
        validation_events=(
            validation_event(
                "validation-1",
                vector=(1.0, 0.0, 0.0),
            ),
        ),
        minimum_reachable_coverage=0.95,
    )

    decisions = calibration.score_events(
        (
            ood_event(
                "ood-2",
                target_item_id="odor-new-2",
                vector=(0.0, 1.0, 0.0),
            ),
            ood_event(
                "ood-1",
                target_item_id="odor-new-1",
                vector=(1.0, 0.0, 0.0),
            ),
        )
    )

    assert tuple(
        decision.event_id
        for decision in decisions
    ) == (
        "ood-2",
        "ood-1",
    )


def test_wrong_training_split_is_rejected() -> None:
    invalid_training = validation_event(
        "not-training",
        vector=(1.0, 0.0, 0.0),
    )

    with pytest.raises(
        MemorySupportCalibrationError,
        match="training_events",
    ):
        calibrate_memory_support(
            training_events=(invalid_training,),
            validation_events=(
                validation_event(
                    "validation-1",
                    vector=(1.0, 0.0, 0.0),
                ),
            ),
            minimum_reachable_coverage=0.95,
        )


def test_wrong_validation_split_is_rejected() -> None:
    invalid_validation = ood_event(
        "not-validation",
        target_item_id="odor-known",
        vector=(1.0, 0.0, 0.0),
    )

    with pytest.raises(
        MemorySupportCalibrationError,
        match="validation_events",
    ):
        calibrate_memory_support(
            training_events=(training_event(),),
            validation_events=(
                invalid_validation,
            ),
            minimum_reachable_coverage=0.95,
        )


def test_empty_reachable_calibration_is_rejected() -> None:
    with pytest.raises(
        MemorySupportCalibrationError,
        match="reachable validation",
    ):
        calibrate_memory_support(
            training_events=(training_event(),),
            validation_events=(
                validation_event(
                    "unreachable",
                    target_item_id="odor-unseen",
                    vector=(0.0, 1.0, 0.0),
                ),
            ),
            minimum_reachable_coverage=0.95,
        )


def test_calibration_and_decisions_are_immutable() -> None:
    calibration = calibrate_memory_support(
        training_events=(training_event(),),
        validation_events=(
            validation_event(
                "validation-1",
                vector=(1.0, 0.0, 0.0),
            ),
        ),
        minimum_reachable_coverage=0.95,
    )
    decision = calibration.score_event(
        ood_event(
            "ood-1",
            target_item_id="odor-new",
            vector=(0.0, 1.0, 0.0),
        )
    )

    with pytest.raises(FrozenInstanceError):
        calibration.threshold = 0.0  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        decision.supported = True  # type: ignore[misc]
