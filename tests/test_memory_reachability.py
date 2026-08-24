"""Tests for memory-target reachability auditing."""

import pytest

from src.evaluation.memory_reachability import (
    MemoryReachabilityError,
    audit_memory_reachability,
)
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
    SyntheticEvent,
    SyntheticOdorTarget,
)


def make_target(
    item_id: str,
    family_id: int,
) -> SyntheticOdorTarget:
    return SyntheticOdorTarget(
        item_id=item_id,
        family_id=family_id,
        odor_vector=(1.0, 0.0),
    )


def make_event(
    event_id: str,
    split: SplitLabel,
    target_item_id: str,
    target_family_id: int,
) -> SyntheticEvent:
    return SyntheticEvent(
        event_id=event_id,
        split=split,
        template_id={
            SplitLabel.TRAIN: 1,
            SplitLabel.VALIDATION: 2,
            SplitLabel.OOD_TEST: 3,
        }[split],
        target_item_id=target_item_id,
        target_family_id=target_family_id,
        text_vector=(1.0, 0.0),
        image_vector=(0.9, 0.1),
        audio_vector=(0.8, 0.2),
    )


def make_dataset() -> SyntheticDataset:
    return SyntheticDataset(
        odor_targets=(
            make_target("odor-a", 0),
            make_target("odor-b", 0),
            make_target("odor-c", 1),
        ),
        events=(
            make_event(
                "train-a",
                SplitLabel.TRAIN,
                "odor-a",
                0,
            ),
            make_event(
                "validation-a",
                SplitLabel.VALIDATION,
                "odor-a",
                0,
            ),
            make_event(
                "validation-b",
                SplitLabel.VALIDATION,
                "odor-b",
                0,
            ),
            make_event(
                "ood-c",
                SplitLabel.OOD_TEST,
                "odor-c",
                1,
            ),
        ),
        generator_version="0.2.0",
        generator_seed=1001,
        ood_seed=9001,
    )


def test_reachability_is_derived_from_training_targets() -> None:
    audit = audit_memory_reachability(make_dataset())

    assert audit.training_target_ids == ("odor-a",)

    validation = audit.for_split(SplitLabel.VALIDATION)

    assert validation.reachable_target_ids == ("odor-a",)
    assert validation.unreachable_target_ids == ("odor-b",)
    assert validation.reachable_event_ids == ("validation-a",)
    assert validation.unreachable_event_ids == ("validation-b",)


def test_reachable_fractions_are_reported() -> None:
    audit = audit_memory_reachability(make_dataset())
    validation = audit.for_split(SplitLabel.VALIDATION)

    assert validation.target_count == 2
    assert validation.event_count == 2
    assert validation.reachable_target_fraction == 0.5
    assert validation.reachable_event_fraction == 0.5


def test_fully_unseen_ood_split_has_zero_reachability() -> None:
    audit = audit_memory_reachability(make_dataset())
    ood = audit.for_split(SplitLabel.OOD_TEST)

    assert ood.reachable_target_ids == ()
    assert ood.unreachable_target_ids == ("odor-c",)
    assert ood.reachable_target_fraction == 0.0
    assert ood.reachable_event_fraction == 0.0


def test_requesting_training_as_evaluation_is_rejected() -> None:
    audit = audit_memory_reachability(make_dataset())

    with pytest.raises(
        MemoryReachabilityError,
        match="evaluation split",
    ):
        audit.for_split(SplitLabel.TRAIN)
