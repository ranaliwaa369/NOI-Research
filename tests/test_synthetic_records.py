"""Tests for independent synthetic dataset records."""

from dataclasses import replace

import pytest

from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
    SyntheticEvent,
    SyntheticOdorTarget,
    SyntheticRecordError,
)


def make_target(
    *,
    item_id: str = "odor-001",
    family_id: int = 0,
) -> SyntheticOdorTarget:
    return SyntheticOdorTarget(
        item_id=item_id,
        family_id=family_id,
        odor_vector=(1.0, 0.0),
    )


def make_event(
    *,
    event_id: str = "event-001",
    target_item_id: str = "odor-001",
    target_family_id: int = 0,
) -> SyntheticEvent:
    return SyntheticEvent(
        event_id=event_id,
        split=SplitLabel.TRAIN,
        template_id=1,
        target_item_id=target_item_id,
        target_family_id=target_family_id,
        text_vector=(1.0, 0.0),
        image_vector=(0.9, 0.1),
        audio_vector=(0.8, 0.2),
    )


def test_valid_synthetic_dataset() -> None:
    dataset = SyntheticDataset(
        odor_targets=(make_target(),),
        events=(make_event(),),
        generator_version="0.1.0",
        generator_seed=1001,
        ood_seed=9001,
    )

    assert len(dataset.odor_targets) == 1
    assert len(dataset.events) == 1


def test_nonfinite_odor_vector_is_rejected() -> None:
    with pytest.raises(
        SyntheticRecordError,
        match="finite values",
    ):
        SyntheticOdorTarget(
            item_id="odor-001",
            family_id=0,
            odor_vector=(1.0, float("nan")),
        )


def test_mismatched_modality_dimensions_are_rejected() -> None:
    with pytest.raises(
        SyntheticRecordError,
        match="identical dimensions",
    ):
        SyntheticEvent(
            event_id="event-001",
            split=SplitLabel.TRAIN,
            template_id=1,
            target_item_id="odor-001",
            target_family_id=0,
            text_vector=(1.0, 0.0),
            image_vector=(1.0, 0.0, 0.0),
            audio_vector=(1.0, 0.0),
        )


def test_duplicate_target_ids_are_rejected() -> None:
    target = make_target()

    with pytest.raises(
        SyntheticRecordError,
        match="target identifiers must be unique",
    ):
        SyntheticDataset(
            odor_targets=(target, target),
            events=(make_event(),),
            generator_version="0.1.0",
            generator_seed=1001,
            ood_seed=9001,
        )


def test_duplicate_event_ids_are_rejected() -> None:
    event = make_event()

    with pytest.raises(
        SyntheticRecordError,
        match="event identifiers must be unique",
    ):
        SyntheticDataset(
            odor_targets=(make_target(),),
            events=(event, event),
            generator_version="0.1.0",
            generator_seed=1001,
            ood_seed=9001,
        )


def test_unknown_event_target_is_rejected() -> None:
    with pytest.raises(
        SyntheticRecordError,
        match="target must exist",
    ):
        SyntheticDataset(
            odor_targets=(make_target(),),
            events=(
                make_event(target_item_id="unknown-odor"),
            ),
            generator_version="0.1.0",
            generator_seed=1001,
            ood_seed=9001,
        )


def test_inconsistent_target_family_is_rejected() -> None:
    with pytest.raises(
        SyntheticRecordError,
        match="ground truth is inconsistent",
    ):
        SyntheticDataset(
            odor_targets=(make_target(family_id=0),),
            events=(make_event(target_family_id=1),),
            generator_version="0.1.0",
            generator_seed=1001,
            ood_seed=9001,
        )


def test_id_and_ood_seed_must_differ() -> None:
    with pytest.raises(
        SyntheticRecordError,
        match="seeds must be different",
    ):
        SyntheticDataset(
            odor_targets=(make_target(),),
            events=(make_event(),),
            generator_version="0.1.0",
            generator_seed=1001,
            ood_seed=1001,
        )