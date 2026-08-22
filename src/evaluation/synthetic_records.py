"""Validated records produced by the independent synthetic generator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


Vector = tuple[float, ...]


class SyntheticRecordError(ValueError):
    """Raised when a generated synthetic record is invalid."""


class SplitLabel(str, Enum):
    """Prespecified dataset split labels."""

    TRAIN = "train"
    VALIDATION = "validation"
    OOD_TEST = "ood_test"


def _validate_vector(name: str, vector: Vector) -> None:
    if not vector:
        raise SyntheticRecordError(
            f"{name} must not be empty."
        )

    if not all(isinstance(value, (int, float)) for value in vector):
        raise SyntheticRecordError(
            f"{name} must contain only numeric values."
        )

    if not all(isfinite(float(value)) for value in vector):
        raise SyntheticRecordError(
            f"{name} must contain only finite values."
        )


@dataclass(frozen=True, slots=True)
class SyntheticOdorTarget:
    """An opaque synthetic target retained outside model features."""

    item_id: str
    family_id: int
    odor_vector: Vector

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise SyntheticRecordError(
                "item_id must not be empty."
            )

        if (
            isinstance(self.family_id, bool)
            or not isinstance(self.family_id, int)
            or self.family_id < 0
        ):
            raise SyntheticRecordError(
                "family_id must be a nonnegative integer."
            )

        _validate_vector("odor_vector", self.odor_vector)


@dataclass(frozen=True, slots=True)
class SyntheticEvent:
    """One synthetic multimodal event and its external ground truth."""

    event_id: str
    split: SplitLabel
    template_id: int
    target_item_id: str
    target_family_id: int
    text_vector: Vector
    image_vector: Vector
    audio_vector: Vector

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise SyntheticRecordError(
                "event_id must not be empty."
            )

        if not isinstance(self.split, SplitLabel):
            raise SyntheticRecordError(
                "split must be a valid SplitLabel."
            )

        integer_fields = {
            "template_id": self.template_id,
            "target_family_id": self.target_family_id,
        }

        for name, value in integer_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise SyntheticRecordError(
                    f"{name} must be a nonnegative integer."
                )

        if not self.target_item_id.strip():
            raise SyntheticRecordError(
                "target_item_id must not be empty."
            )

        _validate_vector("text_vector", self.text_vector)
        _validate_vector("image_vector", self.image_vector)
        _validate_vector("audio_vector", self.audio_vector)

        dimensions = {
            len(self.text_vector),
            len(self.image_vector),
            len(self.audio_vector),
        }

        if len(dimensions) != 1:
            raise SyntheticRecordError(
                "All modality vectors must have identical dimensions."
            )


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """An immutable in-memory synthetic evaluation dataset."""

    odor_targets: tuple[SyntheticOdorTarget, ...]
    events: tuple[SyntheticEvent, ...]
    generator_version: str
    generator_seed: int
    ood_seed: int

    def __post_init__(self) -> None:
        if not self.odor_targets:
            raise SyntheticRecordError(
                "odor_targets must not be empty."
            )

        if not self.events:
            raise SyntheticRecordError(
                "events must not be empty."
            )

        if not self.generator_version.strip():
            raise SyntheticRecordError(
                "generator_version must not be empty."
            )

        target_ids = [
            target.item_id
            for target in self.odor_targets
        ]
        event_ids = [
            event.event_id
            for event in self.events
        ]

        if len(target_ids) != len(set(target_ids)):
            raise SyntheticRecordError(
                "Synthetic target identifiers must be unique."
            )

        if len(event_ids) != len(set(event_ids)):
            raise SyntheticRecordError(
                "Synthetic event identifiers must be unique."
            )

        known_targets = set(target_ids)

        if any(
            event.target_item_id not in known_targets
            for event in self.events
        ):
            raise SyntheticRecordError(
                "Every event target must exist in the odor library."
            )

        target_family_map = {
            target.item_id: target.family_id
            for target in self.odor_targets
        }

        if any(
            target_family_map[event.target_item_id]
            != event.target_family_id
            for event in self.events
        ):
            raise SyntheticRecordError(
                "Event target-family ground truth is inconsistent."
            )

        if self.generator_seed == self.ood_seed:
            raise SyntheticRecordError(
                "Generator and OOD seeds must be different."
            )