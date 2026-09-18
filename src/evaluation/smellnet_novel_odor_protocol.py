"""Locked partition and temporal-window primitives for NOI v0.5.

The module defines data boundaries only. It does not train models, inspect
final-test performance, or select decision thresholds. Family rotation keeps
the final-unknown family absent from model training and validation within
each outer fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Iterable

from src.evaluation.smellnet_adapter import (
    SensorMatrix,
    SmellNetRecording,
    SmellNetSplit,
    audit_smellnet_recordings,
)


LOCKED_FAMILY_ORDER = (
    "fruits",
    "herbs",
    "nuts",
    "spices",
    "vegetables",
)


class NovelOdorProtocolError(ValueError):
    """Raised when a proposed v0.5 partition violates a locked boundary."""


@dataclass(frozen=True, slots=True)
class TemporalSensorWindow:
    """One nonoverlapping interval from a physical-sensor recording."""

    window_id: str
    recording_id: str
    split: SmellNetSplit
    ingredient: str
    family: str
    start: int
    stop: int
    sensor_rows: SensorMatrix
    source_sha256: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise NovelOdorProtocolError("Window bounds must be positive.")
        if len(self.sensor_rows) != self.stop - self.start:
            raise NovelOdorProtocolError(
                "Window length must match its half-open bounds."
            )
        if not self.window_id.startswith(f"{self.recording_id}#"):
            raise NovelOdorProtocolError(
                "window_id must be derived from its recording_id."
            )


@dataclass(frozen=True, slots=True)
class RepeatSensingPair:
    """Two consecutive, disjoint observations from the same recording."""

    pair_id: str
    initial: TemporalSensorWindow
    repeat: TemporalSensorWindow

    def __post_init__(self) -> None:
        if self.initial.recording_id != self.repeat.recording_id:
            raise NovelOdorProtocolError(
                "Repeat-sensing windows must share a recording."
            )
        if self.initial.stop > self.repeat.start:
            raise NovelOdorProtocolError(
                "Repeat-sensing windows must not overlap."
            )
        if self.initial.family != self.repeat.family:
            raise NovelOdorProtocolError(
                "Repeat-sensing windows must share a family label."
            )


@dataclass(frozen=True, slots=True)
class NovelOdorFold:
    """One outer family holdout with disjoint validation and final novelty."""

    fold_id: str
    known_families: tuple[str, ...]
    validation_unknown_family: str
    final_unknown_family: str
    model_train_ids: tuple[str, ...]
    validation_known_ids: tuple[str, ...]
    validation_unknown_ids: tuple[str, ...]
    final_known_ids: tuple[str, ...]
    final_unknown_ids: tuple[str, ...]
    unused_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        family_roles = {
            *self.known_families,
            self.validation_unknown_family,
            self.final_unknown_family,
        }
        if family_roles != set(LOCKED_FAMILY_ORDER):
            raise NovelOdorProtocolError(
                "Every fold must assign all five locked families exactly."
            )
        if len(self.known_families) != 3:
            raise NovelOdorProtocolError(
                "Every fold must contain exactly three known families."
            )
        if self.validation_unknown_family == self.final_unknown_family:
            raise NovelOdorProtocolError(
                "Validation and final unknown families must differ."
            )

        role_sets = tuple(
            set(values)
            for values in (
                self.model_train_ids,
                self.validation_known_ids,
                self.validation_unknown_ids,
                self.final_known_ids,
                self.final_unknown_ids,
                self.unused_ids,
            )
        )
        if any(not values for values in role_sets[:-1]):
            raise NovelOdorProtocolError(
                "Every evaluated partition role must be nonempty."
            )
        for index, left in enumerate(role_sets):
            for right in role_sets[index + 1 :]:
                if left & right:
                    raise NovelOdorProtocolError(
                        "Recording identities must not cross partition roles."
                    )

    @property
    def evaluated_ids(self) -> tuple[str, ...]:
        """Return all training, validation, and final IDs in stable order."""

        return tuple(
            chain(
                self.model_train_ids,
                self.validation_known_ids,
                self.validation_unknown_ids,
                self.final_known_ids,
                self.final_unknown_ids,
            )
        )


def create_nonoverlapping_windows(
    recording: SmellNetRecording,
    *,
    window_size: int,
) -> tuple[TemporalSensorWindow, ...]:
    """Split one recording into fixed, nonoverlapping full windows."""

    if not isinstance(recording, SmellNetRecording):
        raise NovelOdorProtocolError(
            "recording must be a validated SmellNetRecording."
        )
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise NovelOdorProtocolError("window_size must be an integer.")
    if window_size < 2:
        raise NovelOdorProtocolError("window_size must be at least two.")

    windows: list[TemporalSensorWindow] = []
    final_start = len(recording.sensor_rows) - window_size
    for start in range(0, final_start + 1, window_size):
        stop = start + window_size
        windows.append(
            TemporalSensorWindow(
                window_id=f"{recording.recording_id}#{start}:{stop}",
                recording_id=recording.recording_id,
                split=recording.split,
                ingredient=recording.ingredient,
                family=recording.family,
                start=start,
                stop=stop,
                sensor_rows=recording.sensor_rows[start:stop],
                source_sha256=recording.source_sha256,
            )
        )
    if not windows:
        raise NovelOdorProtocolError(
            "Recording is shorter than one complete temporal window."
        )
    return tuple(windows)


def create_disjoint_repeat_sensing_pairs(
    windows: Iterable[TemporalSensorWindow],
) -> tuple[RepeatSensingPair, ...]:
    """Pair windows 0-1, 2-3, and so on without reusing evidence."""

    values = tuple(windows)
    if any(not isinstance(value, TemporalSensorWindow) for value in values):
        raise NovelOdorProtocolError(
            "windows must contain only TemporalSensorWindow values."
        )

    grouped: dict[str, list[TemporalSensorWindow]] = {}
    for value in values:
        grouped.setdefault(value.recording_id, []).append(value)

    pairs: list[RepeatSensingPair] = []
    for recording_id in sorted(grouped):
        ordered = sorted(grouped[recording_id], key=lambda item: item.start)
        for index in range(0, len(ordered) - 1, 2):
            initial = ordered[index]
            repeat = ordered[index + 1]
            pairs.append(
                RepeatSensingPair(
                    pair_id=f"{recording_id}#pair-{index // 2}",
                    initial=initial,
                    repeat=repeat,
                )
            )
    return tuple(pairs)


def build_locked_novel_odor_folds(
    recordings: Iterable[SmellNetRecording],
) -> tuple[NovelOdorFold, ...]:
    """Build five deterministic nested family-holdout folds.

    For each outer fold, the final-unknown family is absent from all model
    development. The next family in the locked cycle supplies validation
    unknowns. Official testing recordings are used only for final roles.
    """

    values = tuple(recordings)
    if not values:
        raise NovelOdorProtocolError("At least one recording is required.")
    report = audit_smellnet_recordings(values)
    if not report.passed:
        raise NovelOdorProtocolError(
            "Recordings must pass the SmellNet integrity audit first."
        )
    observed_families = {value.family for value in values}
    if observed_families != set(LOCKED_FAMILY_ORDER):
        raise NovelOdorProtocolError(
            "Recordings must cover the five locked SmellNet families."
        )

    training = tuple(
        value for value in values if value.split is SmellNetSplit.TRAINING
    )
    testing = tuple(
        value for value in values if value.split is SmellNetSplit.TESTING
    )
    if not training or not testing:
        raise NovelOdorProtocolError(
            "Both official training and testing recordings are required."
        )

    folds: list[NovelOdorFold] = []
    for final_index, final_family in enumerate(LOCKED_FAMILY_ORDER):
        validation_family = LOCKED_FAMILY_ORDER[
            (final_index + 1) % len(LOCKED_FAMILY_ORDER)
        ]
        known_families = tuple(
            family
            for family in LOCKED_FAMILY_ORDER
            if family not in {final_family, validation_family}
        )

        model_train_ids: list[str] = []
        validation_known_ids: list[str] = []
        for ingredient in sorted(
            {
                value.ingredient
                for value in training
                if value.family in known_families
            }
        ):
            ingredient_records = sorted(
                (
                    value
                    for value in training
                    if value.ingredient == ingredient
                ),
                key=lambda value: value.recording_id,
            )
            if len(ingredient_records) < 2:
                raise NovelOdorProtocolError(
                    "Known ingredients need at least two training recordings."
                )
            model_train_ids.extend(
                value.recording_id for value in ingredient_records[:-1]
            )
            validation_known_ids.append(
                ingredient_records[-1].recording_id
            )

        validation_unknown_ids = tuple(
            sorted(
                value.recording_id
                for value in training
                if value.family == validation_family
            )
        )
        final_known_ids = tuple(
            sorted(
                value.recording_id
                for value in testing
                if value.family in known_families
            )
        )
        final_unknown_ids = tuple(
            sorted(
                value.recording_id
                for value in testing
                if value.family == final_family
            )
        )

        assigned = {
            *model_train_ids,
            *validation_known_ids,
            *validation_unknown_ids,
            *final_known_ids,
            *final_unknown_ids,
        }
        unused_ids = tuple(
            sorted(
                value.recording_id
                for value in values
                if value.recording_id not in assigned
            )
        )

        folds.append(
            NovelOdorFold(
                fold_id=f"outer-{final_index + 1}-{final_family}",
                known_families=known_families,
                validation_unknown_family=validation_family,
                final_unknown_family=final_family,
                model_train_ids=tuple(sorted(model_train_ids)),
                validation_known_ids=tuple(sorted(validation_known_ids)),
                validation_unknown_ids=validation_unknown_ids,
                final_known_ids=final_known_ids,
                final_unknown_ids=final_unknown_ids,
                unused_ids=unused_ids,
            )
        )
    return tuple(folds)
