"""Leakage-resistant calibration of memory-support scores."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticEvent,
)


FloatArray = NDArray[np.float64]


class MemorySupportCalibrationError(ValueError):
    """Raised when memory support cannot be calibrated safely."""


@dataclass(frozen=True)
class MemorySupportDecision:
    """One immutable support or abstention decision."""

    event_id: str
    support_score: float
    threshold: float
    supported: bool
    abstained: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.event_id, str)
            or not self.event_id.strip()
        ):
            raise MemorySupportCalibrationError(
                "event_id must be a nonempty string."
            )

        for name, value in (
            ("support_score", self.support_score),
            ("threshold", self.threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise MemorySupportCalibrationError(
                    f"{name} must be finite and in [0, 1]."
                )

        if not isinstance(self.supported, bool):
            raise MemorySupportCalibrationError(
                "supported must be boolean."
            )

        if not isinstance(self.abstained, bool):
            raise MemorySupportCalibrationError(
                "abstained must be boolean."
            )

        if self.supported == self.abstained:
            raise MemorySupportCalibrationError(
                "supported and abstained must be complements."
            )


@dataclass(frozen=True)
class MemorySupportCalibration:
    """Immutable threshold and training-memory representation."""

    threshold: float
    minimum_reachable_coverage: float
    achieved_reachable_coverage: float
    training_event_ids: tuple[str, ...]
    calibration_event_ids: tuple[str, ...]
    excluded_event_ids: tuple[str, ...]
    training_vectors: tuple[
        tuple[float, ...],
        ...,
    ]
    target_identifier_used: bool = False
    family_identifier_used: bool = False
    ood_oracle_used: bool = False
    final_test_tuning_used: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("threshold", self.threshold),
            (
                "minimum_reachable_coverage",
                self.minimum_reachable_coverage,
            ),
            (
                "achieved_reachable_coverage",
                self.achieved_reachable_coverage,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise MemorySupportCalibrationError(
                    f"{name} must be finite and in [0, 1]."
                )

        if not self.training_event_ids:
            raise MemorySupportCalibrationError(
                "At least one training memory is required."
            )

        if not self.calibration_event_ids:
            raise MemorySupportCalibrationError(
                "At least one reachable validation event "
                "is required."
            )

        if (
            len(self.training_event_ids)
            != len(self.training_vectors)
        ):
            raise MemorySupportCalibrationError(
                "Training identifiers and vectors must align."
            )

        if (
            len(set(self.training_event_ids))
            != len(self.training_event_ids)
        ):
            raise MemorySupportCalibrationError(
                "Training event identifiers must be unique."
            )

        if (
            len(set(self.calibration_event_ids))
            != len(self.calibration_event_ids)
        ):
            raise MemorySupportCalibrationError(
                "Calibration event identifiers must be unique."
            )

        if (
            set(self.calibration_event_ids)
            & set(self.excluded_event_ids)
        ):
            raise MemorySupportCalibrationError(
                "Included and excluded calibration events "
                "must not overlap."
            )

        if (
            self.achieved_reachable_coverage
            < self.minimum_reachable_coverage
        ):
            raise MemorySupportCalibrationError(
                "Achieved reachable coverage is below "
                "the required minimum."
            )

        for flag_name, flag_value in (
            (
                "target_identifier_used",
                self.target_identifier_used,
            ),
            (
                "family_identifier_used",
                self.family_identifier_used,
            ),
            ("ood_oracle_used", self.ood_oracle_used),
            (
                "final_test_tuning_used",
                self.final_test_tuning_used,
            ),
        ):
            if flag_value is not False:
                raise MemorySupportCalibrationError(
                    f"{flag_name} must remain false."
                )

        dimensions = {
            len(vector)
            for vector in self.training_vectors
        }

        if len(dimensions) != 1 or not dimensions:
            raise MemorySupportCalibrationError(
                "Training memory vectors must share one dimension."
            )

        for vector in self.training_vectors:
            array = np.asarray(
                vector,
                dtype=np.float64,
            )

            if (
                array.ndim != 1
                or not np.all(np.isfinite(array))
                or not np.isclose(
                    np.linalg.norm(array),
                    1.0,
                    atol=1e-10,
                )
            ):
                raise MemorySupportCalibrationError(
                    "Training memory vectors must be finite "
                    "unit vectors."
                )

    @property
    def calibration_event_count(self) -> int:
        """Return reachable events used for calibration."""

        return len(self.calibration_event_ids)

    @property
    def excluded_event_count(self) -> int:
        """Return unreachable validation events excluded."""

        return len(self.excluded_event_ids)

    @property
    def training_event_count(self) -> int:
        """Return the number of stored training memories."""

        return len(self.training_event_ids)

    def score_event(
        self,
        event: Any,
    ) -> MemorySupportDecision:
        """Score one event without reading its target identity."""

        query = _mean_fuse_event(event)
        memory_matrix = np.asarray(
            self.training_vectors,
            dtype=np.float64,
        )

        if query.shape[0] != memory_matrix.shape[1]:
            raise MemorySupportCalibrationError(
                "Evaluation vector dimension does not match "
                "the training memory dimension."
            )

        similarities = memory_matrix @ query
        maximum = float(np.max(similarities))
        support_score = float(
            np.clip(maximum, 0.0, 1.0)
        )

        supported = bool(
            support_score >= self.threshold
        )

        return MemorySupportDecision(
            event_id=_event_id(event),
            support_score=support_score,
            threshold=self.threshold,
            supported=supported,
            abstained=not supported,
        )

    def score_events(
        self,
        events: Iterable[Any],
    ) -> tuple[MemorySupportDecision, ...]:
        """Score events while preserving their input order."""

        if isinstance(events, (str, bytes)):
            raise MemorySupportCalibrationError(
                "events must be an iterable of event records."
            )

        try:
            event_tuple = tuple(events)
        except TypeError as error:
            raise MemorySupportCalibrationError(
                "events must be iterable."
            ) from error

        return tuple(
            self.score_event(event)
            for event in event_tuple
        )


def calibrate_memory_support(
    *,
    training_events: Sequence[SyntheticEvent],
    validation_events: Sequence[SyntheticEvent],
    minimum_reachable_coverage: float,
) -> MemorySupportCalibration:
    """Select the strictest threshold retaining required coverage."""

    minimum_coverage = _probability(
        minimum_reachable_coverage,
        "minimum_reachable_coverage",
    )

    if minimum_coverage <= 0.0:
        raise MemorySupportCalibrationError(
            "minimum_reachable_coverage must be positive."
        )

    training = _event_sequence(
        training_events,
        name="training_events",
        required_split=SplitLabel.TRAIN,
    )
    validation = _event_sequence(
        validation_events,
        name="validation_events",
        required_split=SplitLabel.VALIDATION,
    )

    training_target_ids = {
        event.target_item_id
        for event in training
    }

    reachable_validation = tuple(
        event
        for event in validation
        if event.target_item_id
        in training_target_ids
    )
    excluded_validation = tuple(
        event
        for event in validation
        if event.target_item_id
        not in training_target_ids
    )

    if not reachable_validation:
        raise MemorySupportCalibrationError(
            "At least one reachable validation event "
            "is required for calibration."
        )

    training_vectors = tuple(
        tuple(
            float(value)
            for value in _mean_fuse_event(event)
        )
        for event in training
    )

    memory_matrix = np.asarray(
        training_vectors,
        dtype=np.float64,
    )

    calibration_scores: list[float] = []

    for event in reachable_validation:
        query = _mean_fuse_event(event)

        if query.shape[0] != memory_matrix.shape[1]:
            raise MemorySupportCalibrationError(
                "Validation vector dimension does not match "
                "the training memory dimension."
            )

        score = float(
            np.clip(
                np.max(memory_matrix @ query),
                0.0,
                1.0,
            )
        )
        calibration_scores.append(score)

    required_supported_count = ceil(
        minimum_coverage
        * len(calibration_scores)
    )

    descending_scores = sorted(
        calibration_scores,
        reverse=True,
    )
    threshold = float(
        descending_scores[
            required_supported_count - 1
        ]
    )

    achieved_coverage = (
        sum(
            score >= threshold
            for score in calibration_scores
        )
        / len(calibration_scores)
    )

    return MemorySupportCalibration(
        threshold=threshold,
        minimum_reachable_coverage=(
            minimum_coverage
        ),
        achieved_reachable_coverage=float(
            achieved_coverage
        ),
        training_event_ids=tuple(
            event.event_id
            for event in training
        ),
        calibration_event_ids=tuple(
            event.event_id
            for event in reachable_validation
        ),
        excluded_event_ids=tuple(
            event.event_id
            for event in excluded_validation
        ),
        training_vectors=training_vectors,
        target_identifier_used=False,
        family_identifier_used=False,
        ood_oracle_used=False,
        final_test_tuning_used=False,
    )


def _event_sequence(
    events: Sequence[SyntheticEvent],
    *,
    name: str,
    required_split: SplitLabel,
) -> tuple[SyntheticEvent, ...]:
    if isinstance(events, (str, bytes)):
        raise MemorySupportCalibrationError(
            f"{name} must contain SyntheticEvent records."
        )

    try:
        event_tuple = tuple(events)
    except TypeError as error:
        raise MemorySupportCalibrationError(
            f"{name} must be a sequence."
        ) from error

    if not event_tuple:
        raise MemorySupportCalibrationError(
            f"{name} must not be empty."
        )

    for event in event_tuple:
        if (
            not isinstance(event, SyntheticEvent)
            or event.split is not required_split
        ):
            raise MemorySupportCalibrationError(
                f"{name} must contain only "
                f"{required_split.value} SyntheticEvent "
                "records."
            )

    event_ids = tuple(
        event.event_id
        for event in event_tuple
    )

    if len(set(event_ids)) != len(event_ids):
        raise MemorySupportCalibrationError(
            f"{name} event identifiers must be unique."
        )

    return event_tuple


def _mean_fuse_event(
    event: Any,
) -> FloatArray:
    """Mean-fuse three modalities without target metadata."""

    try:
        raw_vectors = (
            event.text_vector,
            event.image_vector,
            event.audio_vector,
        )
    except AttributeError as error:
        raise MemorySupportCalibrationError(
            "Event must provide text, image, "
            "and audio vectors."
        ) from error

    vectors = tuple(
        np.asarray(
            vector,
            dtype=np.float64,
        )
        for vector in raw_vectors
    )

    if any(
        vector.ndim != 1
        for vector in vectors
    ):
        raise MemorySupportCalibrationError(
            "Every modality must be a one-dimensional vector."
        )

    shapes = {
        vector.shape
        for vector in vectors
    }

    if len(shapes) != 1:
        raise MemorySupportCalibrationError(
            "Modality vectors must have matching dimensions."
        )

    if not all(
        np.all(np.isfinite(vector))
        for vector in vectors
    ):
        raise MemorySupportCalibrationError(
            "Modality vectors must be finite."
        )

    fused = np.mean(
        np.stack(vectors),
        axis=0,
    )
    norm = float(
        np.linalg.norm(fused)
    )

    if not isfinite(norm) or norm <= 0.0:
        raise MemorySupportCalibrationError(
            "Mean-fused event vector must be nonzero."
        )

    return np.asarray(
        fused / norm,
        dtype=np.float64,
    )


def _event_id(event: Any) -> str:
    """Read only an audit identifier, never target truth."""

    value = getattr(
        event,
        "event_id",
        None,
    )

    if value is None:
        value = getattr(
            event,
            "observed_event_id",
            None,
        )

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise MemorySupportCalibrationError(
            "Event must have a nonempty audit identifier."
        )

    return value


def _probability(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise MemorySupportCalibrationError(
            f"{name} must be numeric."
        )

    converted = float(value)

    if (
        not isfinite(converted)
        or not 0.0 <= converted <= 1.0
    ):
        raise MemorySupportCalibrationError(
            f"{name} must be finite and in [0, 1]."
        )

    return converted
