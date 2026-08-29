"""Validated records for NOI v0.3 multisensory evaluation.

This module defines immutable olfactory-tactile research records without
changing the legacy v0.1/v0.2 synthetic-event representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import TypeAlias


Vector: TypeAlias = tuple[float, ...]

OLFACTORY_DIMENSION = 16
TACTILE_DIMENSION = 8


class MultisensoryRecordError(ValueError):
    """Raised when a multisensory research record violates the protocol."""


class MultisensorySplit(str, Enum):
    """Dataset partitions used by the v0.3 protocol."""

    TRAIN = "train"
    VALIDATION = "validation"
    FINAL_TEST = "final_test"


class SupportRegime(str, Enum):
    """Ground-truth support relationship for one latent event."""

    DEVELOPMENT = "development"
    SEEN_ITEM = "seen_item"
    KNOWN_FAMILY_UNSEEN_ITEM = "known_family_unseen_item"
    UNSEEN_FAMILY = "unseen_family"


class ConditionLabel(str, Enum):
    """Prespecified paired evaluation conditions."""

    CLEAN = "clean"
    DEGRADED_ODOR = "degraded_odor"
    DEGRADED_TOUCH = "degraded_touch"
    MISSING_TOUCH = "missing_touch"
    MISSING_ODOR = "missing_odor"
    CONTRADICTORY_MODALITIES = "contradictory_modalities"
    TEMPORAL_MISALIGNMENT = "temporal_misalignment"


def _validate_identifier(name: str, value: object) -> None:
    """Require a nonempty textual identifier."""

    if not isinstance(value, str) or not value.strip():
        raise MultisensoryRecordError(
            f"{name} must be a nonempty string."
        )


def _validate_nonnegative_integer(name: str, value: object) -> None:
    """Require a nonnegative integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MultisensoryRecordError(
            f"{name} must be a nonnegative integer."
        )


def _validate_vector(
    name: str,
    vector: object,
    expected_dimension: int,
) -> None:
    """Require one finite numeric vector with an exact dimension."""

    if not isinstance(vector, tuple):
        raise MultisensoryRecordError(
            f"{name} must be a tuple containing exactly "
            f"{expected_dimension} values."
        )

    if len(vector) != expected_dimension:
        raise MultisensoryRecordError(
            f"{name} must contain exactly {expected_dimension} values."
        )

    for value in vector:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise MultisensoryRecordError(
                f"{name} must contain only finite numeric values."
            )


def _validate_optional_vector(
    name: str,
    vector: object,
    expected_dimension: int,
) -> None:
    """Validate a modality vector when the modality is available."""

    if vector is not None:
        _validate_vector(
            name,
            vector,
            expected_dimension,
        )


def _validate_quality(name: str, value: object) -> None:
    """Require a finite modality-quality value in the closed unit interval."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise MultisensoryRecordError(
            f"{name} must be finite and between 0 and 1."
        )


def _validate_enum(name: str, value: object, enum_type: type[Enum]) -> None:
    """Require an already parsed protocol enumeration value."""

    if not isinstance(value, enum_type):
        raise MultisensoryRecordError(
            f"{name} must be a {enum_type.__name__} value."
        )


@dataclass(frozen=True, slots=True)
class MultisensoryTarget:
    """One synthetic target with olfactory and tactile prototypes."""

    item_id: str
    family_id: int
    olfactory_prototype: Vector
    tactile_prototype: Vector

    def __post_init__(self) -> None:
        _validate_identifier("item_id", self.item_id)
        _validate_nonnegative_integer("family_id", self.family_id)
        _validate_vector(
            "olfactory_prototype",
            self.olfactory_prototype,
            OLFACTORY_DIMENSION,
        )
        _validate_vector(
            "tactile_prototype",
            self.tactile_prototype,
            TACTILE_DIMENSION,
        )


@dataclass(frozen=True, slots=True)
class LatentMultisensoryEvent:
    """Condition-independent multisensory event and its ground truth."""

    latent_event_id: str
    split: MultisensorySplit
    template_id: str
    target_item_id: str
    target_family_id: int
    support_regime: SupportRegime
    olfactory_vector: Vector
    tactile_vector: Vector
    generator_seed: int

    def __post_init__(self) -> None:
        _validate_identifier(
            "latent_event_id",
            self.latent_event_id,
        )
        _validate_enum(
            "split",
            self.split,
            MultisensorySplit,
        )
        _validate_nonnegative_integer(
            "template_id",
            self.template_id,
        )
        _validate_identifier(
            "target_item_id",
            self.target_item_id,
        )
        _validate_nonnegative_integer(
            "target_family_id",
            self.target_family_id,
        )
        _validate_enum(
            "support_regime",
            self.support_regime,
            SupportRegime,
        )
        _validate_vector(
            "olfactory_vector",
            self.olfactory_vector,
            OLFACTORY_DIMENSION,
        )
        _validate_vector(
            "tactile_vector",
            self.tactile_vector,
            TACTILE_DIMENSION,
        )
        _validate_nonnegative_integer(
            "generator_seed",
            self.generator_seed,
        )

        if (
            self.split is MultisensorySplit.TRAIN
            and self.support_regime is not SupportRegime.DEVELOPMENT
        ):
            raise MultisensoryRecordError(
                "A training event must use the development support regime."
            )

        if (
            self.split is MultisensorySplit.FINAL_TEST
            and self.support_regime is SupportRegime.DEVELOPMENT
        ):
            raise MultisensoryRecordError(
                "A final-test event cannot use the development "
                "support regime."
            )


@dataclass(frozen=True, slots=True)
class MultisensoryConditionView:
    """One prespecified observed condition derived from a latent event."""

    view_id: str
    latent_event_id: str
    condition: ConditionLabel
    target_item_id: str
    target_family_id: int
    olfactory_vector: Vector | None
    tactile_vector: Vector | None
    olfactory_quality: float
    tactile_quality: float
    modality_conflict: bool
    temporal_offset_steps: int

    def __post_init__(self) -> None:
        _validate_identifier("view_id", self.view_id)
        _validate_identifier(
            "latent_event_id",
            self.latent_event_id,
        )
        _validate_enum(
            "condition",
            self.condition,
            ConditionLabel,
        )
        _validate_identifier(
            "target_item_id",
            self.target_item_id,
        )
        _validate_nonnegative_integer(
            "target_family_id",
            self.target_family_id,
        )
        _validate_optional_vector(
            "olfactory_vector",
            self.olfactory_vector,
            OLFACTORY_DIMENSION,
        )
        _validate_optional_vector(
            "tactile_vector",
            self.tactile_vector,
            TACTILE_DIMENSION,
        )
        _validate_quality(
            "olfactory_quality",
            self.olfactory_quality,
        )
        _validate_quality(
            "tactile_quality",
            self.tactile_quality,
        )

        if not isinstance(self.modality_conflict, bool):
            raise MultisensoryRecordError(
                "modality_conflict must be a boolean."
            )

        if (
            isinstance(self.temporal_offset_steps, bool)
            or not isinstance(self.temporal_offset_steps, int)
        ):
            raise MultisensoryRecordError(
                "temporal_offset_steps must be an integer."
            )

        if not self.olfactory_available and self.olfactory_quality != 0:
            raise MultisensoryRecordError(
                "Olfactory quality must be zero when odor is absent."
            )

        if self.olfactory_available and self.olfactory_quality <= 0:
            raise MultisensoryRecordError(
                "Olfactory quality must be positive when odor is present."
            )

        if not self.tactile_available and self.tactile_quality != 0:
            raise MultisensoryRecordError(
                "Tactile quality must be zero when touch is absent."
            )

        if self.tactile_available and self.tactile_quality <= 0:
            raise MultisensoryRecordError(
                "Tactile quality must be positive when touch is present."
            )

        if not self.olfactory_available and not self.tactile_available:
            raise MultisensoryRecordError(
                "At least one modality must be available."
            )

        contradictory = (
            self.condition
            is ConditionLabel.CONTRADICTORY_MODALITIES
        )

        if contradictory != self.modality_conflict:
            raise MultisensoryRecordError(
                "The conflict flag must match the contradictory-modalities "
                "condition."
            )

        temporally_misaligned = (
            self.condition
            is ConditionLabel.TEMPORAL_MISALIGNMENT
        )

        if temporally_misaligned and self.temporal_offset_steps == 0:
            raise MultisensoryRecordError(
                "Temporal misalignment requires a nonzero temporal offset."
            )

        if not temporally_misaligned and self.temporal_offset_steps != 0:
            raise MultisensoryRecordError(
                "A non-temporal condition requires a zero temporal offset."
            )

    @property
    def olfactory_available(self) -> bool:
        """Return whether olfactory evidence is present."""

        return self.olfactory_vector is not None

    @property
    def tactile_available(self) -> bool:
        """Return whether tactile evidence is present."""

        return self.tactile_vector is not None


@dataclass(frozen=True, slots=True)
class MultisensoryDataset:
    """Validated collection of targets, latent events, and paired views."""

    targets: tuple[MultisensoryTarget, ...]
    latent_events: tuple[LatentMultisensoryEvent, ...]
    condition_views: tuple[MultisensoryConditionView, ...]
    generator_version: str
    generator_seed: int

    def __post_init__(self) -> None:
        _validate_identifier(
            "generator_version",
            self.generator_version,
        )
        _validate_nonnegative_integer(
            "generator_seed",
            self.generator_seed,
        )

        if not isinstance(self.targets, tuple) or not self.targets:
            raise MultisensoryRecordError(
                "targets must be a nonempty tuple."
            )

        if not isinstance(self.latent_events, tuple) or not self.latent_events:
            raise MultisensoryRecordError(
                "latent_events must be a nonempty tuple."
            )

        if not isinstance(self.condition_views, tuple) or not self.condition_views:
            raise MultisensoryRecordError(
                "condition_views must be a nonempty tuple."
            )

        if not all(
            isinstance(target, MultisensoryTarget)
            for target in self.targets
        ):
            raise MultisensoryRecordError(
                "targets must contain MultisensoryTarget records."
            )

        if not all(
            isinstance(event, LatentMultisensoryEvent)
            for event in self.latent_events
        ):
            raise MultisensoryRecordError(
                "latent_events must contain LatentMultisensoryEvent records."
            )

        if not all(
            isinstance(view, MultisensoryConditionView)
            for view in self.condition_views
        ):
            raise MultisensoryRecordError(
                "condition_views must contain "
                "MultisensoryConditionView records."
            )

        target_lookup: dict[str, MultisensoryTarget] = {}

        for target in self.targets:
            if target.item_id in target_lookup:
                raise MultisensoryRecordError(
                    "Dataset target identifiers must be unique."
                )
            target_lookup[target.item_id] = target

        event_lookup: dict[str, LatentMultisensoryEvent] = {}

        for event in self.latent_events:
            if event.latent_event_id in event_lookup:
                raise MultisensoryRecordError(
                    "Dataset latent event identifiers must be unique."
                )

            target = target_lookup.get(event.target_item_id)

            if target is None:
                raise MultisensoryRecordError(
                    "Every event target must exist in the target registry."
                )

            if target.family_id != event.target_family_id:
                raise MultisensoryRecordError(
                    "The event target-family assignment must match "
                    "the target registry."
                )

            event_lookup[event.latent_event_id] = event

        view_identifiers: set[str] = set()
        condition_pairs: set[tuple[str, ConditionLabel]] = set()

        for view in self.condition_views:
            if view.view_id in view_identifiers:
                raise MultisensoryRecordError(
                    "Dataset view identifiers must be unique."
                )
            view_identifiers.add(view.view_id)

            event = event_lookup.get(view.latent_event_id)

            if event is None:
                raise MultisensoryRecordError(
                    "Every view latent event must exist in the event registry."
                )

            if (
                view.target_item_id != event.target_item_id
                or view.target_family_id != event.target_family_id
            ):
                raise MultisensoryRecordError(
                    "Condition view ground truth must match its latent event."
                )

            condition_key = (
                view.latent_event_id,
                view.condition,
            )

            if condition_key in condition_pairs:
                raise MultisensoryRecordError(
                    "A duplicate condition view exists for one latent event."
                )

            condition_pairs.add(condition_key)
