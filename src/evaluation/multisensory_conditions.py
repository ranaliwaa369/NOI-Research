"""Paired multisensory stress-view generation for NOI v0.3.

All seven views preserve the same latent-event ground truth. Transformations
modify only the prespecified evidence channel, and contradictory tactile
evidence is selected deterministically from a different target family.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real

from src.evaluation.multisensory_records import (
    ConditionLabel,
    LatentMultisensoryEvent,
    MultisensoryConditionView,
    MultisensoryTarget,
    SupportRegime,
    Vector,
)


class ConditionGenerationError(ValueError):
    """Raised when paired-view generation violates its locked contract."""


def _validate_nonnegative_integer(
    name: str,
    value: object,
) -> None:
    """Require a nonnegative integer while rejecting booleans."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ConditionGenerationError(
            f"{name} must be a nonnegative integer."
        )


def _validate_nonzero_integer(
    name: str,
    value: object,
) -> None:
    """Require a nonzero integer while rejecting booleans."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value == 0
    ):
        raise ConditionGenerationError(
            f"{name} must be a nonzero integer."
        )


def _validate_fraction(
    name: str,
    value: object,
) -> None:
    """Require a finite value strictly between zero and one."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
    ):
        raise ConditionGenerationError(
            f"{name} must be finite and between 0 and 1."
        )


def _validate_version(value: object) -> None:
    """Require explicit version provenance."""

    if not isinstance(value, str) or not value.strip():
        raise ConditionGenerationError(
            "generator_version must be a nonempty string."
        )


@dataclass(frozen=True, slots=True)
class ConditionGenerationConfig:
    """Locked controls for the seven paired stress transformations."""

    seed: int
    odor_noise_scale: float
    tactile_noise_scale: float
    degraded_quality: float
    locked_temporal_offset_steps: int
    generator_version: str

    def __post_init__(self) -> None:
        _validate_nonnegative_integer("seed", self.seed)
        _validate_fraction(
            "odor_noise_scale",
            self.odor_noise_scale,
        )
        _validate_fraction(
            "tactile_noise_scale",
            self.tactile_noise_scale,
        )
        _validate_fraction(
            "degraded_quality",
            self.degraded_quality,
        )
        _validate_nonzero_integer(
            "locked_temporal_offset_steps",
            self.locked_temporal_offset_steps,
        )
        _validate_version(self.generator_version)


@dataclass(frozen=True, slots=True)
class ConflictSource:
    """Provenance for one cross-family contradictory tactile vector."""

    view_id: str
    latent_event_id: str
    source_item_id: str
    source_family_id: int


@dataclass(frozen=True, slots=True)
class ConditionGenerationProvenance:
    """Explicit metadata for one paired-view generation run."""

    generator_version: str
    algorithm: str
    seed: int
    conditions: tuple[ConditionLabel, ...]
    odor_noise_scale: float
    tactile_noise_scale: float
    degraded_quality: float
    locked_temporal_offset_steps: int
    ground_truth_changed: bool
    support_regimes_changed: bool


@dataclass(frozen=True, slots=True)
class ConditionGenerationResult:
    """Immutable paired views and their audit metadata."""

    views: tuple[MultisensoryConditionView, ...]
    conflict_sources: tuple[ConflictSource, ...]
    support_regimes: tuple[tuple[str, SupportRegime], ...]
    provenance: ConditionGenerationProvenance


def _hash_unit_value(
    *,
    seed: int,
    namespace: str,
    latent_event_id: str,
    coordinate: int,
) -> float:
    """Return one deterministic value in [-1, 1]."""

    payload = (
        f"NOI-v0.3-conditions|{seed}|{namespace}|"
        f"{latent_event_id}|{coordinate}"
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()
    integer = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )
    unit = integer / ((1 << 64) - 1)

    return (2.0 * unit) - 1.0


def _clip_unit(value: float) -> float:
    """Clip one feature to [-1, 1]."""

    return max(-1.0, min(1.0, value))


def _add_noise(
    vector: Vector,
    *,
    scale: float,
    seed: int,
    namespace: str,
    latent_event_id: str,
) -> Vector:
    """Add deterministic bounded noise to exactly one modality."""

    return tuple(
        _clip_unit(
            value
            + scale
            * _hash_unit_value(
                seed=seed,
                namespace=namespace,
                latent_event_id=latent_event_id,
                coordinate=coordinate,
            ),
        )
        for coordinate, value in enumerate(vector)
    )


def _select_conflict_target(
    *,
    event: LatentMultisensoryEvent,
    targets: tuple[MultisensoryTarget, ...],
    seed: int,
) -> MultisensoryTarget:
    """Select a deterministic donor target from another family."""

    candidates = tuple(
        target
        for target in targets
        if target.family_id != event.target_family_id
    )

    if not candidates:
        raise ConditionGenerationError(
            "Conflict construction requires at least two target families."
        )

    ordered = tuple(
        sorted(
            candidates,
            key=lambda target: target.item_id,
        ),
    )
    digest = hashlib.sha256(
        (
            f"NOI-v0.3-conflict|{seed}|"
            f"{event.latent_event_id}"
        ).encode("utf-8"),
    ).digest()
    selected_index = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    ) % len(ordered)

    return ordered[selected_index]


def _make_view(
    *,
    event: LatentMultisensoryEvent,
    condition: ConditionLabel,
    olfactory_vector: Vector | None,
    tactile_vector: Vector | None,
    olfactory_quality: float,
    tactile_quality: float,
    modality_conflict: bool = False,
    temporal_offset_steps: int = 0,
) -> MultisensoryConditionView:
    """Construct one ground-truth-preserving paired view."""

    return MultisensoryConditionView(
        view_id=(
            f"{event.latent_event_id}-{condition.value}"
        ),
        latent_event_id=event.latent_event_id,
        condition=condition,
        target_item_id=event.target_item_id,
        target_family_id=event.target_family_id,
        olfactory_vector=olfactory_vector,
        tactile_vector=tactile_vector,
        olfactory_quality=olfactory_quality,
        tactile_quality=tactile_quality,
        modality_conflict=modality_conflict,
        temporal_offset_steps=temporal_offset_steps,
    )


def generate_multisensory_condition_views(
    latent_events: tuple[LatentMultisensoryEvent, ...],
    targets: tuple[MultisensoryTarget, ...],
    config: ConditionGenerationConfig,
) -> ConditionGenerationResult:
    """Generate all seven paired views for every supplied latent event."""

    if (
        not isinstance(latent_events, tuple)
        or not latent_events
        or not all(
            isinstance(event, LatentMultisensoryEvent)
            for event in latent_events
        )
    ):
        raise ConditionGenerationError(
            "latent_events must be a nonempty tuple of latent events."
        )

    if (
        not isinstance(targets, tuple)
        or not targets
        or not all(
            isinstance(target, MultisensoryTarget)
            for target in targets
        )
    ):
        raise ConditionGenerationError(
            "targets must be a nonempty tuple of multisensory targets."
        )

    if not isinstance(config, ConditionGenerationConfig):
        raise ConditionGenerationError(
            "config must be a ConditionGenerationConfig record."
        )

    target_lookup: dict[str, MultisensoryTarget] = {}

    for target in targets:
        if target.item_id in target_lookup:
            raise ConditionGenerationError(
                "Target identifiers must be unique."
            )
        target_lookup[target.item_id] = target

    if len({target.family_id for target in targets}) < 2:
        raise ConditionGenerationError(
            "Condition generation requires at least two target families."
        )

    event_identifiers: set[str] = set()

    for event in latent_events:
        if event.latent_event_id in event_identifiers:
            raise ConditionGenerationError(
                "Latent event identifiers must be unique."
            )
        event_identifiers.add(event.latent_event_id)

        target = target_lookup.get(event.target_item_id)

        if target is None:
            raise ConditionGenerationError(
                "Every event target must exist in the target registry."
            )

        if target.family_id != event.target_family_id:
            raise ConditionGenerationError(
                "Every event target-family assignment must match "
                "the target registry."
            )

    views: list[MultisensoryConditionView] = []
    conflict_sources: list[ConflictSource] = []

    for event in latent_events:
        clean = _make_view(
            event=event,
            condition=ConditionLabel.CLEAN,
            olfactory_vector=event.olfactory_vector,
            tactile_vector=event.tactile_vector,
            olfactory_quality=1.0,
            tactile_quality=1.0,
        )
        views.append(clean)

        degraded_odor = _make_view(
            event=event,
            condition=ConditionLabel.DEGRADED_ODOR,
            olfactory_vector=_add_noise(
                event.olfactory_vector,
                scale=config.odor_noise_scale,
                seed=config.seed,
                namespace="degraded-odor",
                latent_event_id=event.latent_event_id,
            ),
            tactile_vector=event.tactile_vector,
            olfactory_quality=config.degraded_quality,
            tactile_quality=1.0,
        )
        views.append(degraded_odor)

        degraded_touch = _make_view(
            event=event,
            condition=ConditionLabel.DEGRADED_TOUCH,
            olfactory_vector=event.olfactory_vector,
            tactile_vector=_add_noise(
                event.tactile_vector,
                scale=config.tactile_noise_scale,
                seed=config.seed,
                namespace="degraded-touch",
                latent_event_id=event.latent_event_id,
            ),
            olfactory_quality=1.0,
            tactile_quality=config.degraded_quality,
        )
        views.append(degraded_touch)

        missing_touch = _make_view(
            event=event,
            condition=ConditionLabel.MISSING_TOUCH,
            olfactory_vector=event.olfactory_vector,
            tactile_vector=None,
            olfactory_quality=1.0,
            tactile_quality=0.0,
        )
        views.append(missing_touch)

        missing_odor = _make_view(
            event=event,
            condition=ConditionLabel.MISSING_ODOR,
            olfactory_vector=None,
            tactile_vector=event.tactile_vector,
            olfactory_quality=0.0,
            tactile_quality=1.0,
        )
        views.append(missing_odor)

        donor = _select_conflict_target(
            event=event,
            targets=targets,
            seed=config.seed,
        )
        conflict = _make_view(
            event=event,
            condition=ConditionLabel.CONTRADICTORY_MODALITIES,
            olfactory_vector=event.olfactory_vector,
            tactile_vector=donor.tactile_prototype,
            olfactory_quality=1.0,
            tactile_quality=1.0,
            modality_conflict=True,
        )
        views.append(conflict)
        conflict_sources.append(
            ConflictSource(
                view_id=conflict.view_id,
                latent_event_id=event.latent_event_id,
                source_item_id=donor.item_id,
                source_family_id=donor.family_id,
            ),
        )

        temporal = _make_view(
            event=event,
            condition=ConditionLabel.TEMPORAL_MISALIGNMENT,
            olfactory_vector=event.olfactory_vector,
            tactile_vector=event.tactile_vector,
            olfactory_quality=1.0,
            tactile_quality=1.0,
            temporal_offset_steps=(
                config.locked_temporal_offset_steps
            ),
        )
        views.append(temporal)

    support_regimes = tuple(
        (
            event.latent_event_id,
            event.support_regime,
        )
        for event in latent_events
    )

    provenance = ConditionGenerationProvenance(
        generator_version=config.generator_version,
        algorithm=(
            "sha256-modality-specific-noise-"
            "cross-family-conflict-v1"
        ),
        seed=config.seed,
        conditions=tuple(ConditionLabel),
        odor_noise_scale=config.odor_noise_scale,
        tactile_noise_scale=config.tactile_noise_scale,
        degraded_quality=config.degraded_quality,
        locked_temporal_offset_steps=(
            config.locked_temporal_offset_steps
        ),
        ground_truth_changed=False,
        support_regimes_changed=False,
    )

    return ConditionGenerationResult(
        views=tuple(views),
        conflict_sources=tuple(conflict_sources),
        support_regimes=support_regimes,
        provenance=provenance,
    )
