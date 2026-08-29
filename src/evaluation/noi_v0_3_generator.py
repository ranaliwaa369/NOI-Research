"""Deterministic split and latent-event generation for NOI v0.3.

This module creates synthetic computational research records only. It keeps
training, validation-unknown, and final-test-unknown support partitions
explicitly separated and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real

from src.evaluation.multisensory_records import (
    LatentMultisensoryEvent,
    MultisensorySplit,
    MultisensoryTarget,
    SupportRegime,
    Vector,
)
from src.evaluation.tactile_generator import (
    TactileGenerationConfig,
    generate_tactile_prototypes,
)


OLFACTORY_DIMENSION = 16


class NOIV03GenerationError(ValueError):
    """Raised when v0.3 generation violates its registered contract."""


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
        raise NOIV03GenerationError(
            f"{name} must be a nonnegative integer."
        )


def _validate_positive_integer(
    name: str,
    value: object,
) -> None:
    """Require a positive integer while rejecting booleans."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise NOIV03GenerationError(
            f"{name} must be a positive integer."
        )


def _validate_nonempty_string(
    name: str,
    value: object,
) -> None:
    """Require explicit textual provenance."""

    if not isinstance(value, str) or not value.strip():
        raise NOIV03GenerationError(
            f"{name} must be a nonempty string."
        )


@dataclass(frozen=True, slots=True)
class NOIV03GenerationConfig:
    """Allocation and structural controls for one v0.3 seed."""

    seed: int
    train_event_count: int
    validation_event_count: int
    final_test_event_count: int
    validation_seen_item_count: int
    validation_known_family_unseen_item_count: int
    validation_unseen_family_count: int
    final_seen_item_count: int
    final_known_family_unseen_item_count: int
    final_unseen_family_count: int
    known_family_count: int
    training_items_per_family: int
    withheld_items_per_known_family: int
    validation_unknown_family_count: int
    final_unknown_family_count: int
    items_per_unknown_family: int
    generator_version: str
    feasibility_only: bool

    def __post_init__(self) -> None:
        _validate_nonnegative_integer("seed", self.seed)

        positive_fields = (
            "train_event_count",
            "validation_event_count",
            "final_test_event_count",
            "validation_seen_item_count",
            "validation_known_family_unseen_item_count",
            "validation_unseen_family_count",
            "final_seen_item_count",
            "final_known_family_unseen_item_count",
            "final_unseen_family_count",
            "known_family_count",
            "training_items_per_family",
            "withheld_items_per_known_family",
            "validation_unknown_family_count",
            "final_unknown_family_count",
            "items_per_unknown_family",
        )

        for field_name in positive_fields:
            _validate_positive_integer(
                field_name,
                getattr(self, field_name),
            )

        _validate_nonempty_string(
            "generator_version",
            self.generator_version,
        )

        if not isinstance(self.feasibility_only, bool):
            raise NOIV03GenerationError(
                "feasibility_only must be a boolean."
            )

        validation_total = (
            self.validation_seen_item_count
            + self.validation_known_family_unseen_item_count
            + self.validation_unseen_family_count
        )

        if validation_total != self.validation_event_count:
            raise NOIV03GenerationError(
                "validation support allocation must sum exactly "
                "to validation_event_count."
            )

        final_total = (
            self.final_seen_item_count
            + self.final_known_family_unseen_item_count
            + self.final_unseen_family_count
        )

        if final_total != self.final_test_event_count:
            raise NOIV03GenerationError(
                "final-test support allocation must sum exactly "
                "to final_test_event_count."
            )

        if self.withheld_items_per_known_family < 2:
            raise NOIV03GenerationError(
                "withheld_items_per_known_family must be at least 2 "
                "to separate validation and final-test items."
            )

        if not self.feasibility_only:
            if (
                self.train_event_count,
                self.validation_event_count,
                self.final_test_event_count,
            ) != (7000, 1000, 2000):
                raise NOIV03GenerationError(
                    "Production execution requires the exact "
                    "7000/1000/2000 split allocation."
                )

            if (
                self.final_seen_item_count,
                self.final_known_family_unseen_item_count,
                self.final_unseen_family_count,
            ) != (800, 600, 600):
                raise NOIV03GenerationError(
                    "Production final-test execution requires the exact "
                    "800/600/600 support allocation."
                )


@dataclass(frozen=True, slots=True)
class ReachabilityMetadata:
    """Auditable support sets derived without final-test calibration."""

    training_item_ids: tuple[str, ...]
    training_family_ids: tuple[int, ...]
    validation_known_family_unseen_item_ids: tuple[str, ...]
    final_test_known_family_unseen_item_ids: tuple[str, ...]
    validation_unknown_family_ids: tuple[int, ...]
    final_test_unknown_family_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NOIV03GenerationProvenance:
    """Explicit generation metadata for one v0.3 latent dataset."""

    generator_version: str
    algorithm: str
    seed: int
    train_event_count: int
    validation_event_count: int
    final_test_event_count: int
    validation_support_allocation: tuple[int, int, int]
    final_support_allocation: tuple[int, int, int]
    olfactory_dimension: int
    tactile_dimension: int
    feasibility_only: bool
    test_labels_used_for_generation: bool


@dataclass(frozen=True, slots=True)
class NOIV03GenerationResult:
    """Immutable targets, latent events, reachability, and provenance."""

    targets: tuple[MultisensoryTarget, ...]
    latent_events: tuple[LatentMultisensoryEvent, ...]
    reachability: ReachabilityMetadata
    provenance: NOIV03GenerationProvenance


def _hash_unit_value(
    *,
    seed: int,
    namespace: str,
    family_index: int,
    item_index: int,
    coordinate: int,
) -> float:
    """Return a deterministic value in [-1, 1]."""

    payload = (
        f"NOI-v0.3-events|{seed}|{namespace}|"
        f"{family_index}|{item_index}|{coordinate}"
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
    """Clip one generated coordinate to [-1, 1]."""

    return max(-1.0, min(1.0, value))


def _olfactory_prototype(
    *,
    seed: int,
    family_index: int,
    item_index: int,
) -> Vector:
    """Generate a structured 16-dimensional olfactory prototype."""

    values: list[float] = []

    for coordinate in range(OLFACTORY_DIMENSION):
        family_value = 0.65 * _hash_unit_value(
            seed=seed,
            namespace="olfactory-family",
            family_index=family_index,
            item_index=0,
            coordinate=coordinate,
        )
        item_residual = 0.12 * _hash_unit_value(
            seed=seed,
            namespace="olfactory-item",
            family_index=family_index,
            item_index=item_index,
            coordinate=coordinate,
        )

        values.append(
            _clip_unit(family_value + item_residual),
        )

    return tuple(values)


def _ordered_selection(
    records: tuple[MultisensoryTarget, ...],
    count: int,
    *,
    seed: int,
    namespace: str,
) -> tuple[MultisensoryTarget, ...]:
    """Select deterministically while keeping every source record reachable."""

    if not records:
        raise NOIV03GenerationError(
            f"{namespace} target pool cannot be empty."
        )

    namespace_digest = hashlib.sha256(
        f"{seed}|{namespace}".encode("utf-8"),
    ).digest()
    offset = int.from_bytes(
        namespace_digest[:4],
        byteorder="big",
        signed=False,
    ) % len(records)

    ordered = records[offset:] + records[:offset]

    return tuple(
        ordered[index % len(ordered)]
        for index in range(count)
    )


def _make_event(
    *,
    target: MultisensoryTarget,
    split: MultisensorySplit,
    support_regime: SupportRegime,
    split_index: int,
    seed: int,
) -> LatentMultisensoryEvent:
    """Construct one immutable condition-independent latent event."""

    return LatentMultisensoryEvent(
        latent_event_id=(
            f"seed-{seed}-{split.value}-event-{split_index:06d}"
        ),
        split=split,
        template_id=split_index % 16,
        target_item_id=target.item_id,
        target_family_id=target.family_id,
        support_regime=support_regime,
        olfactory_vector=target.olfactory_prototype,
        tactile_vector=target.tactile_prototype,
        generator_seed=seed,
    )


def _target_id(
    *,
    category: str,
    family_index: int,
    item_index: int,
) -> str:
    """Create a deterministic identifier that is not used as a feature."""

    return (
        f"{category}-family-{family_index:03d}"
        f"-item-{item_index:03d}"
    )


def generate_noi_v0_3_events(
    config: NOIV03GenerationConfig,
) -> NOIV03GenerationResult:
    """Generate v0.3 targets and leakage-controlled latent events."""

    if not isinstance(config, NOIV03GenerationConfig):
        raise NOIV03GenerationError(
            "config must be an NOIV03GenerationConfig record."
        )

    known_item_total = (
        config.training_items_per_family
        + config.withheld_items_per_known_family
    )
    tactile_items_per_family = max(
        known_item_total,
        config.items_per_unknown_family,
    )
    total_family_count = (
        config.known_family_count
        + config.validation_unknown_family_count
        + config.final_unknown_family_count
    )

    tactile_result = generate_tactile_prototypes(
        TactileGenerationConfig(
            seed=config.seed,
            family_count=total_family_count,
            items_per_family=tactile_items_per_family,
            family_scale=0.65,
            item_residual_scale=0.12,
            generator_version=config.generator_version,
        ),
    )

    tactile_lookup = {
        (family_index, item_index): vector
        for vector, family_index, item_index in zip(
            tactile_result.prototypes,
            tactile_result.family_indices,
            tactile_result.item_indices,
            strict=True,
        )
    }

    all_targets: list[MultisensoryTarget] = []
    training_targets: list[MultisensoryTarget] = []
    validation_withheld_targets: list[MultisensoryTarget] = []
    final_withheld_targets: list[MultisensoryTarget] = []
    validation_unknown_targets: list[MultisensoryTarget] = []
    final_unknown_targets: list[MultisensoryTarget] = []

    for family_index in range(config.known_family_count):
        for item_index in range(config.training_items_per_family):
            target = MultisensoryTarget(
                item_id=_target_id(
                    category="known-training",
                    family_index=family_index,
                    item_index=item_index,
                ),
                family_id=family_index,
                olfactory_prototype=_olfactory_prototype(
                    seed=config.seed,
                    family_index=family_index,
                    item_index=item_index,
                ),
                tactile_prototype=tactile_lookup[
                    (family_index, item_index)
                ],
            )
            all_targets.append(target)
            training_targets.append(target)

        for withheld_index in range(
            config.withheld_items_per_known_family,
        ):
            item_index = (
                config.training_items_per_family
                + withheld_index
            )
            target = MultisensoryTarget(
                item_id=_target_id(
                    category="known-withheld",
                    family_index=family_index,
                    item_index=item_index,
                ),
                family_id=family_index,
                olfactory_prototype=_olfactory_prototype(
                    seed=config.seed,
                    family_index=family_index,
                    item_index=item_index,
                ),
                tactile_prototype=tactile_lookup[
                    (family_index, item_index)
                ],
            )
            all_targets.append(target)

            if withheld_index % 2 == 0:
                validation_withheld_targets.append(target)
            else:
                final_withheld_targets.append(target)

    validation_family_start = config.known_family_count
    validation_family_stop = (
        validation_family_start
        + config.validation_unknown_family_count
    )

    for family_index in range(
        validation_family_start,
        validation_family_stop,
    ):
        for item_index in range(config.items_per_unknown_family):
            target = MultisensoryTarget(
                item_id=_target_id(
                    category="validation-unknown",
                    family_index=family_index,
                    item_index=item_index,
                ),
                family_id=family_index,
                olfactory_prototype=_olfactory_prototype(
                    seed=config.seed,
                    family_index=family_index,
                    item_index=item_index,
                ),
                tactile_prototype=tactile_lookup[
                    (family_index, item_index)
                ],
            )
            all_targets.append(target)
            validation_unknown_targets.append(target)

    final_family_start = validation_family_stop
    final_family_stop = (
        final_family_start
        + config.final_unknown_family_count
    )

    for family_index in range(
        final_family_start,
        final_family_stop,
    ):
        for item_index in range(config.items_per_unknown_family):
            target = MultisensoryTarget(
                item_id=_target_id(
                    category="final-unknown",
                    family_index=family_index,
                    item_index=item_index,
                ),
                family_id=family_index,
                olfactory_prototype=_olfactory_prototype(
                    seed=config.seed,
                    family_index=family_index,
                    item_index=item_index,
                ),
                tactile_prototype=tactile_lookup[
                    (family_index, item_index)
                ],
            )
            all_targets.append(target)
            final_unknown_targets.append(target)

    training_selection = _ordered_selection(
        tuple(training_targets),
        config.train_event_count,
        seed=config.seed,
        namespace="training",
    )

    validation_groups = (
        (
            SupportRegime.SEEN_ITEM,
            _ordered_selection(
                tuple(training_targets),
                config.validation_seen_item_count,
                seed=config.seed,
                namespace="validation-seen",
            ),
        ),
        (
            SupportRegime.KNOWN_FAMILY_UNSEEN_ITEM,
            _ordered_selection(
                tuple(validation_withheld_targets),
                config.validation_known_family_unseen_item_count,
                seed=config.seed,
                namespace="validation-known-unseen",
            ),
        ),
        (
            SupportRegime.UNSEEN_FAMILY,
            _ordered_selection(
                tuple(validation_unknown_targets),
                config.validation_unseen_family_count,
                seed=config.seed,
                namespace="validation-unseen-family",
            ),
        ),
    )

    final_groups = (
        (
            SupportRegime.SEEN_ITEM,
            _ordered_selection(
                tuple(training_targets),
                config.final_seen_item_count,
                seed=config.seed,
                namespace="final-seen",
            ),
        ),
        (
            SupportRegime.KNOWN_FAMILY_UNSEEN_ITEM,
            _ordered_selection(
                tuple(final_withheld_targets),
                config.final_known_family_unseen_item_count,
                seed=config.seed,
                namespace="final-known-unseen",
            ),
        ),
        (
            SupportRegime.UNSEEN_FAMILY,
            _ordered_selection(
                tuple(final_unknown_targets),
                config.final_unseen_family_count,
                seed=config.seed,
                namespace="final-unseen-family",
            ),
        ),
    )

    latent_events: list[LatentMultisensoryEvent] = []

    for index, target in enumerate(training_selection):
        latent_events.append(
            _make_event(
                target=target,
                split=MultisensorySplit.TRAIN,
                support_regime=SupportRegime.DEVELOPMENT,
                split_index=index,
                seed=config.seed,
            ),
        )

    validation_index = 0

    for support_regime, targets in validation_groups:
        for target in targets:
            latent_events.append(
                _make_event(
                    target=target,
                    split=MultisensorySplit.VALIDATION,
                    support_regime=support_regime,
                    split_index=validation_index,
                    seed=config.seed,
                ),
            )
            validation_index += 1

    final_index = 0

    for support_regime, targets in final_groups:
        for target in targets:
            latent_events.append(
                _make_event(
                    target=target,
                    split=MultisensorySplit.FINAL_TEST,
                    support_regime=support_regime,
                    split_index=final_index,
                    seed=config.seed,
                ),
            )
            final_index += 1

    reachability = ReachabilityMetadata(
        training_item_ids=tuple(
            target.item_id
            for target in training_targets
        ),
        training_family_ids=tuple(
            range(config.known_family_count)
        ),
        validation_known_family_unseen_item_ids=tuple(
            target.item_id
            for target in validation_withheld_targets
        ),
        final_test_known_family_unseen_item_ids=tuple(
            target.item_id
            for target in final_withheld_targets
        ),
        validation_unknown_family_ids=tuple(
            range(
                validation_family_start,
                validation_family_stop,
            )
        ),
        final_test_unknown_family_ids=tuple(
            range(
                final_family_start,
                final_family_stop,
            )
        ),
    )

    provenance = NOIV03GenerationProvenance(
        generator_version=config.generator_version,
        algorithm=(
            "sha256-structured-olfactory-plus-"
            "deterministic-tactile-v1"
        ),
        seed=config.seed,
        train_event_count=config.train_event_count,
        validation_event_count=config.validation_event_count,
        final_test_event_count=config.final_test_event_count,
        validation_support_allocation=(
            config.validation_seen_item_count,
            config.validation_known_family_unseen_item_count,
            config.validation_unseen_family_count,
        ),
        final_support_allocation=(
            config.final_seen_item_count,
            config.final_known_family_unseen_item_count,
            config.final_unseen_family_count,
        ),
        olfactory_dimension=16,
        tactile_dimension=8,
        feasibility_only=config.feasibility_only,
        test_labels_used_for_generation=False,
    )

    return NOIV03GenerationResult(
        targets=tuple(all_targets),
        latent_events=tuple(latent_events),
        reachability=reachability,
        provenance=provenance,
    )
