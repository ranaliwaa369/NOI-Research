"""Deterministic tactile prototype generation for NOI v0.3.

The generator creates synthetic computational features only. It does not
represent measurements from a physical tactile sensor and does not establish
biological, clinical, chemical, or deployment validity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real

from src.evaluation.multisensory_records import TACTILE_DIMENSION, Vector


TACTILE_FEATURE_LAYOUT: tuple[str, ...] = (
    "roughness",
    "stiffness",
    "friction",
    "contact_geometry_1",
    "contact_geometry_2",
    "pressure_response_1",
    "pressure_response_2",
    "pressure_response_3",
)


class TactileGenerationError(ValueError):
    """Raised when tactile generation violates its fixed contract."""


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
        raise TactileGenerationError(
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
        raise TactileGenerationError(
            f"{name} must be a positive integer."
        )


def _validate_scale(
    name: str,
    value: object,
) -> None:
    """Require a finite scale in the interval (0, 1]."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) <= 1.0
    ):
        raise TactileGenerationError(
            f"{name} must be finite and between 0 and 1."
        )


def _validate_version(value: object) -> None:
    """Require an explicit nonempty generator version."""

    if not isinstance(value, str) or not value.strip():
        raise TactileGenerationError(
            "generator_version must be a nonempty string."
        )


def _bounded_hash_value(
    *,
    seed: int,
    namespace: str,
    family_index: int,
    item_index: int,
    coordinate: int,
) -> float:
    """Map explicit deterministic metadata to a value in [-1, 1].

    SHA-256 is used as a reproducible counter-style source. The function
    receives positional indices only; it never receives dataset identifiers,
    split labels, or support labels.
    """

    payload = (
        f"NOI-v0.3|{seed}|{namespace}|"
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
    """Clip one finite feature to the closed unit interval."""

    return max(-1.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class TactileGenerationConfig:
    """Validated feasibility configuration for tactile generation."""

    seed: int
    family_count: int
    items_per_family: int
    family_scale: float
    item_residual_scale: float
    generator_version: str

    def __post_init__(self) -> None:
        _validate_nonnegative_integer("seed", self.seed)
        _validate_positive_integer(
            "family_count",
            self.family_count,
        )
        _validate_positive_integer(
            "items_per_family",
            self.items_per_family,
        )
        _validate_scale("family_scale", self.family_scale)
        _validate_scale(
            "item_residual_scale",
            self.item_residual_scale,
        )
        _validate_version(self.generator_version)

        if self.item_residual_scale >= self.family_scale:
            raise TactileGenerationError(
                "item_residual_scale must be smaller than family_scale."
            )


@dataclass(frozen=True, slots=True)
class TactileGenerationProvenance:
    """Explicit record of how one tactile batch was generated."""

    generator_version: str
    algorithm: str
    seed: int
    dimension: int
    feature_layout: tuple[str, ...]
    family_count: int
    items_per_family: int
    family_scale: float
    item_residual_scale: float
    feasibility_only: bool


@dataclass(frozen=True, slots=True)
class TactileGenerationResult:
    """Immutable generated tactile prototypes and positional metadata."""

    prototypes: tuple[Vector, ...]
    family_indices: tuple[int, ...]
    item_indices: tuple[int, ...]
    provenance: TactileGenerationProvenance


def _family_prototype(
    config: TactileGenerationConfig,
    family_index: int,
) -> Vector:
    """Create one family-level tactile center."""

    return tuple(
        config.family_scale
        * _bounded_hash_value(
            seed=config.seed,
            namespace="family",
            family_index=family_index,
            item_index=0,
            coordinate=coordinate,
        )
        for coordinate in range(TACTILE_DIMENSION)
    )


def _item_prototype(
    config: TactileGenerationConfig,
    family_index: int,
    item_index: int,
    family_prototype: Vector,
) -> Vector:
    """Add a bounded item residual to one family center."""

    values: list[float] = []

    for coordinate, family_value in enumerate(family_prototype):
        residual = (
            config.item_residual_scale
            * _bounded_hash_value(
                seed=config.seed,
                namespace="item-residual",
                family_index=family_index,
                item_index=item_index,
                coordinate=coordinate,
            )
        )

        values.append(
            _clip_unit(family_value + residual),
        )

    return tuple(values)


def generate_tactile_prototypes(
    config: TactileGenerationConfig,
) -> TactileGenerationResult:
    """Generate deterministic, bounded tactile prototypes.

    The public interface intentionally accepts only a validated configuration.
    Dataset IDs, split labels, and support labels cannot influence features.
    """

    if not isinstance(config, TactileGenerationConfig):
        raise TactileGenerationError(
            "config must be a TactileGenerationConfig record."
        )

    prototypes: list[Vector] = []
    family_indices: list[int] = []
    item_indices: list[int] = []

    for family_index in range(config.family_count):
        family_prototype = _family_prototype(
            config,
            family_index,
        )

        for item_index in range(config.items_per_family):
            prototypes.append(
                _item_prototype(
                    config,
                    family_index,
                    item_index,
                    family_prototype,
                ),
            )
            family_indices.append(family_index)
            item_indices.append(item_index)

    provenance = TactileGenerationProvenance(
        generator_version=config.generator_version,
        algorithm="sha256-family-plus-item-residual-v1",
        seed=config.seed,
        dimension=TACTILE_DIMENSION,
        feature_layout=TACTILE_FEATURE_LAYOUT,
        family_count=config.family_count,
        items_per_family=config.items_per_family,
        family_scale=config.family_scale,
        item_residual_scale=config.item_residual_scale,
        feasibility_only=True,
    )

    return TactileGenerationResult(
        prototypes=tuple(prototypes),
        family_indices=tuple(family_indices),
        item_indices=tuple(item_indices),
        provenance=provenance,
    )
