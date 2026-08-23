"""Paired graded-OOD transformations for synthetic NOI evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

import numpy as np

from src.evaluation.synthetic_records import SyntheticEvent


Vector = tuple[float, ...]

EXPECTED_TIER_STRENGTHS = {
    "mild": 0.25,
    "moderate": 0.50,
    "severe": 1.00,
}

EXPECTED_TIER_SEEDS = {
    "mild": 7001,
    "moderate": 8001,
    "severe": 9001,
}


class GradedOODError(ValueError):
    """Raised when graded-OOD views cannot be created safely."""


class OODTier(str, Enum):
    """Preregistered OOD severity levels."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass(frozen=True)
class OODTierSpecification:
    """Locked metadata for one OOD tier."""

    tier: OODTier
    shift_strength: float
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.tier, OODTier):
            raise GradedOODError(
                "tier must be an OODTier."
            )

        expected_strength = EXPECTED_TIER_STRENGTHS[
            self.tier.value
        ]
        expected_seed = EXPECTED_TIER_SEEDS[
            self.tier.value
        ]

        if (
            isinstance(self.shift_strength, bool)
            or not isinstance(self.shift_strength, (int, float))
            or not isfinite(float(self.shift_strength))
        ):
            raise GradedOODError(
                "shift_strength must be a finite number."
            )

        if float(self.shift_strength) != expected_strength:
            raise GradedOODError(
                f"{self.tier.value} shift_strength must be "
                f"{expected_strength}."
            )

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise GradedOODError(
                "seed must be a nonnegative integer."
            )

        if self.seed != expected_seed:
            raise GradedOODError(
                f"{self.tier.value} seed must be {expected_seed}."
            )


@dataclass(frozen=True)
class GradedOODEvent:
    """One paired severity view of a latent synthetic event."""

    latent_event_id: str
    observed_event_id: str
    tier: OODTier
    shift_strength: float
    tier_seed: int
    template_id: int
    target_item_id: str
    target_family_id: int
    text_vector: Vector
    image_vector: Vector
    audio_vector: Vector

    def __post_init__(self) -> None:
        if (
            not isinstance(self.latent_event_id, str)
            or not self.latent_event_id.strip()
        ):
            raise GradedOODError(
                "latent_event_id must not be empty."
            )

        expected_id = (
            f"{self.latent_event_id}::{self.tier.value}"
        )

        if self.observed_event_id != expected_id:
            raise GradedOODError(
                "observed_event_id must combine latent_event_id and tier."
            )

        OODTierSpecification(
            tier=self.tier,
            shift_strength=self.shift_strength,
            seed=self.tier_seed,
        )

        if (
            isinstance(self.template_id, bool)
            or not isinstance(self.template_id, int)
            or self.template_id < 0
        ):
            raise GradedOODError(
                "template_id must be a nonnegative integer."
            )

        if not self.target_item_id.strip():
            raise GradedOODError(
                "target_item_id must not be empty."
            )

        if (
            isinstance(self.target_family_id, bool)
            or not isinstance(self.target_family_id, int)
            or self.target_family_id < 0
        ):
            raise GradedOODError(
                "target_family_id must be a nonnegative integer."
            )

        vectors = (
            self.text_vector,
            self.image_vector,
            self.audio_vector,
        )

        dimensions = {len(vector) for vector in vectors}

        if len(dimensions) != 1 or 0 in dimensions:
            raise GradedOODError(
                "All modality vectors must share one positive dimension."
            )

        for vector in vectors:
            array = np.asarray(vector, dtype=np.float64)

            if array.ndim != 1:
                raise GradedOODError(
                    "Every modality vector must be one-dimensional."
                )

            if not np.all(np.isfinite(array)):
                raise GradedOODError(
                    "Modality vectors must contain only finite values."
                )

            norm = float(np.linalg.norm(array))

            if (
                not np.isfinite(norm)
                or abs(norm - 1.0) > 1e-10
            ):
                raise GradedOODError(
                    "Every modality vector must have unit L2 norm."
                )


def tier_specifications_from_amendment(
    amendment: dict[str, Any],
) -> tuple[OODTierSpecification, ...]:
    """Extract locked tier specifications from the amendment."""

    if not isinstance(amendment, dict):
        raise GradedOODError(
            "amendment must be a dictionary."
        )

    try:
        tiers = amendment["graded_ood_design"]["tiers"]
    except (KeyError, TypeError) as error:
        raise GradedOODError(
            "The amendment does not contain graded OOD tiers."
        ) from error

    if not isinstance(tiers, dict):
        raise GradedOODError(
            "The graded OOD tiers must be a mapping."
        )

    if tuple(tiers) != (
        "mild",
        "moderate",
        "severe",
    ):
        raise GradedOODError(
            "The tiers must be mild, moderate, and severe "
            "in locked order."
        )

    specifications = []

    for tier in OODTier:
        configuration = tiers.get(tier.value)

        if not isinstance(configuration, dict):
            raise GradedOODError(
                f"Missing configuration for {tier.value}."
            )

        specifications.append(
            OODTierSpecification(
                tier=tier,
                shift_strength=configuration.get(
                    "shift_strength"
                ),
                seed=configuration.get("seed"),
            )
        )

    return tuple(specifications)


def create_paired_graded_ood_views(
    identity_event: SyntheticEvent,
    severe_event: SyntheticEvent,
    amendment: dict[str, Any],
    *,
    latent_event_id: str | None = None,
) -> tuple[GradedOODEvent, ...]:
    """Create paired mild, moderate, and severe event views."""

    if not isinstance(identity_event, SyntheticEvent):
        raise GradedOODError(
            "identity_event must be a SyntheticEvent."
        )

    if not isinstance(severe_event, SyntheticEvent):
        raise GradedOODError(
            "severe_event must be a SyntheticEvent."
        )

    _validate_pair(identity_event, severe_event)

    specifications = tier_specifications_from_amendment(
        amendment
    )

    resolved_id = (
        identity_event.event_id
        if latent_event_id is None
        else latent_event_id
    )

    if (
        not isinstance(resolved_id, str)
        or not resolved_id.strip()
    ):
        raise GradedOODError(
            "latent_event_id must be a nonempty string."
        )

    views = []

    for specification in specifications:
        strength = specification.shift_strength

        views.append(
            GradedOODEvent(
                latent_event_id=resolved_id,
                observed_event_id=(
                    f"{resolved_id}::{specification.tier.value}"
                ),
                tier=specification.tier,
                shift_strength=strength,
                tier_seed=specification.seed,
                template_id=identity_event.template_id,
                target_item_id=identity_event.target_item_id,
                target_family_id=identity_event.target_family_id,
                text_vector=_blend_and_normalize(
                    identity_event.text_vector,
                    severe_event.text_vector,
                    shift_strength=strength,
                    modality="text",
                ),
                image_vector=_blend_and_normalize(
                    identity_event.image_vector,
                    severe_event.image_vector,
                    shift_strength=strength,
                    modality="image",
                ),
                audio_vector=_blend_and_normalize(
                    identity_event.audio_vector,
                    severe_event.audio_vector,
                    shift_strength=strength,
                    modality="audio",
                ),
            )
        )

    return tuple(views)


def _validate_pair(
    identity_event: SyntheticEvent,
    severe_event: SyntheticEvent,
) -> None:
    """Require both observations to describe one latent event."""

    if identity_event.target_item_id != severe_event.target_item_id:
        raise GradedOODError(
            "Paired events must preserve target_item_id."
        )

    if identity_event.target_family_id != severe_event.target_family_id:
        raise GradedOODError(
            "Paired events must preserve target_family_id."
        )

    if identity_event.template_id != severe_event.template_id:
        raise GradedOODError(
            "Paired events must preserve template_id."
        )

    identity_vectors = (
        identity_event.text_vector,
        identity_event.image_vector,
        identity_event.audio_vector,
    )
    severe_vectors = (
        severe_event.text_vector,
        severe_event.image_vector,
        severe_event.audio_vector,
    )

    for identity_vector, severe_vector in zip(
        identity_vectors,
        severe_vectors,
        strict=True,
    ):
        identity_array = np.asarray(
            identity_vector,
            dtype=np.float64,
        )
        severe_array = np.asarray(
            severe_vector,
            dtype=np.float64,
        )

        if (
            identity_array.ndim != 1
            or severe_array.ndim != 1
        ):
            raise GradedOODError(
                "Paired modality vectors must be one-dimensional."
            )

        if identity_array.shape != severe_array.shape:
            raise GradedOODError(
                "Paired modality dimensions must match."
            )


def _blend_and_normalize(
    identity_vector: Vector,
    severe_vector: Vector,
    *,
    shift_strength: float,
    modality: str,
) -> Vector:
    """Interpolate identity and severe vectors and L2-normalize."""

    identity = np.asarray(
        identity_vector,
        dtype=np.float64,
    )
    severe = np.asarray(
        severe_vector,
        dtype=np.float64,
    )

    if identity.ndim != 1 or severe.ndim != 1:
        raise GradedOODError(
            f"{modality} vectors must be one-dimensional."
        )

    if identity.shape != severe.shape:
        raise GradedOODError(
            f"{modality} vector dimensions must match."
        )

    if not np.all(np.isfinite(identity)):
        raise GradedOODError(
            f"Identity {modality} vector must contain finite values."
        )

    if not np.all(np.isfinite(severe)):
        raise GradedOODError(
            f"Severe {modality} vector must contain finite values."
        )

    blended = (
        (1.0 - float(shift_strength)) * identity
        + float(shift_strength) * severe
    )

    norm = float(np.linalg.norm(blended))

    if (
        not np.isfinite(norm)
        or norm <= np.finfo(np.float64).eps
    ):
        raise GradedOODError(
            f"The blended {modality} vector has zero or invalid norm."
        )

    return tuple(
        float(value)
        for value in blended / norm
    )
