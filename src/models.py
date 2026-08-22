"""Core data models for the Neuro-Olfactive Interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Any


Vector = tuple[float, ...]


def _validate_vector(name: str, vector: Vector | None) -> None:
    """Validate that an optional vector is finite and nonempty."""

    if vector is None:
        return

    if not vector:
        raise ValueError(f"{name} must not be empty.")

    if not all(isinstance(value, (int, float)) for value in vector):
        raise TypeError(f"{name} must contain only numeric values.")

    if not all(isfinite(float(value)) for value in vector):
        raise ValueError(f"{name} must contain only finite values.")


@dataclass(frozen=True, slots=True)
class MultimodalContext:
    """A contextual event supplied to the NOI retrieval system."""

    event_id: str
    timestamp_utc: datetime
    text_vector: Vector | None = None
    image_vector: Vector | None = None
    audio_vector: Vector | None = None
    temporal_vector: Vector | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty.")

        if self.timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware.")

        vectors = (
            self.text_vector,
            self.image_vector,
            self.audio_vector,
            self.temporal_vector,
        )

        if all(vector is None for vector in vectors):
            raise ValueError(
                "At least one contextual modality must be provided."
            )

        _validate_vector("text_vector", self.text_vector)
        _validate_vector("image_vector", self.image_vector)
        _validate_vector("audio_vector", self.audio_vector)
        _validate_vector("temporal_vector", self.temporal_vector)


@dataclass(frozen=True, slots=True)
class OdorLibraryItem:
    """A fixed target in the computational odor library."""

    item_id: str
    odor_vector: Vector
    descriptors: tuple[str, ...]
    cartridge_id: str | None = None
    source_reference: str | None = None
    odor_family: str | None = None

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id must not be empty.")

        _validate_vector("odor_vector", self.odor_vector)

        if not self.descriptors:
            raise ValueError(
                "At least one documented odor descriptor is required."
            )


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """A ranked odor-library candidate returned by retrieval."""

    item_id: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id must not be empty.")

        if not isfinite(self.score):
            raise ValueError("score must be finite.")

        if self.rank < 1:
            raise ValueError("rank must be at least 1.")


@dataclass(frozen=True, slots=True)
class OutputRequest:
    """A simulated request evaluated by the policy gate."""

    request_id: str
    item_id: str
    concentration_ppm: float | None
    duration_seconds: float | None
    environment_volume_m3: float | None
    ventilation_ach: float | None
    user_consent: bool | None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty.")

        if not self.item_id.strip():
            raise ValueError("item_id must not be empty.")

        numeric_fields = {
            "concentration_ppm": self.concentration_ppm,
            "duration_seconds": self.duration_seconds,
            "environment_volume_m3": self.environment_volume_m3,
            "ventilation_ach": self.ventilation_ach,
        }

        for name, value in numeric_fields.items():
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(
                    f"{name} must be finite and nonnegative."
                )


class PolicyOutcome(str, Enum):
    """Possible deterministic policy-gate outcomes."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REQUIRE_MISSING_INFORMATION = "REQUIRE_MISSING_INFORMATION"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """An auditable policy decision for a simulated output request."""

    request_id: str
    outcome: PolicyOutcome
    rule_ids: tuple[str, ...]
    explanation: str
    protocol_hash: str

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty.")

        if not self.rule_ids:
            raise ValueError(
                "Every policy decision must identify at least one rule."
            )

        if not self.explanation.strip():
            raise ValueError("A decision explanation is required.")

        if len(self.protocol_hash) != 64:
            raise ValueError(
                "protocol_hash must be a 64-character SHA-256 value."
            )