"""Validated records for the NOI associative-memory subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite


Vector = tuple[float, ...]


def _require_aware_timestamp(
    name: str,
    value: datetime,
) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")


def _require_finite_vector(
    name: str,
    value: Vector,
) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty.")

    if not all(isinstance(item, (int, float)) for item in value):
        raise TypeError(f"{name} must contain only numeric values.")

    if not all(isfinite(float(item)) for item in value):
        raise ValueError(f"{name} must contain only finite values.")


@dataclass(frozen=True, slots=True)
class AssociativeMemoryRecord:
    """A stored computational context-to-odor association."""

    memory_id: str
    context_vector: Vector
    odor_item_id: str
    created_at_utc: datetime
    updated_at_utc: datetime
    strength: float = 1.0
    correction_count: int = 0
    active: bool = True

    def __post_init__(self) -> None:
        if not self.memory_id.strip():
            raise ValueError("memory_id must not be empty.")

        if not self.odor_item_id.strip():
            raise ValueError("odor_item_id must not be empty.")

        _require_finite_vector(
            "context_vector",
            self.context_vector,
        )

        _require_aware_timestamp(
            "created_at_utc",
            self.created_at_utc,
        )
        _require_aware_timestamp(
            "updated_at_utc",
            self.updated_at_utc,
        )

        if self.updated_at_utc < self.created_at_utc:
            raise ValueError(
                "updated_at_utc cannot precede created_at_utc."
            )

        if not isfinite(self.strength) or not 0.0 < self.strength <= 1.0:
            raise ValueError(
                "strength must be finite and within (0, 1]."
            )

        if (
            isinstance(self.correction_count, bool)
            or not isinstance(self.correction_count, int)
            or self.correction_count < 0
        ):
            raise ValueError(
                "correction_count must be a nonnegative integer."
            )


@dataclass(frozen=True, slots=True)
class MemoryRetrievalCandidate:
    """An auditable candidate retrieved from associative memory."""

    memory_id: str
    odor_item_id: str
    score: float
    contextual_similarity: float
    temporal_weight: float
    memory_strength: float
    rank: int

    def __post_init__(self) -> None:
        if not self.memory_id.strip():
            raise ValueError("memory_id must not be empty.")

        if not self.odor_item_id.strip():
            raise ValueError("odor_item_id must not be empty.")

        numeric_values = {
            "score": self.score,
            "contextual_similarity": self.contextual_similarity,
            "temporal_weight": self.temporal_weight,
            "memory_strength": self.memory_strength,
        }

        if not all(isfinite(value) for value in numeric_values.values()):
            raise ValueError(
                "All candidate scores and weights must be finite."
            )

        if not 0.0 <= self.contextual_similarity <= 1.0:
            raise ValueError(
                "contextual_similarity must be within [0, 1]."
            )

        if not 0.0 < self.temporal_weight <= 1.0:
            raise ValueError(
                "temporal_weight must be within (0, 1]."
            )

        if not 0.0 < self.memory_strength <= 1.0:
            raise ValueError(
                "memory_strength must be within (0, 1]."
            )

        if self.score < 0.0:
            raise ValueError("score must be nonnegative.")

        if self.rank < 1:
            raise ValueError("rank must be at least 1.")