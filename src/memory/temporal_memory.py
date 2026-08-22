"""Temporally structured associative memory for the NOI architecture."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from math import exp, isfinite

import numpy as np
from numpy.typing import NDArray

from src.memory.records import (
    AssociativeMemoryRecord,
    MemoryRetrievalCandidate,
)


FloatArray = NDArray[np.float64]
SECONDS_PER_DAY = 86_400.0


class AssociativeMemoryError(ValueError):
    """Raised when associative-memory operations are invalid."""


class TemporalAssociativeMemory:
    """Store and retrieve computational context-to-odor associations."""

    def __init__(
        self,
        *,
        dimension: int,
        decay_rate_per_day: float,
    ) -> None:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise AssociativeMemoryError(
                "dimension must be a positive integer."
            )

        if (
            not isfinite(decay_rate_per_day)
            or decay_rate_per_day < 0.0
        ):
            raise AssociativeMemoryError(
                "decay_rate_per_day must be finite and nonnegative."
            )

        self._dimension = dimension
        self._decay_rate_per_day = float(decay_rate_per_day)
        self._records: dict[str, AssociativeMemoryRecord] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def decay_rate_per_day(self) -> float:
        return self._decay_rate_per_day

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def active_size(self) -> int:
        return sum(record.active for record in self._records.values())

    def add(self, record: AssociativeMemoryRecord) -> None:
        """Add a validated association without overwriting existing data."""

        if record.memory_id in self._records:
            raise AssociativeMemoryError(
                f"Duplicate memory_id: {record.memory_id}"
            )

        if len(record.context_vector) != self._dimension:
            raise AssociativeMemoryError(
                "The record dimension does not match the memory dimension."
            )

        self._records[record.memory_id] = record

    def get(self, memory_id: str) -> AssociativeMemoryRecord:
        """Return one stored association by identifier."""

        try:
            return self._records[memory_id]
        except KeyError as error:
            raise AssociativeMemoryError(
                f"Unknown memory_id: {memory_id}"
            ) from error

    def retrieve(
        self,
        query_vector: Iterable[float] | FloatArray,
        *,
        as_of_utc: datetime,
        top_k: int = 10,
        apply_temporal_decay: bool = True,
    ) -> tuple[MemoryRetrievalCandidate, ...]:
        """Rank active memories using context, strength, and temporal weight."""

        self._validate_as_of(as_of_utc)

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise AssociativeMemoryError(
                "top_k must be an integer."
            )

        if top_k < 1:
            raise AssociativeMemoryError(
                "top_k must be at least 1."
            )

        if not isinstance(apply_temporal_decay, bool):
            raise AssociativeMemoryError(
                "apply_temporal_decay must be boolean."
            )

        query = np.asarray(
            tuple(query_vector),
            dtype=np.float64,
        )

        if query.ndim != 1:
            raise AssociativeMemoryError(
                "The query must be one-dimensional."
            )

        if query.shape[0] != self._dimension:
            raise AssociativeMemoryError(
                "The query dimension does not match the memory dimension."
            )

        if not np.all(np.isfinite(query)):
            raise AssociativeMemoryError(
                "The query must contain only finite values."
            )

        query_norm = float(np.linalg.norm(query))

        if (
            not np.isfinite(query_norm)
            or query_norm <= np.finfo(np.float64).eps
        ):
            raise AssociativeMemoryError(
                "The query must have a nonzero finite norm."
            )

        normalized_query = query / query_norm
        candidates: list[MemoryRetrievalCandidate] = []

        for record in self._records.values():
            if not record.active:
                continue

            if as_of_utc < record.updated_at_utc:
                raise AssociativeMemoryError(
                    "as_of_utc cannot precede a record's updated_at_utc."
                )

            context = np.asarray(
                record.context_vector,
                dtype=np.float64,
            )
            context_norm = float(np.linalg.norm(context))

            if context_norm <= np.finfo(np.float64).eps:
                raise AssociativeMemoryError(
                    f"Memory {record.memory_id} has a zero context norm."
                )

            raw_similarity = float(
                np.dot(normalized_query, context / context_norm)
            )

            contextual_similarity = min(
                1.0,
                max(0.0, raw_similarity),
            )

            age_days = (
                as_of_utc - record.updated_at_utc
            ).total_seconds() / SECONDS_PER_DAY

            if apply_temporal_decay:
                temporal_weight = exp(
                    -self._decay_rate_per_day * age_days
                )
                temporal_weight = max(
                    temporal_weight,
                    np.finfo(np.float64).tiny,
                )
            else:
                temporal_weight = 1.0

            score = (
                contextual_similarity
                * record.strength
                * temporal_weight
            )

            candidates.append(
                MemoryRetrievalCandidate(
                    memory_id=record.memory_id,
                    odor_item_id=record.odor_item_id,
                    score=float(score),
                    contextual_similarity=contextual_similarity,
                    temporal_weight=float(temporal_weight),
                    memory_strength=record.strength,
                    rank=1,
                )
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.odor_item_id,
                candidate.memory_id,
            )
        )

        selected = candidates[: min(top_k, len(candidates))]

        return tuple(
            MemoryRetrievalCandidate(
                memory_id=candidate.memory_id,
                odor_item_id=candidate.odor_item_id,
                score=candidate.score,
                contextual_similarity=candidate.contextual_similarity,
                temporal_weight=candidate.temporal_weight,
                memory_strength=candidate.memory_strength,
                rank=rank,
            )
            for rank, candidate in enumerate(selected, start=1)
        )

    @staticmethod
    def _validate_as_of(as_of_utc: datetime) -> None:
        if (
            as_of_utc.tzinfo is None
            or as_of_utc.utcoffset() is None
        ):
            raise AssociativeMemoryError(
                "as_of_utc must be timezone-aware."
            )