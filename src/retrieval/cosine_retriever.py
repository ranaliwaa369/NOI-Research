"""Transparent cosine-similarity baseline for odor-library retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from src.models import OdorLibraryItem, RetrievalCandidate


FloatArray = NDArray[np.float64]


class RetrievalError(ValueError):
    """Raised when cosine retrieval cannot be performed fairly."""


class CosineOdorRetriever:
    """Rank a fixed odor library using normalized cosine similarity."""

    def __init__(
        self,
        library: Sequence[OdorLibraryItem],
    ) -> None:
        if not library:
            raise RetrievalError(
                "The odor library must contain at least one item."
            )

        item_ids = [item.item_id for item in library]

        if len(item_ids) != len(set(item_ids)):
            raise RetrievalError(
                "Every odor-library item_id must be unique."
            )

        vectors = [
            np.asarray(item.odor_vector, dtype=np.float64)
            for item in library
        ]

        if any(vector.ndim != 1 for vector in vectors):
            raise RetrievalError(
                "Every odor-library vector must be one-dimensional."
            )

        dimensions = {vector.shape[0] for vector in vectors}

        if len(dimensions) != 1:
            raise RetrievalError(
                "All odor-library vectors must have identical dimensions."
            )

        if any(not np.all(np.isfinite(vector)) for vector in vectors):
            raise RetrievalError(
                "Odor-library vectors must contain only finite values."
            )

        norms = np.asarray(
            [np.linalg.norm(vector) for vector in vectors],
            dtype=np.float64,
        )

        if np.any(norms <= np.finfo(np.float64).eps):
            raise RetrievalError(
                "Odor-library vectors must have nonzero norms."
            )

        matrix = np.stack(vectors, axis=0)

        self._library = tuple(library)
        self._dimension = matrix.shape[1]
        self._normalized_matrix = matrix / norms[:, np.newaxis]

    @property
    def dimension(self) -> int:
        """Return the required query-vector dimension."""

        return self._dimension

    @property
    def library_size(self) -> int:
        """Return the number of indexed odor-library items."""

        return len(self._library)

    def retrieve(
        self,
        query_vector: Iterable[float] | FloatArray,
        *,
        top_k: int = 10,
    ) -> tuple[RetrievalCandidate, ...]:
        """Return deterministically ranked cosine-similarity candidates."""

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise RetrievalError("top_k must be an integer.")

        if top_k < 1:
            raise RetrievalError("top_k must be at least 1.")

        query = np.asarray(
            tuple(query_vector),
            dtype=np.float64,
        )

        if query.ndim != 1:
            raise RetrievalError(
                "The query must be a one-dimensional vector."
            )

        if query.shape[0] != self._dimension:
            raise RetrievalError(
                "The query dimension does not match the odor library."
            )

        if not np.all(np.isfinite(query)):
            raise RetrievalError(
                "The query must contain only finite values."
            )

        query_norm = float(np.linalg.norm(query))

        if (
            not np.isfinite(query_norm)
            or query_norm <= np.finfo(np.float64).eps
        ):
            raise RetrievalError(
                "The query vector must have a nonzero finite norm."
            )

        normalized_query = query / query_norm
        scores = self._normalized_matrix @ normalized_query

        indexed_scores = [
            (index, float(score))
            for index, score in enumerate(scores)
        ]

        indexed_scores.sort(
            key=lambda pair: (
                -pair[1],
                self._library[pair[0]].item_id,
            )
        )

        selected = indexed_scores[
            : min(top_k, len(indexed_scores))
        ]

        return tuple(
            RetrievalCandidate(
                item_id=self._library[index].item_id,
                score=score,
                rank=rank,
            )
            for rank, (index, score) in enumerate(
                selected,
                start=1,
            )
        )