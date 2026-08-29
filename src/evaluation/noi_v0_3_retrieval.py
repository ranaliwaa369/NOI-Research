"""Metadata-blind retrieval mechanics for locked NOI v0.3 execution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from src.evaluation.multisensory_records import (
    LatentMultisensoryEvent,
    MultisensorySplit,
    MultisensoryTarget,
)


FloatArray = NDArray[np.float64]


class NOIV03RetrievalError(ValueError):
    """Raised when locked v0.3 retrieval inputs are invalid."""


@dataclass(frozen=True, slots=True)
class NOIV03RetrievalResult:
    """One metadata-blind ranked or abstaining inference result."""

    event_id: str
    ranking: tuple[str, ...]
    scores: tuple[float, ...]
    abstained: bool
    odor_weight: float
    touch_weight: float
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise NOIV03RetrievalError(
                "event_id must be a nonempty string."
            )

        if len(self.ranking) != len(self.scores):
            raise NOIV03RetrievalError(
                "ranking and scores must have equal length."
            )

        if len(self.ranking) != len(set(self.ranking)):
            raise NOIV03RetrievalError(
                "ranking item identifiers must be unique."
            )

        if any(
            not isinstance(item_id, str) or not item_id.strip()
            for item_id in self.ranking
        ):
            raise NOIV03RetrievalError(
                "ranking item identifiers must be nonempty strings."
            )

        if any(
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
            for score in self.scores
        ):
            raise NOIV03RetrievalError(
                "scores must contain finite numeric values."
            )

        _validate_weight("odor_weight", self.odor_weight)
        _validate_weight("touch_weight", self.touch_weight)

        if not isinstance(self.abstained, bool):
            raise NOIV03RetrievalError(
                "abstained must be a Boolean."
            )

        if self.abstained and (
            self.ranking
            or self.scores
            or self.odor_weight != 0.0
            or self.touch_weight != 0.0
        ):
            raise NOIV03RetrievalError(
                "An abstention cannot contain ranking evidence."
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise NOIV03RetrievalError(
                "reason must be a nonempty string."
            )


def _validate_weight(
    name: str,
    value: float,
) -> float:
    """Return one finite probability weight."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise NOIV03RetrievalError(
            f"{name} must be finite and between 0 and 1."
        )

    return float(value)


def _normalized_vector(
    vector: Iterable[float],
    *,
    expected_dimension: int,
    label: str,
) -> FloatArray:
    """Return one validated L2-normalized vector."""

    try:
        values = tuple(vector)
    except TypeError as error:
        raise NOIV03RetrievalError(
            f"{label} must be an iterable numeric vector."
        ) from error

    if len(values) != expected_dimension:
        raise NOIV03RetrievalError(
            f"{label} must have dimension {expected_dimension}."
        )

    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise NOIV03RetrievalError(
            f"{label} must contain finite numeric values."
        )

    array = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(array))

    if (
        not math.isfinite(norm)
        or norm <= np.finfo(np.float64).eps
    ):
        raise NOIV03RetrievalError(
            f"{label} must have a nonzero finite norm."
        )

    return array / norm


@dataclass(frozen=True, slots=True)
class NOIV03RetrievalLibrary:
    """Training-only odor and touch prototype retrieval memory."""

    item_ids: tuple[str, ...]
    _normalized_odor_matrix: FloatArray
    _normalized_touch_matrix: FloatArray
    training_event_count: int

    @property
    def library_size(self) -> int:
        """Return the number of represented training items."""

        return len(self.item_ids)

    @classmethod
    def from_training_records(
        cls,
        *,
        training_events: Sequence[LatentMultisensoryEvent],
        targets: Sequence[MultisensoryTarget],
    ) -> "NOIV03RetrievalLibrary":
        """Build candidate memory from represented training items only."""

        if not isinstance(training_events, Sequence):
            raise NOIV03RetrievalError(
                "training_events must be a sequence."
            )

        if not training_events:
            raise NOIV03RetrievalError(
                "At least one training event is required."
            )

        if any(
            not isinstance(event, LatentMultisensoryEvent)
            for event in training_events
        ):
            raise NOIV03RetrievalError(
                "training_events must contain latent event records."
            )

        if any(
            event.split is not MultisensorySplit.TRAIN
            for event in training_events
        ):
            raise NOIV03RetrievalError(
                "Retrieval memory may use training records only."
            )

        if not isinstance(targets, Sequence):
            raise NOIV03RetrievalError(
                "targets must be a sequence."
            )

        if any(
            not isinstance(item, MultisensoryTarget)
            for item in targets
        ):
            raise NOIV03RetrievalError(
                "targets must contain multisensory target records."
            )

        target_map: dict[str, MultisensoryTarget] = {}

        for item in targets:
            if item.item_id in target_map:
                raise NOIV03RetrievalError(
                    "Target item identifiers must be unique."
                )
            target_map[item.item_id] = item

        represented_ids = tuple(
            sorted(
                {
                    event.target_item_id
                    for event in training_events
                }
            )
        )

        missing = tuple(
            item_id
            for item_id in represented_ids
            if item_id not in target_map
        )

        if missing:
            raise NOIV03RetrievalError(
                "Training target prototypes are absent: "
                f"{list(missing)}"
            )

        represented_targets = tuple(
            target_map[item_id]
            for item_id in represented_ids
        )

        odor_matrix = np.stack(
            [
                _normalized_vector(
                    item.olfactory_prototype,
                    expected_dimension=16,
                    label=(
                        f"olfactory prototype for {item.item_id}"
                    ),
                )
                for item in represented_targets
            ],
            axis=0,
        )

        touch_matrix = np.stack(
            [
                _normalized_vector(
                    item.tactile_prototype,
                    expected_dimension=8,
                    label=(
                        f"tactile prototype for {item.item_id}"
                    ),
                )
                for item in represented_targets
            ],
            axis=0,
        )

        return cls(
            item_ids=represented_ids,
            _normalized_odor_matrix=odor_matrix,
            _normalized_touch_matrix=touch_matrix,
            training_event_count=len(training_events),
        )

    def abstain(
        self,
        *,
        event_id: str,
        reason: str,
    ) -> NOIV03RetrievalResult:
        """Return an explicit identity abstention."""

        return NOIV03RetrievalResult(
            event_id=event_id,
            ranking=(),
            scores=(),
            abstained=True,
            odor_weight=0.0,
            touch_weight=0.0,
            reason=reason,
        )

    def rank(
        self,
        *,
        event_id: str,
        olfactory_vector: Iterable[float] | None,
        tactile_vector: Iterable[float] | None,
        odor_weight: float,
        touch_weight: float,
        top_k: int = 10,
    ) -> NOIV03RetrievalResult:
        """Rank training items using locked weighted modality scores."""

        if not isinstance(event_id, str) or not event_id.strip():
            raise NOIV03RetrievalError(
                "event_id must be a nonempty string."
            )

        odor_weight_value = _validate_weight(
            "odor_weight",
            odor_weight,
        )
        touch_weight_value = _validate_weight(
            "touch_weight",
            touch_weight,
        )

        odor_available = olfactory_vector is not None
        touch_available = tactile_vector is not None

        if not odor_available and odor_weight_value != 0.0:
            raise NOIV03RetrievalError(
                "Unavailable odor must receive zero weight."
            )

        if not touch_available and touch_weight_value != 0.0:
            raise NOIV03RetrievalError(
                "Unavailable touch must receive zero weight."
            )

        if not odor_available and not touch_available:
            raise NOIV03RetrievalError(
                "At least one modality must be available."
            )

        if not math.isclose(
            odor_weight_value + touch_weight_value,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise NOIV03RetrievalError(
                "Available modality weights must sum to 1."
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise NOIV03RetrievalError(
                "top_k must be a positive integer."
            )

        if top_k > self.library_size:
            raise NOIV03RetrievalError(
                "top_k cannot exceed the training library size."
            )

        combined_scores = np.zeros(
            self.library_size,
            dtype=np.float64,
        )

        if odor_weight_value > 0.0:
            assert olfactory_vector is not None
            odor_query = _normalized_vector(
                olfactory_vector,
                expected_dimension=16,
                label="olfactory_vector",
            )
            combined_scores += (
                odor_weight_value
                * (self._normalized_odor_matrix @ odor_query)
            )

        if touch_weight_value > 0.0:
            assert tactile_vector is not None
            touch_query = _normalized_vector(
                tactile_vector,
                expected_dimension=8,
                label="tactile_vector",
            )
            combined_scores += (
                touch_weight_value
                * (self._normalized_touch_matrix @ touch_query)
            )

        indexed_scores = tuple(
            (
                index,
                float(score),
            )
            for index, score in enumerate(combined_scores)
        )

        ordered = tuple(
            sorted(
                indexed_scores,
                key=lambda pair: (
                    -pair[1],
                    self.item_ids[pair[0]],
                ),
            )
        )[:top_k]

        return NOIV03RetrievalResult(
            event_id=event_id,
            ranking=tuple(
                self.item_ids[index]
                for index, _ in ordered
            ),
            scores=tuple(
                score
                for _, score in ordered
            ),
            abstained=False,
            odor_weight=odor_weight_value,
            touch_weight=touch_weight_value,
            reason=(
                "Ranked the training-only candidate library "
                "using locked weighted modality cosine scores."
            ),
        )
