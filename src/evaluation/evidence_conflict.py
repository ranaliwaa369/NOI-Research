"""Evidence-derived modality reliability and conflict detection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from src.evaluation.multisensory_records import (
    LatentMultisensoryEvent,
    MultisensorySplit,
    SupportRegime,
)


OLFACTORY_DIMENSION = 16
TACTILE_DIMENSION = 8


class EvidenceConflictError(ValueError):
    """Raised when evidence-conflict controls are invalid."""


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Auditable evidence-only reliability and conflict result."""

    family_ids: tuple[int, ...]
    odor_available: bool
    touch_available: bool
    odor_family_distribution: tuple[float, ...]
    touch_family_distribution: tuple[float, ...]
    odor_reliability: float
    touch_reliability: float
    conflict_available: bool
    conflict_score: float


def _normalize_vector(
    vector: Iterable[float],
    *,
    expected_dimension: int,
    label: str,
) -> np.ndarray:
    """Return one finite L2-normalized evidence vector."""

    if isinstance(vector, (str, bytes)):
        raise EvidenceConflictError(
            f"{label} must be a numeric vector."
        )

    try:
        values = tuple(vector)
    except TypeError as exc:
        raise EvidenceConflictError(
            f"{label} must be an iterable numeric vector."
        ) from exc

    if len(values) != expected_dimension:
        raise EvidenceConflictError(
            f"{label} must have dimension {expected_dimension}."
        )

    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise EvidenceConflictError(
            f"{label} must contain finite numeric values."
        )

    array = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(array))

    if norm <= np.finfo(np.float64).eps:
        raise EvidenceConflictError(
            f"{label} must have a nonzero norm."
        )

    return array / norm


def _distribution(
    prototypes: np.ndarray,
    vector: np.ndarray,
) -> tuple[float, ...]:
    """Convert cosine similarities into a stable distribution."""

    similarities = prototypes @ vector
    shifted = similarities - float(np.max(similarities))
    exponentials = np.exp(shifted)
    total = float(np.sum(exponentials))

    if not math.isfinite(total) or total <= 0.0:
        raise EvidenceConflictError(
            "Evidence distribution normalization failed."
        )

    probabilities = exponentials / total

    return tuple(float(value) for value in probabilities)


def _reliability(
    distribution: tuple[float, ...],
) -> float:
    """Return the normalized top-two probability margin."""

    ordered = sorted(distribution, reverse=True)

    if len(ordered) < 2:
        raise EvidenceConflictError(
            "At least two family probabilities are required."
        )

    margin = float(ordered[0] - ordered[1])

    return min(1.0, max(0.0, margin))


def _conflict_score(
    odor_distribution: tuple[float, ...],
    touch_distribution: tuple[float, ...],
) -> float:
    """Return total-variation distance between modality distributions."""

    score = 0.5 * sum(
        abs(odor - touch)
        for odor, touch in zip(
            odor_distribution,
            touch_distribution,
            strict=True,
        )
    )

    return min(1.0, max(0.0, float(score)))


class EvidenceConflictDetector:
    """Training-only family-evidence model for two modalities."""

    def __init__(self) -> None:
        self._family_ids: tuple[int, ...] = ()
        self._odor_prototypes: np.ndarray | None = None
        self._touch_prototypes: np.ndarray | None = None
        self._training_event_count = 0

    @property
    def is_fitted(self) -> bool:
        """Return whether training-only prototypes exist."""

        return (
            self._odor_prototypes is not None
            and self._touch_prototypes is not None
        )

    @property
    def training_event_count(self) -> int:
        """Return the number of training events used."""

        return self._training_event_count

    @property
    def training_family_count(self) -> int:
        """Return the number of training families."""

        return len(self._family_ids)

    def fit(
        self,
        events: tuple[LatentMultisensoryEvent, ...],
    ) -> "EvidenceConflictDetector":
        """Fit separate olfactory and tactile family prototypes."""

        if (
            not isinstance(events, tuple)
            or not events
            or not all(
                isinstance(event, LatentMultisensoryEvent)
                for event in events
            )
        ):
            raise EvidenceConflictError(
                "fit events must be a nonempty tuple of latent events."
            )

        if any(
            event.split is not MultisensorySplit.TRAIN
            for event in events
        ):
            raise EvidenceConflictError(
                "Evidence fitting permits training data only."
            )

        if any(
            event.support_regime is not SupportRegime.DEVELOPMENT
            for event in events
        ):
            raise EvidenceConflictError(
                "Training events must retain development support."
            )

        grouped_odor: dict[int, list[np.ndarray]] = {}
        grouped_touch: dict[int, list[np.ndarray]] = {}

        for event in events:
            family_id = event.target_family_id
            grouped_odor.setdefault(family_id, []).append(
                _normalize_vector(
                    event.olfactory_vector,
                    expected_dimension=OLFACTORY_DIMENSION,
                    label="training olfactory vector",
                )
            )
            grouped_touch.setdefault(family_id, []).append(
                _normalize_vector(
                    event.tactile_vector,
                    expected_dimension=TACTILE_DIMENSION,
                    label="training tactile vector",
                )
            )

        family_ids = tuple(sorted(grouped_odor))

        if len(family_ids) < 2:
            raise EvidenceConflictError(
                "At least two training families are required."
            )

        if family_ids != tuple(sorted(grouped_touch)):
            raise EvidenceConflictError(
                "Modality family registries must match."
            )

        odor_prototypes: list[np.ndarray] = []
        touch_prototypes: list[np.ndarray] = []

        for family_id in family_ids:
            odor_centroid = np.stack(
                grouped_odor[family_id],
                axis=0,
            ).mean(axis=0)
            touch_centroid = np.stack(
                grouped_touch[family_id],
                axis=0,
            ).mean(axis=0)

            odor_norm = float(np.linalg.norm(odor_centroid))
            touch_norm = float(np.linalg.norm(touch_centroid))

            if (
                odor_norm <= np.finfo(np.float64).eps
                or touch_norm <= np.finfo(np.float64).eps
            ):
                raise EvidenceConflictError(
                    "A family evidence prototype has zero norm."
                )

            odor_prototypes.append(
                odor_centroid / odor_norm
            )
            touch_prototypes.append(
                touch_centroid / touch_norm
            )

        self._family_ids = family_ids
        self._odor_prototypes = np.stack(
            odor_prototypes,
            axis=0,
        )
        self._touch_prototypes = np.stack(
            touch_prototypes,
            axis=0,
        )
        self._training_event_count = len(events)

        return self

    def assess(
        self,
        *,
        olfactory_vector: Iterable[float] | None,
        tactile_vector: Iterable[float] | None,
    ) -> EvidenceAssessment:
        """Assess reliability and conflict from modality evidence only."""

        if not self.is_fitted:
            raise EvidenceConflictError(
                "EvidenceConflictDetector must be fitted before assessment."
            )

        if olfactory_vector is None and tactile_vector is None:
            raise EvidenceConflictError(
                "At least one modality must be available."
            )

        assert self._odor_prototypes is not None
        assert self._touch_prototypes is not None

        odor_available = olfactory_vector is not None
        touch_available = tactile_vector is not None

        if odor_available:
            assert olfactory_vector is not None
            normalized_odor = _normalize_vector(
                olfactory_vector,
                expected_dimension=OLFACTORY_DIMENSION,
                label="olfactory_vector",
            )
            odor_distribution = _distribution(
                self._odor_prototypes,
                normalized_odor,
            )
            odor_reliability = _reliability(
                odor_distribution
            )
        else:
            odor_distribution = ()
            odor_reliability = 0.0

        if touch_available:
            assert tactile_vector is not None
            normalized_touch = _normalize_vector(
                tactile_vector,
                expected_dimension=TACTILE_DIMENSION,
                label="tactile_vector",
            )
            touch_distribution = _distribution(
                self._touch_prototypes,
                normalized_touch,
            )
            touch_reliability = _reliability(
                touch_distribution
            )
        else:
            touch_distribution = ()
            touch_reliability = 0.0

        conflict_available = (
            odor_available and touch_available
        )

        conflict_score = (
            _conflict_score(
                odor_distribution,
                touch_distribution,
            )
            if conflict_available
            else 0.0
        )

        return EvidenceAssessment(
            family_ids=self._family_ids,
            odor_available=odor_available,
            touch_available=touch_available,
            odor_family_distribution=odor_distribution,
            touch_family_distribution=touch_distribution,
            odor_reliability=odor_reliability,
            touch_reliability=touch_reliability,
            conflict_available=conflict_available,
            conflict_score=conflict_score,
        )
