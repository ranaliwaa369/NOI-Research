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

@dataclass(frozen=True, slots=True)
class ReliabilityCalibrationObservation:
    """One validation-only reliability calibration target."""

    source_split: MultisensorySplit
    reliability: float
    prediction_correct: bool

    def __post_init__(self) -> None:
        if not isinstance(self.source_split, MultisensorySplit):
            raise EvidenceConflictError(
                "source_split must be a MultisensorySplit value."
            )
        _validate_probability(
            "reliability",
            self.reliability,
        )
        if not isinstance(self.prediction_correct, bool):
            raise EvidenceConflictError(
                "prediction_correct must be a boolean."
            )


@dataclass(frozen=True, slots=True)
class ConflictCalibrationObservation:
    """One validation-only modality-conflict calibration target."""

    source_split: MultisensorySplit
    conflict_score: float
    conflict_present: bool

    def __post_init__(self) -> None:
        if not isinstance(self.source_split, MultisensorySplit):
            raise EvidenceConflictError(
                "source_split must be a MultisensorySplit value."
            )
        _validate_probability(
            "conflict_score",
            self.conflict_score,
        )
        if not isinstance(self.conflict_present, bool):
            raise EvidenceConflictError(
                "conflict_present must be a boolean."
            )


@dataclass(frozen=True, slots=True)
class ReliabilityCalibrationReport:
    """Immutable validation-derived reliability threshold."""

    source_split: MultisensorySplit
    validation_observation_count: int
    correct_count: int
    incorrect_count: int
    threshold: float
    balanced_accuracy: float
    final_test_labels_used: bool


@dataclass(frozen=True, slots=True)
class ConflictCalibrationReport:
    """Immutable constrained validation conflict threshold."""

    source_split: MultisensorySplit
    validation_observation_count: int
    conflict_count: int
    nonconflict_count: int
    threshold: float
    maximum_false_conflict_rate: float
    validation_false_conflict_rate: float
    conflict_true_positive_rate: float
    balanced_accuracy: float
    final_test_labels_used: bool


def _validate_probability(
    name: str,
    value: object,
) -> float:
    """Require one finite probability in the closed unit interval."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise EvidenceConflictError(
            f"{name} must be finite and between 0 and 1."
        )

    return float(value)


def _threshold_candidates(
    values: tuple[float, ...],
) -> tuple[float, ...]:
    """Return deterministic boundaries and adjacent midpoints."""

    unique = tuple(sorted(set(values)))

    if not unique:
        raise EvidenceConflictError(
            "Threshold calibration requires observations."
        )

    candidates: list[float] = [
        float(np.nextafter(unique[0], -np.inf))
    ]
    candidates.extend(
        (left + right) / 2.0
        for left, right in zip(
            unique,
            unique[1:],
        )
    )
    candidates.append(
        float(np.nextafter(unique[-1], np.inf))
    )

    return tuple(candidates)


def calibrate_reliability_threshold(
    observations: tuple[
        ReliabilityCalibrationObservation,
        ...
    ],
) -> ReliabilityCalibrationReport:
    """Select maximum-balanced-accuracy reliability threshold."""

    if (
        not isinstance(observations, tuple)
        or not observations
        or not all(
            isinstance(
                observation,
                ReliabilityCalibrationObservation,
            )
            for observation in observations
        )
    ):
        raise EvidenceConflictError(
            "Reliability calibration requires a nonempty tuple "
            "of observations."
        )

    if any(
        observation.source_split
        is not MultisensorySplit.VALIDATION
        for observation in observations
    ):
        raise EvidenceConflictError(
            "Reliability calibration permits validation "
            "observations only."
        )

    correct_count = sum(
        observation.prediction_correct
        for observation in observations
    )
    incorrect_count = len(observations) - correct_count

    if correct_count == 0 or incorrect_count == 0:
        raise EvidenceConflictError(
            "Reliability calibration requires correct and "
            "incorrect validation predictions."
        )

    scores = tuple(
        float(observation.reliability)
        for observation in observations
    )
    candidates = _threshold_candidates(scores)

    eligible: list[tuple[float, float]] = []

    for candidate in candidates:
        predicted_reliable = tuple(
            score >= candidate
            for score in scores
        )
        true_positive_rate = (
            sum(
                prediction
                and observation.prediction_correct
                for prediction, observation in zip(
                    predicted_reliable,
                    observations,
                    strict=True,
                )
            )
            / correct_count
        )
        true_negative_rate = (
            sum(
                (not prediction)
                and (not observation.prediction_correct)
                for prediction, observation in zip(
                    predicted_reliable,
                    observations,
                    strict=True,
                )
            )
            / incorrect_count
        )
        balanced_accuracy = (
            true_positive_rate + true_negative_rate
        ) / 2.0
        eligible.append(
            (
                balanced_accuracy,
                candidate,
            )
        )

    balanced_accuracy, threshold = max(
        eligible,
        key=lambda item: (
            item[0],
            item[1],
        ),
    )

    return ReliabilityCalibrationReport(
        source_split=MultisensorySplit.VALIDATION,
        validation_observation_count=len(observations),
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        threshold=float(threshold),
        balanced_accuracy=float(balanced_accuracy),
        final_test_labels_used=False,
    )


def calibrate_conflict_threshold(
    observations: tuple[
        ConflictCalibrationObservation,
        ...
    ],
    *,
    maximum_false_conflict_rate: float,
) -> ConflictCalibrationReport:
    """Select constrained validation-only modality-conflict threshold."""

    maximum_rate = _validate_probability(
        "maximum false-conflict rate",
        maximum_false_conflict_rate,
    )

    if (
        not isinstance(observations, tuple)
        or not observations
        or not all(
            isinstance(
                observation,
                ConflictCalibrationObservation,
            )
            for observation in observations
        )
    ):
        raise EvidenceConflictError(
            "Conflict calibration requires a nonempty tuple "
            "of observations."
        )

    if any(
        observation.source_split
        is not MultisensorySplit.VALIDATION
        for observation in observations
    ):
        raise EvidenceConflictError(
            "Conflict calibration permits validation "
            "observations only."
        )

    conflict_count = sum(
        observation.conflict_present
        for observation in observations
    )
    nonconflict_count = len(observations) - conflict_count

    if conflict_count == 0 or nonconflict_count == 0:
        raise EvidenceConflictError(
            "Conflict calibration requires conflict and "
            "nonconflict validation observations."
        )

    scores = tuple(
        float(observation.conflict_score)
        for observation in observations
    )
    candidates = _threshold_candidates(scores)
    eligible: list[
        tuple[float, float, float, float]
    ] = []

    for candidate in candidates:
        predicted_conflict = tuple(
            score >= candidate
            for score in scores
        )
        true_positive_rate = (
            sum(
                prediction
                and observation.conflict_present
                for prediction, observation in zip(
                    predicted_conflict,
                    observations,
                    strict=True,
                )
            )
            / conflict_count
        )
        false_conflict_rate = (
            sum(
                prediction
                and (not observation.conflict_present)
                for prediction, observation in zip(
                    predicted_conflict,
                    observations,
                    strict=True,
                )
            )
            / nonconflict_count
        )
        true_negative_rate = 1.0 - false_conflict_rate
        balanced_accuracy = (
            true_positive_rate + true_negative_rate
        ) / 2.0

        if false_conflict_rate <= maximum_rate:
            eligible.append(
                (
                    true_positive_rate,
                    balanced_accuracy,
                    -candidate,
                    false_conflict_rate,
                )
            )

    if not eligible:
        raise EvidenceConflictError(
            "No threshold satisfies the registered "
            "false-conflict constraint."
        )

    (
        true_positive_rate,
        balanced_accuracy,
        negative_threshold,
        false_conflict_rate,
    ) = max(
        eligible,
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        ),
    )
    threshold = -negative_threshold

    return ConflictCalibrationReport(
        source_split=MultisensorySplit.VALIDATION,
        validation_observation_count=len(observations),
        conflict_count=conflict_count,
        nonconflict_count=nonconflict_count,
        threshold=float(threshold),
        maximum_false_conflict_rate=maximum_rate,
        validation_false_conflict_rate=float(
            false_conflict_rate
        ),
        conflict_true_positive_rate=float(
            true_positive_rate
        ),
        balanced_accuracy=float(balanced_accuracy),
        final_test_labels_used=False,
    )
