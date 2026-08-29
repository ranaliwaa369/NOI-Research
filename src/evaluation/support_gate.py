"""Validation-calibrated support gate for NOI v0.3.

The gate implements the three preregistered support comparisons. Model fitting
accepts training events only, threshold selection accepts validation events
only, and final-test labels are never accepted by either operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from src.evaluation.multisensory_records import (
    LatentMultisensoryEvent,
    MultisensorySplit,
    SupportRegime,
)


FloatArray = NDArray[np.float64]


class SupportGateError(ValueError):
    """Raised when support-gate integrity or numerical checks fail."""


class SupportMethod(str, Enum):
    """The three preregistered support comparisons."""

    MAHALANOBIS = "mahalanobis"
    COSINE_MARGIN = "cosine_margin"
    NEAREST_PROTOTYPE_DISTANCE = "nearest_prototype_distance"


class UncertaintyStatus(str, Enum):
    """Decision position relative to the validation-locked band."""

    CERTAIN_SUPPORTED = "certain_supported"
    UNCERTAIN = "uncertain"
    CERTAIN_UNSUPPORTED = "certain_unsupported"


@dataclass(frozen=True, slots=True)
class SupportCalibrationReport:
    """Immutable record of validation-only threshold selection."""

    method: SupportMethod
    source_split: MultisensorySplit
    validation_event_count: int
    supported_count: int
    unsupported_count: int
    threshold: float
    uncertainty_width: float
    balanced_accuracy: float
    final_test_labels_used: bool


@dataclass(frozen=True, slots=True)
class SupportDecision:
    """One support score and its uncertainty-aware operational decision."""

    event_id: str
    method: SupportMethod
    support_score: float
    threshold: float
    is_supported: bool
    uncertainty_status: UncertaintyStatus
    request_touch: bool


def _normalize_vector(
    vector: Iterable[float],
    *,
    expected_dimension: int | None,
    label: str,
) -> FloatArray:
    """Return one finite nonzero L2-normalized vector."""

    try:
        array = np.asarray(
            tuple(vector),
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise SupportGateError(
            f"{label} must contain numeric values."
        ) from error

    if array.ndim != 1:
        raise SupportGateError(
            f"{label} must be one-dimensional."
        )

    if expected_dimension is not None and array.shape[0] != expected_dimension:
        raise SupportGateError(
            f"{label} dimension does not match the fitted dimension."
        )

    if array.shape[0] < 1:
        raise SupportGateError(
            f"{label} cannot be empty."
        )

    if not np.all(np.isfinite(array)):
        raise SupportGateError(
            f"{label} must contain only finite values."
        )

    norm = float(np.linalg.norm(array))

    if (
        not math.isfinite(norm)
        or norm <= np.finfo(np.float64).eps
    ):
        raise SupportGateError(
            f"{label} must have a nonzero finite norm."
        )

    return array / norm


def _validate_event_tuple(
    events: object,
    *,
    label: str,
) -> tuple[LatentMultisensoryEvent, ...]:
    """Require a nonempty tuple of latent events."""

    if (
        not isinstance(events, tuple)
        or not events
        or not all(
            isinstance(event, LatentMultisensoryEvent)
            for event in events
        )
    ):
        raise SupportGateError(
            f"{label} must be a nonempty tuple of latent events."
        )

    identifiers = [
        event.latent_event_id
        for event in events
    ]

    if len(identifiers) != len(set(identifiers)):
        raise SupportGateError(
            f"{label} event identifiers must be unique."
        )

    return events


def _validate_uncertainty_width(value: object) -> float:
    """Require a finite uncertainty width strictly between zero and one."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
    ):
        raise SupportGateError(
            "uncertainty_width must be finite and between 0 and 1."
        )

    return float(value)


class SupportGate:
    """Fit, validation-calibrate, and apply a preregistered support gate."""

    def __init__(
        self,
        *,
        method: SupportMethod,
        covariance_regularization: float = 1e-3,
    ) -> None:
        if not isinstance(method, SupportMethod):
            raise SupportGateError(
                "method must be a SupportMethod value."
            )

        if (
            isinstance(covariance_regularization, bool)
            or not isinstance(covariance_regularization, Real)
            or not math.isfinite(float(covariance_regularization))
            or float(covariance_regularization) <= 0.0
        ):
            raise SupportGateError(
                "covariance_regularization must be finite and positive."
            )

        self._method = method
        self._covariance_regularization = float(
            covariance_regularization,
        )
        self._dimension: int | None = None
        self._family_ids: tuple[int, ...] = ()
        self._prototypes: FloatArray | None = None
        self._inverse_covariances: tuple[FloatArray, ...] = ()
        self._training_event_count = 0
        self._threshold: float | None = None
        self._uncertainty_width: float | None = None
        self._calibration_report: SupportCalibrationReport | None = None

    @property
    def method(self) -> SupportMethod:
        """Return the registered comparison method."""

        return self._method

    @property
    def is_fitted(self) -> bool:
        """Return whether training-only fitting has completed."""

        return self._prototypes is not None

    @property
    def training_event_count(self) -> int:
        """Return the number of fitting events."""

        return self._training_event_count

    @property
    def training_family_count(self) -> int:
        """Return the number of fitted class-conditional families."""

        return len(self._family_ids)

    @property
    def threshold(self) -> float | None:
        """Return the validation-selected threshold, if calibrated."""

        return self._threshold

    @property
    def calibration_report(self) -> SupportCalibrationReport | None:
        """Return immutable calibration provenance."""

        return self._calibration_report

    def fit(
        self,
        events: tuple[LatentMultisensoryEvent, ...],
    ) -> "SupportGate":
        """Fit normalized class-conditional statistics on training only."""

        checked = _validate_event_tuple(
            events,
            label="fit events",
        )

        if any(
            event.split is not MultisensorySplit.TRAIN
            for event in checked
        ):
            raise SupportGateError(
                "Support-gate fitting permits training data only."
            )

        if any(
            event.support_regime is not SupportRegime.DEVELOPMENT
            for event in checked
        ):
            raise SupportGateError(
                "Training events must retain the development support label."
            )

        first = _normalize_vector(
            checked[0].olfactory_vector,
            expected_dimension=None,
            label="training vector",
        )
        dimension = int(first.shape[0])

        normalized = tuple(
            _normalize_vector(
                event.olfactory_vector,
                expected_dimension=dimension,
                label="training vector",
            )
            for event in checked
        )

        grouped: dict[int, list[FloatArray]] = {}

        for event, vector in zip(
            checked,
            normalized,
            strict=True,
        ):
            grouped.setdefault(
                event.target_family_id,
                [],
            ).append(vector)

        if len(grouped) < 2:
            raise SupportGateError(
                "At least two training families are required."
            )

        family_ids = tuple(sorted(grouped))
        prototypes: list[FloatArray] = []
        inverse_covariances: list[FloatArray] = []

        for family_id in family_ids:
            matrix = np.stack(
                grouped[family_id],
                axis=0,
            )
            centroid = matrix.mean(axis=0)
            centroid_norm = float(np.linalg.norm(centroid))

            if centroid_norm <= np.finfo(np.float64).eps:
                raise SupportGateError(
                    "A family prototype has a zero norm."
                )

            prototype = centroid / centroid_norm
            prototypes.append(prototype)

            centered = matrix - prototype

            if matrix.shape[0] > 1:
                covariance = (
                    centered.T @ centered
                ) / float(matrix.shape[0] - 1)
            else:
                covariance = np.zeros(
                    (dimension, dimension),
                    dtype=np.float64,
                )

            regularized = covariance + (
                self._covariance_regularization
                * np.eye(
                    dimension,
                    dtype=np.float64,
                )
            )
            inverse_covariances.append(
                np.linalg.pinv(
                    regularized,
                    hermitian=True,
                ),
            )

        self._dimension = dimension
        self._family_ids = family_ids
        self._prototypes = np.stack(
            prototypes,
            axis=0,
        )
        self._inverse_covariances = tuple(
            inverse_covariances,
        )
        self._training_event_count = len(checked)

        self._threshold = None
        self._uncertainty_width = None
        self._calibration_report = None

        return self

    def score(
        self,
        query_vector: Iterable[float],
    ) -> float:
        """Return a finite score where larger means more supported."""

        if self._prototypes is None or self._dimension is None:
            raise SupportGateError(
                "SupportGate must be fitted before scoring."
            )

        query = _normalize_vector(
            query_vector,
            expected_dimension=self._dimension,
            label="query vector",
        )

        if self._method is SupportMethod.COSINE_MARGIN:
            similarities = self._prototypes @ query
            ordered = np.sort(similarities)[::-1]
            score = float(ordered[0] - ordered[1])

        elif self._method is SupportMethod.NEAREST_PROTOTYPE_DISTANCE:
            distances = np.linalg.norm(
                self._prototypes - query,
                axis=1,
            )
            score = -float(np.min(distances))

        else:
            distances: list[float] = []

            for prototype, inverse_covariance in zip(
                self._prototypes,
                self._inverse_covariances,
                strict=True,
            ):
                difference = query - prototype
                squared = float(
                    difference.T
                    @ inverse_covariance
                    @ difference
                )
                distances.append(
                    math.sqrt(max(0.0, squared)),
                )

            score = -min(distances)

        if not math.isfinite(score):
            raise SupportGateError(
                "Support score must be finite."
            )

        return score

    def calibrate(
        self,
        events: tuple[LatentMultisensoryEvent, ...],
        *,
        uncertainty_width: float,
    ) -> SupportCalibrationReport:
        """Select a deterministic threshold from validation labels only."""

        if not self.is_fitted:
            raise SupportGateError(
                "SupportGate must be fitted before calibration."
            )

        checked = _validate_event_tuple(
            events,
            label="calibration events",
        )

        if any(
            event.split is not MultisensorySplit.VALIDATION
            for event in checked
        ):
            raise SupportGateError(
                "Threshold calibration permits validation data only."
            )

        width = _validate_uncertainty_width(
            uncertainty_width,
        )

        scores = tuple(
            self.score(event.olfactory_vector)
            for event in checked
        )
        labels = tuple(
            event.support_regime
            in {
                SupportRegime.SEEN_ITEM,
                SupportRegime.KNOWN_FAMILY_UNSEEN_ITEM,
            }
            for event in checked
        )

        supported_count = sum(labels)
        unsupported_count = len(labels) - supported_count

        if supported_count == 0 or unsupported_count == 0:
            raise SupportGateError(
                "Validation calibration requires supported and unsupported "
                "examples."
            )

        unique_scores = tuple(sorted(set(scores)))
        candidates: list[float] = [
            float(
                np.nextafter(
                    unique_scores[0],
                    -np.inf,
                ),
            ),
        ]

        candidates.extend(
            (
                left + right
            ) / 2.0
            for left, right in zip(
                unique_scores,
                unique_scores[1:],
            )
        )
        candidates.append(
            float(
                np.nextafter(
                    unique_scores[-1],
                    np.inf,
                ),
            ),
        )

        best_threshold: float | None = None
        best_balanced_accuracy = -1.0

        for candidate in candidates:
            predictions = tuple(
                score >= candidate
                for score in scores
            )

            true_positive_rate = (
                sum(
                    prediction and label
                    for prediction, label in zip(
                        predictions,
                        labels,
                        strict=True,
                    )
                )
                / supported_count
            )
            true_negative_rate = (
                sum(
                    (not prediction) and (not label)
                    for prediction, label in zip(
                        predictions,
                        labels,
                        strict=True,
                    )
                )
                / unsupported_count
            )
            balanced_accuracy = (
                true_positive_rate + true_negative_rate
            ) / 2.0

            if (
                balanced_accuracy > best_balanced_accuracy
                or (
                    balanced_accuracy == best_balanced_accuracy
                    and (
                        best_threshold is None
                        or candidate > best_threshold
                    )
                )
            ):
                best_balanced_accuracy = balanced_accuracy
                best_threshold = candidate

        if best_threshold is None or not math.isfinite(best_threshold):
            raise SupportGateError(
                "Validation calibration did not produce a finite threshold."
            )

        report = SupportCalibrationReport(
            method=self._method,
            source_split=MultisensorySplit.VALIDATION,
            validation_event_count=len(checked),
            supported_count=supported_count,
            unsupported_count=unsupported_count,
            threshold=best_threshold,
            uncertainty_width=width,
            balanced_accuracy=best_balanced_accuracy,
            final_test_labels_used=False,
        )

        self._threshold = best_threshold
        self._uncertainty_width = width
        self._calibration_report = report

        return report

    def decide(
        self,
        *,
        event_id: str,
        query_vector: Iterable[float],
    ) -> SupportDecision:
        """Score one query and apply the validation-locked policy."""

        score = self.score(query_vector)

        return self.decide_from_score(
            event_id=event_id,
            support_score=score,
        )

    def decide_from_score(
        self,
        *,
        event_id: str,
        support_score: float,
    ) -> SupportDecision:
        """Apply the locked threshold to an already computed score."""

        if (
            self._threshold is None
            or self._uncertainty_width is None
            or self._calibration_report is None
        ):
            raise SupportGateError(
                "SupportGate must be calibrated before decisions."
            )

        if not isinstance(event_id, str) or not event_id.strip():
            raise SupportGateError(
                "event_id must be a nonempty string."
            )

        if (
            isinstance(support_score, bool)
            or not isinstance(support_score, Real)
            or not math.isfinite(float(support_score))
        ):
            raise SupportGateError(
                "support_score must be finite."
            )

        score = float(support_score)
        lower = self._threshold - self._uncertainty_width
        upper = self._threshold + self._uncertainty_width

        if score < lower:
            uncertainty_status = (
                UncertaintyStatus.CERTAIN_UNSUPPORTED
            )
        elif score > upper:
            uncertainty_status = (
                UncertaintyStatus.CERTAIN_SUPPORTED
            )
        else:
            uncertainty_status = UncertaintyStatus.UNCERTAIN

        return SupportDecision(
            event_id=event_id,
            method=self._method,
            support_score=score,
            threshold=self._threshold,
            is_supported=score >= self._threshold,
            uncertainty_status=uncertainty_status,
            request_touch=(
                uncertainty_status
                is UncertaintyStatus.UNCERTAIN
            ),
        )
