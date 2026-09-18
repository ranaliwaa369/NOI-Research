"""Executable open-world odor decision engine for NOI v0.5.

Unlike the v0.5 partition module, this module trains an actual model on
physical-sensor windows and emits auditable recognize/repeat/abstain actions.
It is intentionally lightweight so the complete pilot can run on a laptop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.evaluation.smellnet_novel_odor_protocol import (
    RepeatSensingPair,
    TemporalSensorWindow,
)


class OdorSentinelError(ValueError):
    """Raised when the executable sentinel receives invalid evidence."""


@dataclass(frozen=True, slots=True)
class SentinelCalibration:
    """Validation-selected operating point for one family-holdout fold."""

    support_threshold: float
    repeat_margin: float
    required_known_acceptance: float
    achieved_known_acceptance: float
    achieved_unknown_rejection: float
    memory_distance_scale: float


@dataclass(frozen=True, slots=True)
class SentinelDecision:
    """One traceable decision from physical temporal evidence."""

    window_id: str
    recording_id: str
    true_family: str
    predicted_family: str
    action: str
    classifier_confidence: float
    memory_support: float
    combined_support: float
    novelty_score: float
    classifier_memory_agree: bool
    used_repeat: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SentinelMetrics:
    """Decision metrics for known and genuinely held-out families."""

    sample_count: int
    known_count: int
    unknown_count: int
    false_known_rate: float
    unknown_rejection_rate: float
    known_acceptance_rate: float
    known_top1_accuracy: float
    selective_known_accuracy: float
    unknown_auroc: float | None
    unknown_aupr: float | None
    decision_utility: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_temporal_features(window: TemporalSensorWindow) -> np.ndarray:
    """Convert a six-channel sensor window into a compact dynamic signature.

    Each channel contributes level, dispersion, range, temporal-difference,
    and trend statistics. Labels and path metadata never enter the vector.
    """

    matrix = np.asarray(window.sensor_rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] != 6:
        raise OdorSentinelError(
            "A temporal feature requires at least two rows and six channels."
        )
    if not np.isfinite(matrix).all():
        raise OdorSentinelError("Sensor evidence must be finite.")

    differences = np.diff(matrix, axis=0)
    time = np.linspace(-1.0, 1.0, matrix.shape[0], dtype=np.float64)
    denominator = float(np.dot(time, time))
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    slopes = np.dot(time, centered) / denominator

    blocks = (
        matrix.mean(axis=0),
        matrix.std(axis=0),
        np.median(matrix, axis=0),
        np.quantile(matrix, 0.25, axis=0),
        np.quantile(matrix, 0.75, axis=0),
        matrix.min(axis=0),
        matrix.max(axis=0),
        matrix[-1] - matrix[0],
        np.mean(np.abs(differences), axis=0),
        differences.std(axis=0),
        slopes,
    )
    features = np.concatenate(blocks).astype(np.float64, copy=False)
    if not np.isfinite(features).all():
        raise OdorSentinelError("Extracted temporal features are not finite.")
    return features


def _stack_features(
    windows: Sequence[TemporalSensorWindow],
) -> tuple[np.ndarray, np.ndarray]:
    if not windows:
        raise OdorSentinelError("At least one temporal window is required.")
    features = np.vstack([extract_temporal_features(item) for item in windows])
    labels = np.asarray([item.family for item in windows], dtype=object)
    return features, labels


class OdorSentinel:
    """Classifier, associative memory, novelty gate, and repeat controller."""

    def __init__(self, *, random_state: int = 1729) -> None:
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
        self._memory_features: np.ndarray | None = None
        self._memory_labels: np.ndarray | None = None
        self.calibration: SentinelCalibration | None = None

    def fit(self, windows: Iterable[TemporalSensorWindow]) -> "OdorSentinel":
        """Train the family classifier and training-only associative memory."""

        values = tuple(windows)
        features, labels = _stack_features(values)
        if len(set(labels.tolist())) < 2:
            raise OdorSentinelError("Training requires at least two families.")
        transformed = self.scaler.fit_transform(features)
        self.classifier.fit(transformed, labels)
        self._memory_features = transformed
        self._memory_labels = labels
        return self

    def _require_fit(self) -> tuple[np.ndarray, np.ndarray]:
        if self._memory_features is None or self._memory_labels is None:
            raise OdorSentinelError("The sentinel must be fit before use.")
        return self._memory_features, self._memory_labels

    def _raw_evidence_from_features(
        self,
        features: np.ndarray,
        *,
        distance_scale: float,
    ) -> tuple[str, float, float, float, bool]:
        memory_features, memory_labels = self._require_fit()
        vector = self.scaler.transform(features.reshape(1, -1))[0]
        probabilities = self.classifier.predict_proba(vector.reshape(1, -1))[0]
        classifier_index = int(np.argmax(probabilities))
        predicted_family = str(self.classifier.classes_[classifier_index])
        classifier_confidence = float(probabilities[classifier_index])

        distances = np.linalg.norm(memory_features - vector, axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        memory_family = str(memory_labels[nearest_index])
        scale = max(float(distance_scale), np.finfo(np.float64).eps)
        memory_support = float(math.exp(-nearest_distance / scale))
        agreement = predicted_family == memory_family
        agreement_factor = 1.0 if agreement else 0.72
        combined_support = float(
            math.sqrt(classifier_confidence * memory_support)
            * agreement_factor
        )
        return (
            predicted_family,
            classifier_confidence,
            memory_support,
            combined_support,
            agreement,
        )

    def _training_distance_scale(self) -> float:
        memory_features, _ = self._require_fit()
        if len(memory_features) < 2:
            raise OdorSentinelError("Memory needs at least two entries.")
        distances = np.linalg.norm(
            memory_features[:, None, :] - memory_features[None, :, :],
            axis=2,
        )
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        positive = nearest[np.isfinite(nearest) & (nearest > 0)]
        if not len(positive):
            return 1.0
        return float(np.median(positive))

    def calibrate(
        self,
        known_windows: Iterable[TemporalSensorWindow],
        unknown_windows: Iterable[TemporalSensorWindow],
        *,
        required_known_acceptance: float = 0.80,
        repeat_margin: float = 0.06,
    ) -> SentinelCalibration:
        """Select a novelty threshold using validation evidence only."""

        if not 0 < required_known_acceptance <= 1:
            raise OdorSentinelError(
                "required_known_acceptance must be in (0, 1]."
            )
        if repeat_margin < 0 or repeat_margin >= 1:
            raise OdorSentinelError("repeat_margin must be in [0, 1).")
        known = tuple(known_windows)
        unknown = tuple(unknown_windows)
        if not known or not unknown:
            raise OdorSentinelError(
                "Calibration needs known and unknown validation windows."
            )

        distance_scale = self._training_distance_scale()

        def score(window: TemporalSensorWindow) -> float:
            return self._raw_evidence_from_features(
                extract_temporal_features(window),
                distance_scale=distance_scale,
            )[3]

        known_scores = np.asarray([score(item) for item in known])
        unknown_scores = np.asarray([score(item) for item in unknown])
        candidates = np.unique(np.concatenate((known_scores, unknown_scores)))
        best: tuple[float, float, float, float] | None = None
        for threshold in candidates:
            known_acceptance = float(np.mean(known_scores >= threshold))
            if known_acceptance + 1e-12 < required_known_acceptance:
                continue
            unknown_rejection = float(np.mean(unknown_scores < threshold))
            balanced = 0.5 * (known_acceptance + unknown_rejection)
            candidate = (
                balanced,
                unknown_rejection,
                float(threshold),
                known_acceptance,
            )
            if best is None or candidate > best:
                best = candidate
        if best is None:
            threshold = float(np.min(known_scores))
            known_acceptance = 1.0
            unknown_rejection = float(np.mean(unknown_scores < threshold))
        else:
            _, unknown_rejection, threshold, known_acceptance = best

        self.calibration = SentinelCalibration(
            support_threshold=threshold,
            repeat_margin=repeat_margin,
            required_known_acceptance=required_known_acceptance,
            achieved_known_acceptance=known_acceptance,
            achieved_unknown_rejection=unknown_rejection,
            memory_distance_scale=distance_scale,
        )
        return self.calibration

    def _decision_from_features(
        self,
        window: TemporalSensorWindow,
        features: np.ndarray,
        *,
        allow_repeat: bool,
        used_repeat: bool,
    ) -> SentinelDecision:
        if self.calibration is None:
            raise OdorSentinelError("The sentinel must be calibrated first.")
        (
            predicted,
            confidence,
            memory_support,
            combined_support,
            agreement,
        ) = self._raw_evidence_from_features(
            features,
            distance_scale=self.calibration.memory_distance_scale,
        )
        threshold = self.calibration.support_threshold
        ambiguous = abs(combined_support - threshold) <= (
            self.calibration.repeat_margin
        )
        if allow_repeat and ambiguous:
            action = "repeat_sense"
        elif combined_support >= threshold:
            action = "recognize"
        else:
            action = "abstain_unknown"
        return SentinelDecision(
            window_id=window.window_id,
            recording_id=window.recording_id,
            true_family=window.family,
            predicted_family=predicted,
            action=action,
            classifier_confidence=confidence,
            memory_support=memory_support,
            combined_support=combined_support,
            novelty_score=1.0 - combined_support,
            classifier_memory_agree=agreement,
            used_repeat=used_repeat,
        )

    def decide(
        self,
        window: TemporalSensorWindow,
        *,
        allow_repeat: bool = True,
    ) -> SentinelDecision:
        return self._decision_from_features(
            window,
            extract_temporal_features(window),
            allow_repeat=allow_repeat,
            used_repeat=False,
        )

    def predict_memory_family(self, window: TemporalSensorWindow) -> str:
        """Return the nearest training-only associative-memory family."""

        memory_features, memory_labels = self._require_fit()
        features = extract_temporal_features(window)
        vector = self.scaler.transform(features.reshape(1, -1))[0]
        nearest_index = int(
            np.argmin(np.linalg.norm(memory_features - vector, axis=1))
        )
        return str(memory_labels[nearest_index])

    def decide_pair(self, pair: RepeatSensingPair) -> SentinelDecision:
        """Fuse two disjoint sensor windows into one terminal decision."""

        initial = extract_temporal_features(pair.initial)
        repeat = extract_temporal_features(pair.repeat)
        fused = 0.5 * (initial + repeat)
        return self._decision_from_features(
            pair.initial,
            fused,
            allow_repeat=False,
            used_repeat=True,
        )

    def decide_pair_asymmetric_veto(
        self,
        pair: RepeatSensingPair,
    ) -> SentinelDecision:
        """Use repeat sensing as a safety veto, never as recognition promotion.

        The initial window determines the maximum-permitted action. A repeat
        may veto an initially recognized odor when fused evidence falls below
        support, but it may not turn an initial abstention into recognition.
        This asymmetry prevents a noisy second window from manufacturing
        positive evidence for an unsupported odor.
        """

        repeat_request = self.decide(pair.initial, allow_repeat=True)
        initial_terminal = self.decide(pair.initial, allow_repeat=False)
        if repeat_request.action != "repeat_sense":
            return initial_terminal

        fused = self.decide_pair(pair)
        if (
            initial_terminal.action == "recognize"
            and fused.action != "recognize"
        ):
            return replace(fused, action="abstain_conflict", used_repeat=True)

        return replace(initial_terminal, used_repeat=True)


def evaluate_sentinel_decisions(
    decisions: Iterable[SentinelDecision],
    *,
    unknown_family: str,
) -> SentinelMetrics:
    """Score terminal decisions without treating abstention as a class label."""

    values = tuple(decisions)
    if not values:
        raise OdorSentinelError("At least one decision is required.")
    if any(value.action == "repeat_sense" for value in values):
        raise OdorSentinelError(
            "Metrics require terminal recognize/abstain decisions."
        )
    is_unknown = np.asarray(
        [value.true_family == unknown_family for value in values],
        dtype=bool,
    )
    is_known = ~is_unknown
    recognized = np.asarray(
        [value.action == "recognize" for value in values],
        dtype=bool,
    )
    correct_family = np.asarray(
        [value.predicted_family == value.true_family for value in values],
        dtype=bool,
    )
    novelty = np.asarray([value.novelty_score for value in values])

    def safe_mean(mask: np.ndarray) -> float:
        return float(np.mean(mask)) if len(mask) else 0.0

    known_recognized = recognized[is_known]
    known_correct = correct_family[is_known]
    selective_mask = is_known & recognized
    false_known_rate = safe_mean(recognized[is_unknown])
    unknown_rejection = safe_mean((~recognized)[is_unknown])
    known_acceptance = safe_mean(known_recognized)
    known_accuracy = safe_mean(known_correct)
    selective_accuracy = (
        safe_mean(correct_family[selective_mask])
        if np.any(selective_mask)
        else 0.0
    )

    labels = is_unknown.astype(int)
    unknown_auroc = (
        float(roc_auc_score(labels, novelty))
        if len(np.unique(labels)) == 2
        else None
    )
    unknown_aupr = (
        float(average_precision_score(labels, novelty))
        if len(np.unique(labels)) == 2
        else None
    )

    utilities: list[float] = []
    for unknown, accept, correct in zip(
        is_unknown, recognized, correct_family, strict=True
    ):
        if unknown:
            utilities.append(-2.0 if accept else 1.0)
        elif accept:
            utilities.append(1.0 if correct else -1.0)
        else:
            utilities.append(-0.25)

    return SentinelMetrics(
        sample_count=len(values),
        known_count=int(np.sum(is_known)),
        unknown_count=int(np.sum(is_unknown)),
        false_known_rate=false_known_rate,
        unknown_rejection_rate=unknown_rejection,
        known_acceptance_rate=known_acceptance,
        known_top1_accuracy=known_accuracy,
        selective_known_accuracy=selective_accuracy,
        unknown_auroc=unknown_auroc,
        unknown_aupr=unknown_aupr,
        decision_utility=float(np.mean(utilities)),
    )
