"""Metadata-blind retrieval mechanics for locked NOI v0.3 execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from src.evaluation.multisensory_records import (
    LatentMultisensoryEvent,
    MultisensorySplit,
    MultisensoryTarget,
)
from src.evaluation.reliability_fusion import (
    FusionAction,
    LockedFusionDecision,
)
from src.evaluation.support_gate import (
    SupportDecision,
    UncertaintyStatus,
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

class NOIV03Modality(str, Enum):
    """Registered independent modality spaces."""

    ODOR = "odor"
    TOUCH = "touch"


class NOIV03RidgeRetriever:
    """Training-only ridge mapping in one registered modality space."""

    def __init__(
        self,
        *,
        modality: NOIV03Modality,
        alpha: float = 1.0,
    ) -> None:
        if not isinstance(modality, NOIV03Modality):
            raise NOIV03RetrievalError(
                "modality must be a NOIV03Modality value."
            )

        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, Real)
            or not math.isfinite(float(alpha))
            or float(alpha) <= 0.0
        ):
            raise NOIV03RetrievalError(
                "alpha must be a finite positive number."
            )

        self._modality = modality
        self._alpha = float(alpha)
        self._model: Ridge | None = None
        self._library: NOIV03RetrievalLibrary | None = None
        self._training_event_count = 0

    @property
    def modality(self) -> NOIV03Modality:
        """Return the registered modality."""

        return self._modality

    @property
    def alpha(self) -> float:
        """Return the fixed ridge penalty."""

        return self._alpha

    @property
    def is_fitted(self) -> bool:
        """Return whether training-only fitting completed."""

        return self._model is not None and self._library is not None

    @property
    def training_event_count(self) -> int:
        """Return the number of training records used."""

        return self._training_event_count

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return the represented training candidate identifiers."""

        if self._library is None:
            return ()

        return self._library.item_ids

    def fit(
        self,
        *,
        training_events: Sequence[LatentMultisensoryEvent],
        targets: Sequence[MultisensoryTarget],
    ) -> "NOIV03RidgeRetriever":
        """Fit one modality mapping using training records only."""

        library = NOIV03RetrievalLibrary.from_training_records(
            training_events=training_events,
            targets=targets,
        )

        target_map = {
            item.item_id: item
            for item in targets
        }

        ordered_events = tuple(
            sorted(
                training_events,
                key=lambda event: event.latent_event_id,
            )
        )

        if self._modality is NOIV03Modality.ODOR:
            expected_dimension = 16
            input_vectors = tuple(
                event.olfactory_vector
                for event in ordered_events
            )
            target_vectors = tuple(
                target_map[event.target_item_id]
                .olfactory_prototype
                for event in ordered_events
            )
            input_label = "training olfactory vector"
            target_label = "target olfactory prototype"
        else:
            expected_dimension = 8
            input_vectors = tuple(
                event.tactile_vector
                for event in ordered_events
            )
            target_vectors = tuple(
                target_map[event.target_item_id]
                .tactile_prototype
                for event in ordered_events
            )
            input_label = "training tactile vector"
            target_label = "target tactile prototype"

        input_matrix = np.stack(
            [
                _normalized_vector(
                    vector,
                    expected_dimension=expected_dimension,
                    label=input_label,
                )
                for vector in input_vectors
            ],
            axis=0,
        )

        target_matrix = np.stack(
            [
                _normalized_vector(
                    vector,
                    expected_dimension=expected_dimension,
                    label=target_label,
                )
                for vector in target_vectors
            ],
            axis=0,
        )

        model = Ridge(
            alpha=self._alpha,
            fit_intercept=True,
        )
        model.fit(
            input_matrix,
            target_matrix,
        )

        self._model = model
        self._library = library
        self._training_event_count = len(ordered_events)

        return self

    def retrieve(
        self,
        *,
        event_id: str,
        query_vector: Iterable[float],
        top_k: int = 10,
    ) -> NOIV03RetrievalResult:
        """Predict one modality prototype and rank training memory."""

        if self._model is None or self._library is None:
            raise NOIV03RetrievalError(
                "NOIV03RidgeRetriever must be fitted before retrieval."
            )

        expected_dimension = (
            16
            if self._modality is NOIV03Modality.ODOR
            else 8
        )

        normalized_query = _normalized_vector(
            query_vector,
            expected_dimension=expected_dimension,
            label=f"{self._modality.value} query_vector",
        )

        predicted = np.asarray(
            self._model.predict(
                normalized_query.reshape(1, -1)
            )[0],
            dtype=np.float64,
        )

        if (
            predicted.ndim != 1
            or predicted.shape[0] != expected_dimension
            or not np.all(np.isfinite(predicted))
        ):
            raise NOIV03RetrievalError(
                "Ridge prediction must be one finite modality vector."
            )

        if self._modality is NOIV03Modality.ODOR:
            return self._library.rank(
                event_id=event_id,
                olfactory_vector=tuple(
                    float(value)
                    for value in predicted
                ),
                tactile_vector=None,
                odor_weight=1.0,
                touch_weight=0.0,
                top_k=top_k,
            )

        return self._library.rank(
            event_id=event_id,
            olfactory_vector=None,
            tactile_vector=tuple(
                float(value)
                for value in predicted
            ),
            odor_weight=0.0,
            touch_weight=1.0,
            top_k=top_k,
        )

class NOIV03System(str, Enum):
    """Nine systems registered for v0.3 confirmatory execution."""

    ODOR_ONLY_RIDGE = "odor_only_ridge"
    ODOR_ONLY_COSINE = "odor_only_cosine"
    TOUCH_ONLY_RIDGE = "touch_only_ridge"
    TOUCH_ONLY_COSINE = "touch_only_cosine"
    NAIVE_CONCATENATION = "naive_concatenation"
    FIXED_WEIGHT_FUSION = "fixed_weight_fusion"
    SUPPORT_GATE_ODOR_ONLY = "support_gate_odor_only"
    RELIABILITY_GATED_OLFACTORY_TACTILE_FUSION = (
        "reliability_gated_olfactory_tactile_fusion"
    )
    SUPPORT_GATE_RELIABILITY_FUSION_WITH_ABSTENTION = (
        "support_gate_reliability_fusion_with_abstention"
    )


@dataclass(frozen=True, slots=True)
class NOIV03SystemResult:
    """One named-system inference result."""

    system: NOIV03System
    retrieval: NOIV03RetrievalResult

    def __post_init__(self) -> None:
        if not isinstance(self.system, NOIV03System):
            raise NOIV03RetrievalError(
                "system must be a NOIV03System value."
            )

        if not isinstance(
            self.retrieval,
            NOIV03RetrievalResult,
        ):
            raise NOIV03RetrievalError(
                "retrieval must be a NOIV03RetrievalResult."
            )


class NOIV03SystemPolicy:
    """Apply the nine preconfirmatory retrieval definitions."""

    def __init__(
        self,
        *,
        library: NOIV03RetrievalLibrary,
        odor_ridge: NOIV03RidgeRetriever,
        touch_ridge: NOIV03RidgeRetriever,
    ) -> None:
        if not isinstance(library, NOIV03RetrievalLibrary):
            raise NOIV03RetrievalError(
                "library must be a NOIV03RetrievalLibrary."
            )

        if (
            not isinstance(odor_ridge, NOIV03RidgeRetriever)
            or odor_ridge.modality is not NOIV03Modality.ODOR
            or not odor_ridge.is_fitted
        ):
            raise NOIV03RetrievalError(
                "odor_ridge must be a fitted odor ridge retriever."
            )

        if (
            not isinstance(touch_ridge, NOIV03RidgeRetriever)
            or touch_ridge.modality is not NOIV03Modality.TOUCH
            or not touch_ridge.is_fitted
        ):
            raise NOIV03RetrievalError(
                "touch_ridge must be a fitted touch ridge retriever."
            )

        if (
            library.item_ids != odor_ridge.item_ids
            or library.item_ids != touch_ridge.item_ids
        ):
            raise NOIV03RetrievalError(
                "All retrieval systems must share one candidate library."
            )

        self._library = library
        self._odor_ridge = odor_ridge
        self._touch_ridge = touch_ridge

    @classmethod
    def fit(
        cls,
        *,
        training_events: Sequence[LatentMultisensoryEvent],
        targets: Sequence[MultisensoryTarget],
        ridge_alpha: float = 1.0,
    ) -> "NOIV03SystemPolicy":
        """Fit all trainable components on training records only."""

        library = NOIV03RetrievalLibrary.from_training_records(
            training_events=training_events,
            targets=targets,
        )

        odor_ridge = NOIV03RidgeRetriever(
            modality=NOIV03Modality.ODOR,
            alpha=ridge_alpha,
        ).fit(
            training_events=training_events,
            targets=targets,
        )

        touch_ridge = NOIV03RidgeRetriever(
            modality=NOIV03Modality.TOUCH,
            alpha=ridge_alpha,
        ).fit(
            training_events=training_events,
            targets=targets,
        )

        return cls(
            library=library,
            odor_ridge=odor_ridge,
            touch_ridge=touch_ridge,
        )

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return the shared training-only candidates."""

        return self._library.item_ids

    @staticmethod
    def _available_weights(
        *,
        olfactory_vector: Iterable[float] | None,
        tactile_vector: Iterable[float] | None,
    ) -> tuple[float, float]:
        """Return equal renormalized weights for available modalities."""

        odor_available = olfactory_vector is not None
        touch_available = tactile_vector is not None

        if odor_available and touch_available:
            return 0.5, 0.5

        if odor_available:
            return 1.0, 0.0

        if touch_available:
            return 0.0, 1.0

        raise NOIV03RetrievalError(
            "At least one modality must be available."
        )

    @staticmethod
    def _require_support_decision(
        decision: SupportDecision | None,
        *,
        event_id: str,
    ) -> SupportDecision:
        """Return one event-aligned locked support decision."""

        if not isinstance(decision, SupportDecision):
            raise NOIV03RetrievalError(
                "support_decision must be supplied for this system."
            )

        if decision.event_id != event_id:
            raise NOIV03RetrievalError(
                "support_decision event_id must match the query."
            )

        return decision

    @staticmethod
    def _require_fusion_decision(
        decision: LockedFusionDecision | None,
        *,
        event_id: str,
    ) -> LockedFusionDecision:
        """Return one event-aligned locked fusion decision."""

        if not isinstance(decision, LockedFusionDecision):
            raise NOIV03RetrievalError(
                "fusion_decision must be supplied for this system."
            )

        if decision.event_id != event_id:
            raise NOIV03RetrievalError(
                "fusion_decision event_id must match the query."
            )

        if (
            decision.trace.condition_metadata_used
            or decision.trace.target_labels_used
            or decision.trace.final_test_labels_used
        ):
            raise NOIV03RetrievalError(
                "fusion_decision must be metadata-blind."
            )

        return decision

    def _wrap(
        self,
        *,
        system: NOIV03System,
        retrieval: NOIV03RetrievalResult,
    ) -> NOIV03SystemResult:
        """Attach the registered system identifier."""

        return NOIV03SystemResult(
            system=system,
            retrieval=retrieval,
        )

    def _apply_fusion(
        self,
        *,
        system: NOIV03System,
        event_id: str,
        olfactory_vector: Iterable[float] | None,
        tactile_vector: Iterable[float] | None,
        fusion_decision: LockedFusionDecision,
        top_k: int,
    ) -> NOIV03SystemResult:
        """Apply one validated locked fusion action."""

        decision = self._require_fusion_decision(
            fusion_decision,
            event_id=event_id,
        )

        if (
            decision.action is FusionAction.ABSTAIN
            or decision.abstained
        ):
            return self._wrap(
                system=system,
                retrieval=self._library.abstain(
                    event_id=event_id,
                    reason=(
                        "The validation-locked fusion policy "
                        "required abstention."
                    ),
                ),
            )

        return self._wrap(
            system=system,
            retrieval=self._library.rank(
                event_id=event_id,
                olfactory_vector=olfactory_vector,
                tactile_vector=tactile_vector,
                odor_weight=decision.odor_weight,
                touch_weight=decision.touch_weight,
                top_k=top_k,
            ),
        )

    def evaluate(
        self,
        *,
        system: NOIV03System,
        event_id: str,
        olfactory_vector: Iterable[float] | None,
        tactile_vector: Iterable[float] | None,
        support_decision: SupportDecision | None = None,
        fusion_decision: LockedFusionDecision | None = None,
        top_k: int = 10,
    ) -> NOIV03SystemResult:
        """Evaluate one system without ground-truth or condition inputs."""

        if not isinstance(system, NOIV03System):
            raise NOIV03RetrievalError(
                "system must be a NOIV03System value."
            )

        if system is NOIV03System.ODOR_ONLY_RIDGE:
            if olfactory_vector is None:
                retrieval = self._library.abstain(
                    event_id=event_id,
                    reason="Odor is unavailable.",
                )
            else:
                retrieval = self._odor_ridge.retrieve(
                    event_id=event_id,
                    query_vector=olfactory_vector,
                    top_k=top_k,
                )
            return self._wrap(system=system, retrieval=retrieval)

        if system is NOIV03System.TOUCH_ONLY_RIDGE:
            if tactile_vector is None:
                retrieval = self._library.abstain(
                    event_id=event_id,
                    reason="Touch is unavailable.",
                )
            else:
                retrieval = self._touch_ridge.retrieve(
                    event_id=event_id,
                    query_vector=tactile_vector,
                    top_k=top_k,
                )
            return self._wrap(system=system, retrieval=retrieval)

        if system is NOIV03System.ODOR_ONLY_COSINE:
            if olfactory_vector is None:
                retrieval = self._library.abstain(
                    event_id=event_id,
                    reason="Odor is unavailable.",
                )
            else:
                retrieval = self._library.rank(
                    event_id=event_id,
                    olfactory_vector=olfactory_vector,
                    tactile_vector=None,
                    odor_weight=1.0,
                    touch_weight=0.0,
                    top_k=top_k,
                )
            return self._wrap(system=system, retrieval=retrieval)

        if system is NOIV03System.TOUCH_ONLY_COSINE:
            if tactile_vector is None:
                retrieval = self._library.abstain(
                    event_id=event_id,
                    reason="Touch is unavailable.",
                )
            else:
                retrieval = self._library.rank(
                    event_id=event_id,
                    olfactory_vector=None,
                    tactile_vector=tactile_vector,
                    odor_weight=0.0,
                    touch_weight=1.0,
                    top_k=top_k,
                )
            return self._wrap(system=system, retrieval=retrieval)

        if system in (
            NOIV03System.NAIVE_CONCATENATION,
            NOIV03System.FIXED_WEIGHT_FUSION,
        ):
            odor_weight, touch_weight = self._available_weights(
                olfactory_vector=olfactory_vector,
                tactile_vector=tactile_vector,
            )
            return self._wrap(
                system=system,
                retrieval=self._library.rank(
                    event_id=event_id,
                    olfactory_vector=olfactory_vector,
                    tactile_vector=tactile_vector,
                    odor_weight=odor_weight,
                    touch_weight=touch_weight,
                    top_k=top_k,
                ),
            )

        if system is NOIV03System.SUPPORT_GATE_ODOR_ONLY:
            decision = self._require_support_decision(
                support_decision,
                event_id=event_id,
            )

            if not decision.is_supported or olfactory_vector is None:
                retrieval = self._library.abstain(
                    event_id=event_id,
                    reason=(
                        "The validation-locked support gate "
                        "rejected odor-only identity retrieval."
                    ),
                )
            else:
                retrieval = self._library.rank(
                    event_id=event_id,
                    olfactory_vector=olfactory_vector,
                    tactile_vector=None,
                    odor_weight=1.0,
                    touch_weight=0.0,
                    top_k=top_k,
                )

            return self._wrap(system=system, retrieval=retrieval)

        if system is (
            NOIV03System
            .RELIABILITY_GATED_OLFACTORY_TACTILE_FUSION
        ):
            return self._apply_fusion(
                system=system,
                event_id=event_id,
                olfactory_vector=olfactory_vector,
                tactile_vector=tactile_vector,
                fusion_decision=self._require_fusion_decision(
                    fusion_decision,
                    event_id=event_id,
                ),
                top_k=top_k,
            )

        decision = self._require_support_decision(
            support_decision,
            event_id=event_id,
        )

        if (
            decision.uncertainty_status
            is UncertaintyStatus.CERTAIN_UNSUPPORTED
        ):
            return self._wrap(
                system=system,
                retrieval=self._library.abstain(
                    event_id=event_id,
                    reason=(
                        "The support gate classified the query "
                        "as certainly unsupported."
                    ),
                ),
            )

        use_touch = decision.request_touch

        if (
            not use_touch
            and fusion_decision is not None
            and tactile_vector is not None
        ):
            checked_fusion = self._require_fusion_decision(
                fusion_decision,
                event_id=event_id,
            )
            use_touch = (
                checked_fusion.odor_reliability
                < checked_fusion.trace.reliability_threshold
            )

        if use_touch:
            return self._apply_fusion(
                system=system,
                event_id=event_id,
                olfactory_vector=olfactory_vector,
                tactile_vector=tactile_vector,
                fusion_decision=self._require_fusion_decision(
                    fusion_decision,
                    event_id=event_id,
                ),
                top_k=top_k,
            )

        if not decision.is_supported or olfactory_vector is None:
            return self._wrap(
                system=system,
                retrieval=self._library.abstain(
                    event_id=event_id,
                    reason=(
                        "Supported odor-only retrieval was unavailable."
                    ),
                ),
            )

        return self._wrap(
            system=system,
            retrieval=self._library.rank(
                event_id=event_id,
                olfactory_vector=olfactory_vector,
                tactile_vector=None,
                odor_weight=1.0,
                touch_weight=0.0,
                top_k=top_k,
            ),
        )
