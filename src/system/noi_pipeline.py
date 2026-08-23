"""Integrated Neuro-Olfactive Intelligence retrieval pipeline.

This module integrates deterministic multimodal fusion, a train-only ridge
projection, temporal associative memory, corrective updating, and a separate
fail-closed simulated policy gate.

The implementation is for synthetic computational evaluation only. It does
not control physical odor emission and does not establish perceptual,
clinical, chemical, or physical safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from src.baselines.retrieval_baselines import (
    build_odor_library,
    mean_fuse_event,
)
from src.correction.corrective_update import (
    CorrectionAuditRecord,
    CorrectiveMemoryUpdater,
)
from src.encoders.mean_fusion import mean_fuse_context
from src.evaluation.retrieval_metrics import mean_reciprocal_rank
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
    SyntheticEvent,
)
from src.memory.records import (
    AssociativeMemoryRecord,
)
from src.memory.temporal_memory import TemporalAssociativeMemory
from src.models import (
    MultimodalContext,
    OdorLibraryItem,
    OutputRequest,
    PolicyDecision,
)
from src.safety.policy_gate import DeterministicPolicyGate


FloatArray = NDArray[np.float64]

DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_MEMORY_DECAY_RATE_PER_DAY = 0.01
DEFAULT_ALPHA_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_TOP_K = 10


class NOIPipelineError(ValueError):
    """Raised when the integrated NOI pipeline cannot operate safely."""


@dataclass(frozen=True)
class HybridRetrievalCandidate:
    """One ranked odor candidate produced by hybrid retrieval."""

    item_id: str
    hybrid_score: float
    library_score: float
    memory_score: float
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id.strip():
            raise NOIPipelineError("Candidate item_id must not be empty.")

        for label, value in (
            ("hybrid_score", self.hybrid_score),
            ("library_score", self.library_score),
            ("memory_score", self.memory_score),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise NOIPipelineError(
                    f"{label} must be finite and in [0, 1]."
                )

        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise NOIPipelineError("Candidate rank must be positive.")


@dataclass(frozen=True)
class NOIRetrievalResult:
    """Immutable result of one integrated NOI retrieval."""

    event_id: str
    timestamp_utc: datetime
    selected_alpha: float
    candidates: tuple[HybridRetrievalCandidate, ...]
    temporal_decay_applied: bool
    protocol_hash: str
    oracle_used: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise NOIPipelineError("event_id must not be empty.")

        if (
            self.timestamp_utc.tzinfo is None
            or self.timestamp_utc.utcoffset() is None
        ):
            raise NOIPipelineError(
                "timestamp_utc must be timezone-aware."
            )

        if not 0.0 <= self.selected_alpha <= 1.0:
            raise NOIPipelineError(
                "selected_alpha must be in [0, 1]."
            )

        if not self.candidates:
            raise NOIPipelineError(
                "A retrieval result must contain candidates."
            )

        expected_ranks = tuple(
            range(1, len(self.candidates) + 1)
        )
        observed_ranks = tuple(
            candidate.rank for candidate in self.candidates
        )

        if observed_ranks != expected_ranks:
            raise NOIPipelineError(
                "Candidate ranks must be consecutive from one."
            )

        if not isinstance(self.protocol_hash, str) or not self.protocol_hash:
            raise NOIPipelineError(
                "protocol_hash must not be empty."
            )

        if self.oracle_used is not False:
            raise NOIPipelineError(
                "The deployable NOI pipeline cannot use an OOD oracle."
            )


def load_noi_system_configuration(
    path: str | Path,
) -> dict[str, Any]:
    """Load and minimally validate the locked NOI system definition."""

    configuration_path = Path(path)

    if not configuration_path.is_file():
        raise NOIPipelineError(
            f"NOI system configuration not found: {configuration_path}"
        )

    try:
        with configuration_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            configuration = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise NOIPipelineError(
            "Could not load the NOI system configuration."
        ) from error

    if not isinstance(configuration, dict):
        raise NOIPipelineError(
            "NOI system configuration must be a mapping."
        )

    hybrid = configuration.get("hybrid_retrieval")

    if not isinstance(hybrid, dict):
        raise NOIPipelineError(
            "hybrid_retrieval configuration is required."
        )

    if hybrid.get("negative_cosine_handling") != "clip_to_zero":
        raise NOIPipelineError(
            "negative_cosine_handling must be clip_to_zero."
        )

    if hybrid.get("absent_memory_item_score") != 0.0:
        raise NOIPipelineError(
            "absent_memory_item_score must be 0.0."
        )

    if hybrid.get("memory_aggregation_per_odor") != "maximum":
        raise NOIPipelineError(
            "memory_aggregation_per_odor must be maximum."
        )

    alpha_selection = hybrid.get("alpha_selection")

    if not isinstance(alpha_selection, dict):
        raise NOIPipelineError(
            "hybrid_retrieval.alpha_selection is required."
        )

    candidates = alpha_selection.get("candidates")

    if candidates != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise NOIPipelineError(
            "Alpha candidates must match the locked definition."
        )

    return configuration


class NOIPipeline:
    """Integrated deterministic NOI retrieval and governance pipeline."""

    def __init__(
        self,
        *,
        system_configuration: Mapping[str, Any],
        policy_configuration: Mapping[str, Any],
        protocol_hash: str,
        ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
        memory_decay_rate_per_day: float = (
            DEFAULT_MEMORY_DECAY_RATE_PER_DAY
        ),
    ) -> None:
        if not isinstance(system_configuration, Mapping):
            raise NOIPipelineError(
                "system_configuration must be a mapping."
            )

        if not isinstance(policy_configuration, Mapping):
            raise NOIPipelineError(
                "policy_configuration must be a mapping."
            )

        if not isinstance(protocol_hash, str) or not protocol_hash.strip():
            raise NOIPipelineError(
                "protocol_hash must not be empty."
            )

        self._ridge_alpha = self._validate_nonnegative_number(
            ridge_alpha,
            "ridge_alpha",
        )
        self._memory_decay_rate_per_day = (
            self._validate_nonnegative_number(
                memory_decay_rate_per_day,
                "memory_decay_rate_per_day",
            )
        )

        hybrid = system_configuration.get("hybrid_retrieval")

        if not isinstance(hybrid, Mapping):
            raise NOIPipelineError(
                "hybrid_retrieval configuration is required."
            )

        if hybrid.get("negative_cosine_handling") != "clip_to_zero":
            raise NOIPipelineError(
                "Negative cosine handling must be clip_to_zero."
            )

        if hybrid.get("absent_memory_item_score") != 0.0:
            raise NOIPipelineError(
                "Absent memory item score must be 0.0."
            )

        if hybrid.get("memory_aggregation_per_odor") != "maximum":
            raise NOIPipelineError(
                "Memory aggregation must be maximum."
            )

        alpha_selection = hybrid.get("alpha_selection")

        if not isinstance(alpha_selection, Mapping):
            raise NOIPipelineError(
                "alpha_selection configuration is required."
            )

        raw_candidates = alpha_selection.get(
            "candidates",
            DEFAULT_ALPHA_CANDIDATES,
        )

        if (
            not isinstance(raw_candidates, Sequence)
            or isinstance(raw_candidates, (str, bytes))
        ):
            raise NOIPipelineError(
                "Alpha candidates must be a sequence."
            )

        alpha_candidates = tuple(
            self._validate_alpha(value)
            for value in raw_candidates
        )

        if alpha_candidates != DEFAULT_ALPHA_CANDIDATES:
            raise NOIPipelineError(
                "Alpha candidates differ from the locked definition."
            )

        self._system_configuration = dict(system_configuration)
        self._protocol_hash = protocol_hash
        self._alpha_candidates = alpha_candidates
        self._policy_gate = DeterministicPolicyGate(
            dict(policy_configuration),
            protocol_hash,
        )

        self._dataset: SyntheticDataset | None = None
        self._library: tuple[OdorLibraryItem, ...] = ()
        self._library_vectors: dict[str, FloatArray] = {}
        self._ridge: Ridge | None = None
        self._memory: TemporalAssociativeMemory | None = None
        self._corrective_updater: CorrectiveMemoryUpdater | None = None
        self._trained_at_utc: datetime | None = None
        self._selected_alpha: float | None = None
        self._training_event_ids: frozenset[str] = frozenset()
        self._validation_event_ids: frozenset[str] = frozenset()

    @property
    def is_fitted(self) -> bool:
        """Return whether training and validation selection are complete."""

        return self._ridge is not None

    @property
    def selected_alpha(self) -> float:
        """Return the validation-selected hybrid weight."""

        self._require_fitted()

        assert self._selected_alpha is not None
        return self._selected_alpha

    @property
    def training_event_ids(self) -> frozenset[str]:
        """Return event identifiers used to fit ridge and memory."""

        return self._training_event_ids

    @property
    def validation_event_ids(self) -> frozenset[str]:
        """Return event identifiers used only for alpha selection."""

        return self._validation_event_ids

    @property
    def protocol_hash(self) -> str:
        """Return the locked protocol hash."""

        return self._protocol_hash

    def fit(
        self,
        dataset: SyntheticDataset,
        *,
        trained_at_utc: datetime,
    ) -> "NOIPipeline":
        """Fit ridge and memory on train, then select alpha on validation."""

        if not isinstance(dataset, SyntheticDataset):
            raise NOIPipelineError(
                "dataset must be a SyntheticDataset."
            )

        self._validate_timezone_aware(
            trained_at_utc,
            "trained_at_utc",
        )

        training_events = tuple(
            sorted(
                (
                    event for event in dataset.events
                    if event.split is SplitLabel.TRAIN
                ),
                key=lambda event: event.event_id,
            )
        )
        validation_events = tuple(
            sorted(
                (
                    event for event in dataset.events
                    if event.split is SplitLabel.VALIDATION
                ),
                key=lambda event: event.event_id,
            )
        )

        if not training_events:
            raise NOIPipelineError(
                "At least one training event is required."
            )

        if not validation_events:
            raise NOIPipelineError(
                "At least one validation event is required."
            )

        target_map = {
            target.item_id: np.asarray(
                target.odor_vector,
                dtype=np.float64,
            )
            for target in dataset.odor_targets
        }

        training_features = np.stack(
            [mean_fuse_event(event) for event in training_events]
        )
        training_targets = np.stack(
            [
                target_map[event.target_item_id]
                for event in training_events
            ]
        )

        if training_features.ndim != 2:
            raise NOIPipelineError(
                "Training features must be a matrix."
            )

        dimension = int(training_features.shape[1])

        ridge = Ridge(
            alpha=self._ridge_alpha,
            fit_intercept=True,
        )
        ridge.fit(
            training_features,
            training_targets,
        )

        memory = TemporalAssociativeMemory(
            dimension=dimension,
            decay_rate_per_day=self._memory_decay_rate_per_day,
        )

        for event, feature in zip(
            training_events,
            training_features,
            strict=True,
        ):
            memory.add(
                AssociativeMemoryRecord(
                    memory_id=f"memory::{event.event_id}",
                    context_vector=tuple(
                        float(value) for value in feature
                    ),
                    odor_item_id=event.target_item_id,
                    created_at_utc=trained_at_utc,
                    updated_at_utc=trained_at_utc,
                    strength=1.0,
                    correction_count=0,
                    active=True,
                )
            )

        library = tuple(build_odor_library(dataset))

        if not library:
            raise NOIPipelineError(
                "The odor library must not be empty."
            )

        library_vectors = {
            item.item_id: self._normalize(
                np.asarray(
                    item.odor_vector,
                    dtype=np.float64,
                ),
                f"odor library item {item.item_id}",
            )
            for item in library
        }

        self._dataset = dataset
        self._library = library
        self._library_vectors = library_vectors
        self._ridge = ridge
        self._memory = memory
        self._corrective_updater = CorrectiveMemoryUpdater(
            memory=memory,
            protocol_hash=self._protocol_hash,
        )
        self._trained_at_utc = trained_at_utc
        self._training_event_ids = frozenset(
            event.event_id for event in training_events
        )
        self._validation_event_ids = frozenset(
            event.event_id for event in validation_events
        )

        self._selected_alpha = self._select_alpha(
            validation_events,
            as_of_utc=trained_at_utc,
        )

        return self

    def retrieve(
        self,
        context: MultimodalContext,
        *,
        top_k: int = DEFAULT_TOP_K,
        alpha: float | None = None,
        apply_temporal_decay: bool = True,
    ) -> NOIRetrievalResult:
        """Retrieve odor candidates without invoking the policy gate."""

        self._require_fitted()

        if not isinstance(context, MultimodalContext):
            raise NOIPipelineError(
                "context must be a MultimodalContext."
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise NOIPipelineError(
                "top_k must be a positive integer."
            )

        if not isinstance(apply_temporal_decay, bool):
            raise NOIPipelineError(
                "apply_temporal_decay must be boolean."
            )

        selected_alpha = (
            self.selected_alpha
            if alpha is None
            else self._validate_alpha(alpha)
        )

        candidates = self._retrieve_from_vector(
            query_vector=mean_fuse_context(context),
            as_of_utc=context.timestamp_utc,
            top_k=top_k,
            alpha=selected_alpha,
            apply_temporal_decay=apply_temporal_decay,
        )

        return NOIRetrievalResult(
            event_id=context.event_id,
            timestamp_utc=context.timestamp_utc,
            selected_alpha=selected_alpha,
            candidates=candidates,
            temporal_decay_applied=apply_temporal_decay,
            protocol_hash=self._protocol_hash,
            oracle_used=False,
        )

    def correct_memory(
        self,
        *,
        correction_id: str,
        memory_id: str,
        corrected_at_utc: datetime,
        reason: str,
        corrected_odor_item_id: str | None = None,
        corrected_context_vector: Sequence[float] | None = None,
        corrected_strength: float | None = None,
    ) -> CorrectionAuditRecord:
        """Apply an auditable correction to one stored association."""

        self._require_fitted()

        if (
            corrected_odor_item_id is not None
            and corrected_odor_item_id not in self._library_vectors
        ):
            raise NOIPipelineError(
                "corrected_odor_item_id is not in the odor library."
            )

        assert self._corrective_updater is not None

        return self._corrective_updater.apply(
            correction_id=correction_id,
            memory_id=memory_id,
            corrected_at_utc=corrected_at_utc,
            reason=reason,
            corrected_odor_item_id=corrected_odor_item_id,
            corrected_context_vector=corrected_context_vector,
            corrected_strength=corrected_strength,
        )

    def evaluate_output_request(
        self,
        request: OutputRequest,
    ) -> PolicyDecision:
        """Evaluate a separate simulated request using the policy gate.

        This method never changes retrieval rankings and never performs
        physical emission.
        """

        if not isinstance(request, OutputRequest):
            raise NOIPipelineError(
                "request must be an OutputRequest."
            )

        return self._policy_gate.evaluate(request)

    def _select_alpha(
        self,
        validation_events: Sequence[SyntheticEvent],
        *,
        as_of_utc: datetime,
    ) -> float:
        """Select alpha using validation MRR only; ties choose larger alpha."""

        results: list[tuple[float, float]] = []

        for alpha in self._alpha_candidates:
            rankings = []
            relevant_items = []

            for event in validation_events:
                candidates = self._retrieve_from_vector(
                    query_vector=mean_fuse_event(event),
                    as_of_utc=as_of_utc,
                    top_k=DEFAULT_TOP_K,
                    alpha=alpha,
                    apply_temporal_decay=True,
                )
                rankings.append(
                    tuple(
                        candidate.item_id
                        for candidate in candidates
                    )
                )
                relevant_items.append(
                    frozenset((event.target_item_id,))
                )

            score = mean_reciprocal_rank(
                rankings,
                relevant_items,
            )
            results.append((float(score), alpha))

        return max(
            results,
            key=lambda pair: (
                pair[0],
                pair[1],
            ),
        )[1]

    def _retrieve_from_vector(
        self,
        *,
        query_vector: FloatArray,
        as_of_utc: datetime,
        top_k: int,
        alpha: float,
        apply_temporal_decay: bool,
    ) -> tuple[HybridRetrievalCandidate, ...]:
        """Compute deterministic library-memory hybrid rankings."""

        if (
            self._ridge is None
            or self._memory is None
            or self._corrective_updater is None
            or not self._library_vectors
        ):
            raise NOIPipelineError(
                "Retrieval components must be initialized before use."
            )

        self._validate_timezone_aware(
            as_of_utc,
            "as_of_utc",
        )

        assert self._ridge is not None
        assert self._memory is not None

        query = self._normalize(
            np.asarray(
                query_vector,
                dtype=np.float64,
            ),
            "query vector",
        )

        prediction = self._ridge.predict(
            query.reshape(1, -1)
        )[0]
        normalized_prediction = self._normalize(
            np.asarray(
                prediction,
                dtype=np.float64,
            ),
            "ridge prediction",
        )

        library_scores = {
            item_id: float(
                np.clip(
                    np.dot(
                        normalized_prediction,
                        odor_vector,
                    ),
                    0.0,
                    1.0,
                )
            )
            for item_id, odor_vector in self._library_vectors.items()
        }

        memory_candidates = self._memory.retrieve(
            query,
            as_of_utc=as_of_utc,
            top_k=max(1, len(self._training_event_ids)),
            apply_temporal_decay=apply_temporal_decay,
        )

        memory_scores = {
            item_id: 0.0
            for item_id in self._library_vectors
        }

        for candidate in memory_candidates:
            current = memory_scores.get(
                candidate.odor_item_id,
                0.0,
            )
            memory_scores[candidate.odor_item_id] = max(
                current,
                float(
                    np.clip(
                        candidate.score,
                        0.0,
                        1.0,
                    )
                ),
            )

        scored = []

        for item_id in self._library_vectors:
            library_score = library_scores[item_id]
            memory_score = memory_scores[item_id]
            hybrid_score = (
                alpha * library_score
                + (1.0 - alpha) * memory_score
            )

            scored.append(
                (
                    item_id,
                    float(
                        np.clip(
                            hybrid_score,
                            0.0,
                            1.0,
                        )
                    ),
                    library_score,
                    memory_score,
                )
            )

        scored.sort(
            key=lambda row: (
                -row[1],
                row[0],
            )
        )

        selected = scored[
            : min(top_k, len(scored))
        ]

        return tuple(
            HybridRetrievalCandidate(
                item_id=item_id,
                hybrid_score=hybrid_score,
                library_score=library_score,
                memory_score=memory_score,
                rank=rank,
            )
            for rank, (
                item_id,
                hybrid_score,
                library_score,
                memory_score,
            ) in enumerate(
                selected,
                start=1,
            )
        )

    def _require_fitted(self) -> None:
        if (
            self._ridge is None
            or self._memory is None
            or self._corrective_updater is None
            or self._selected_alpha is None
        ):
            raise NOIPipelineError(
                "The NOI pipeline must be fitted before use."
            )

    @staticmethod
    def _normalize(
        vector: FloatArray,
        label: str,
    ) -> FloatArray:
        if vector.ndim != 1:
            raise NOIPipelineError(
                f"{label} must be one-dimensional."
            )

        if not np.all(np.isfinite(vector)):
            raise NOIPipelineError(
                f"{label} must contain only finite values."
            )

        norm = float(np.linalg.norm(vector))

        if (
            not np.isfinite(norm)
            or norm <= np.finfo(np.float64).eps
        ):
            raise NOIPipelineError(
                f"{label} must have a nonzero finite norm."
            )

        return vector / norm

    @staticmethod
    def _validate_nonnegative_number(
        value: float,
        label: str,
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) < 0.0
        ):
            raise NOIPipelineError(
                f"{label} must be finite and nonnegative."
            )

        return float(value)

    @staticmethod
    def _validate_alpha(value: float) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise NOIPipelineError(
                "alpha must be finite and in [0, 1]."
            )

        return float(value)

    @staticmethod
    def _validate_timezone_aware(
        value: datetime,
        label: str,
    ) -> None:
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise NOIPipelineError(
                f"{label} must be timezone-aware."
            )
