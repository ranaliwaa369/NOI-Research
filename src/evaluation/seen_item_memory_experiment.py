"""Seen-item episodic-memory evaluation for NOI v0.2.

This module evaluates retrieval only on validation events whose
target items are represented in training memory. OOD evaluation
remains separate and unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite

from src.evaluation.memory_reachability import (
    audit_memory_reachability,
)
from src.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
)
from src.models import MultimodalContext
from src.system.noi_pipeline import NOIPipeline


TOP_K = 10


class SeenItemMemoryExperimentError(ValueError):
    """Raised when seen-item evaluation cannot be completed."""


class SeenItemSystem(str, Enum):
    """Prespecified systems for seen-item evaluation."""

    MEMORY_ONLY = "memory_only"
    RIDGE_ONLY = "ridge_only"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class SeenItemEvaluation:
    """Metrics and rankings for one retrieval system."""

    system: SeenItemSystem
    alpha: float
    event_ids: tuple[str, ...]
    rankings: tuple[tuple[str, ...], ...]
    relevant_items: tuple[frozenset[str], ...]
    recall_at_1: float
    recall_at_10: float
    mean_reciprocal_rank: float
    ndcg_at_10: float

    def __post_init__(self) -> None:
        if not isinstance(self.system, SeenItemSystem):
            raise SeenItemMemoryExperimentError(
                "system must be a SeenItemSystem."
            )

        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not isfinite(float(self.alpha))
            or not 0.0 <= float(self.alpha) <= 1.0
        ):
            raise SeenItemMemoryExperimentError(
                "alpha must be finite and in [0, 1]."
            )

        event_count = len(self.event_ids)

        if event_count < 1:
            raise SeenItemMemoryExperimentError(
                "At least one reachable event is required."
            )

        if len(set(self.event_ids)) != event_count:
            raise SeenItemMemoryExperimentError(
                "Event identifiers must be unique."
            )

        if (
            len(self.rankings) != event_count
            or len(self.relevant_items) != event_count
        ):
            raise SeenItemMemoryExperimentError(
                "Evaluation arrays must have equal lengths."
            )

        for ranking in self.rankings:
            if not ranking or len(ranking) > TOP_K:
                raise SeenItemMemoryExperimentError(
                    "Every ranking must respect TOP_K."
                )

            if len(ranking) != len(set(ranking)):
                raise SeenItemMemoryExperimentError(
                    "Rankings cannot contain duplicate items."
                )

        for relevant in self.relevant_items:
            if not relevant:
                raise SeenItemMemoryExperimentError(
                    "Every event requires a relevant item."
                )

        for name, value in (
            ("recall_at_1", self.recall_at_1),
            ("recall_at_10", self.recall_at_10),
            (
                "mean_reciprocal_rank",
                self.mean_reciprocal_rank,
            ),
            ("ndcg_at_10", self.ndcg_at_10),
        ):
            if (
                not isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise SeenItemMemoryExperimentError(
                    f"{name} must be finite and in [0, 1]."
                )


@dataclass(frozen=True, slots=True)
class SeenItemMemoryExperiment:
    """Complete reachability-stratified validation experiment."""

    total_validation_event_count: int
    reachable_validation_event_count: int
    reachable_event_fraction: float
    reachable_event_ids: tuple[str, ...]
    evaluations: tuple[SeenItemEvaluation, ...]

    def __post_init__(self) -> None:
        if self.total_validation_event_count < 1:
            raise SeenItemMemoryExperimentError(
                "Validation must contain at least one event."
            )

        if (
            self.reachable_validation_event_count < 1
            or self.reachable_validation_event_count
            > self.total_validation_event_count
        ):
            raise SeenItemMemoryExperimentError(
                "Reachable event count is invalid."
            )

        expected_fraction = (
            self.reachable_validation_event_count
            / self.total_validation_event_count
        )

        if self.reachable_event_fraction != expected_fraction:
            raise SeenItemMemoryExperimentError(
                "Reachable event fraction is inconsistent."
            )

        if (
            len(self.reachable_event_ids)
            != self.reachable_validation_event_count
            or len(set(self.reachable_event_ids))
            != len(self.reachable_event_ids)
        ):
            raise SeenItemMemoryExperimentError(
                "Reachable event identifiers are inconsistent."
            )

        systems = tuple(
            evaluation.system
            for evaluation in self.evaluations
        )

        if set(systems) != set(SeenItemSystem):
            raise SeenItemMemoryExperimentError(
                "Every prespecified system must be evaluated."
            )

        if len(systems) != len(set(systems)):
            raise SeenItemMemoryExperimentError(
                "Every system may have only one evaluation."
            )

        if any(
            evaluation.event_ids != self.reachable_event_ids
            for evaluation in self.evaluations
        ):
            raise SeenItemMemoryExperimentError(
                "Systems must use identical reachable events."
            )

    def for_system(
        self,
        system: SeenItemSystem,
    ) -> SeenItemEvaluation:
        """Return the evaluation for one exact system."""

        if not isinstance(system, SeenItemSystem):
            raise SeenItemMemoryExperimentError(
                "system must be a SeenItemSystem."
            )

        for evaluation in self.evaluations:
            if evaluation.system is system:
                return evaluation

        raise SeenItemMemoryExperimentError(
            f"No evaluation exists for {system.value}."
        )


def evaluate_seen_item_memory(
    pipeline: NOIPipeline,
    dataset: SyntheticDataset,
    *,
    evaluated_at_utc: datetime,
) -> SeenItemMemoryExperiment:
    """Evaluate three systems on memory-reachable validation events."""

    if not isinstance(pipeline, NOIPipeline):
        raise SeenItemMemoryExperimentError(
            "pipeline must be a NOIPipeline."
        )

    if not isinstance(dataset, SyntheticDataset):
        raise SeenItemMemoryExperimentError(
            "dataset must be a SyntheticDataset."
        )

    if (
        not isinstance(evaluated_at_utc, datetime)
        or evaluated_at_utc.tzinfo is None
        or evaluated_at_utc.utcoffset() is None
    ):
        raise SeenItemMemoryExperimentError(
            "evaluated_at_utc must be timezone-aware."
        )

    if not pipeline.is_fitted:
        raise SeenItemMemoryExperimentError(
            "pipeline must be fitted before evaluation."
        )

    validation_events = tuple(
        sorted(
            (
                event
                for event in dataset.events
                if event.split is SplitLabel.VALIDATION
            ),
            key=lambda event: event.event_id,
        )
    )

    if not validation_events:
        raise SeenItemMemoryExperimentError(
            "Dataset has no validation events."
        )

    reachability = audit_memory_reachability(dataset)
    validation_summary = reachability.for_split(
        SplitLabel.VALIDATION
    )

    reachable_ids = set(
        validation_summary.reachable_event_ids
    )

    reachable_events = tuple(
        event
        for event in validation_events
        if event.event_id in reachable_ids
    )

    if not reachable_events:
        raise SeenItemMemoryExperimentError(
            "Validation has no memory-reachable events."
        )

    reachable_event_ids = tuple(
        event.event_id
        for event in reachable_events
    )

    system_alphas = (
        (SeenItemSystem.MEMORY_ONLY, 0.0),
        (SeenItemSystem.RIDGE_ONLY, 1.0),
        (
            SeenItemSystem.HYBRID,
            float(pipeline.selected_alpha),
        ),
    )

    evaluations = []

    for system, alpha in system_alphas:
        rankings = []
        relevant_items = []

        for event in reachable_events:
            context = MultimodalContext(
                event_id=event.event_id,
                timestamp_utc=evaluated_at_utc,
                text_vector=event.text_vector,
                image_vector=event.image_vector,
                audio_vector=event.audio_vector,
                metadata={},
            )

            result = pipeline.retrieve(
                context,
                top_k=TOP_K,
                alpha=alpha,
                apply_temporal_decay=True,
            )

            rankings.append(
                tuple(
                    candidate.item_id
                    for candidate in result.candidates
                )
            )
            relevant_items.append(
                frozenset((event.target_item_id,))
            )

        ranking_tuple = tuple(rankings)
        relevant_tuple = tuple(relevant_items)

        evaluations.append(
            SeenItemEvaluation(
                system=system,
                alpha=alpha,
                event_ids=reachable_event_ids,
                rankings=ranking_tuple,
                relevant_items=relevant_tuple,
                recall_at_1=recall_at_k(
                    ranking_tuple,
                    relevant_tuple,
                    k=1,
                ),
                recall_at_10=recall_at_k(
                    ranking_tuple,
                    relevant_tuple,
                    k=10,
                ),
                mean_reciprocal_rank=(
                    mean_reciprocal_rank(
                        ranking_tuple,
                        relevant_tuple,
                    )
                ),
                ndcg_at_10=ndcg_at_k(
                    ranking_tuple,
                    relevant_tuple,
                    k=10,
                ),
            )
        )

    return SeenItemMemoryExperiment(
        total_validation_event_count=len(
            validation_events
        ),
        reachable_validation_event_count=len(
            reachable_events
        ),
        reachable_event_fraction=(
            len(reachable_events)
            / len(validation_events)
        ),
        reachable_event_ids=reachable_event_ids,
        evaluations=tuple(evaluations),
    )
