"""Paired controlled evaluation of corrective associative memory."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
    SyntheticEvent,
)
from src.models import MultimodalContext
from src.system.noi_pipeline import NOIPipeline


FloatArray = NDArray[np.float64]

LOCKED_TOP_K = 10
LOCKED_BOOTSTRAP_SEED = 4242
LOCKED_BOOTSTRAP_RESAMPLES = 10_000
LOCKED_CONFIDENCE_LEVEL = 0.95
LOCKED_ALPHA = 0.0
LOCKED_APPLY_TEMPORAL_DECAY = False
LOCKED_EXPECTED_ELIGIBLE_EVENTS = 15
LOCKED_EXPECTED_ELIGIBLE_TARGETS = 14
LOCKED_MAXIMUM_OLD_MEMORY_DEGRADATION = 0.02
LOCKED_MINIMUM_MRR_IMPROVEMENT = 0.05


class CorrectiveMemoryExperimentError(ValueError):
    """Raised when the locked corrective-memory experiment fails."""


@dataclass(frozen=True)
class CorrectiveTargetResult:
    """Paired correction outcome for one known target."""

    target_item_id: str
    decoy_item_id: str
    validation_event_ids: tuple[str, ...]
    corrupted_memory_ids: tuple[str, ...]
    no_update_rankings: tuple[tuple[str, ...], ...]
    corrected_rankings: tuple[tuple[str, ...], ...]
    relevant_items: tuple[frozenset[str], ...]
    no_update_mean_reciprocal_rank: float
    corrected_mean_reciprocal_rank: float
    reciprocal_rank_difference: float
    no_update_recall_at_1: float
    corrected_recall_at_1: float
    no_update_recall_at_10: float
    corrected_recall_at_10: float
    no_update_ndcg_at_10: float
    corrected_ndcg_at_10: float
    old_memory_baseline_mrr: float
    old_memory_post_correction_mrr: float
    old_memory_degradation: float
    restoration_audit_count: int

    def __post_init__(self) -> None:
        for label, value in (
            ("target_item_id", self.target_item_id),
            ("decoy_item_id", self.decoy_item_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise CorrectiveMemoryExperimentError(
                    f"{label} must not be empty."
                )

        if self.target_item_id == self.decoy_item_id:
            raise CorrectiveMemoryExperimentError(
                "The decoy must differ from the true target."
            )

        query_count = len(self.validation_event_ids)

        if query_count < 1:
            raise CorrectiveMemoryExperimentError(
                "Each target requires at least one validation query."
            )

        if (
            len(set(self.validation_event_ids)) != query_count
            or len(self.no_update_rankings) != query_count
            or len(self.corrected_rankings) != query_count
            or len(self.relevant_items) != query_count
        ):
            raise CorrectiveMemoryExperimentError(
                "Target-level paired query arrays must align."
            )

        if not self.corrupted_memory_ids:
            raise CorrectiveMemoryExperimentError(
                "Each target requires at least one corrupted memory."
            )

        if (
            len(set(self.corrupted_memory_ids))
            != len(self.corrupted_memory_ids)
        ):
            raise CorrectiveMemoryExperimentError(
                "Corrupted memory identifiers must be unique."
            )

        if (
            self.restoration_audit_count
            != len(self.corrupted_memory_ids)
        ):
            raise CorrectiveMemoryExperimentError(
                "Every corrupted memory requires a restoration audit."
            )

        for label, value in (
            (
                "no_update_mean_reciprocal_rank",
                self.no_update_mean_reciprocal_rank,
            ),
            (
                "corrected_mean_reciprocal_rank",
                self.corrected_mean_reciprocal_rank,
            ),
            (
                "no_update_recall_at_1",
                self.no_update_recall_at_1,
            ),
            (
                "corrected_recall_at_1",
                self.corrected_recall_at_1,
            ),
            (
                "no_update_recall_at_10",
                self.no_update_recall_at_10,
            ),
            (
                "corrected_recall_at_10",
                self.corrected_recall_at_10,
            ),
            (
                "no_update_ndcg_at_10",
                self.no_update_ndcg_at_10,
            ),
            (
                "corrected_ndcg_at_10",
                self.corrected_ndcg_at_10,
            ),
            (
                "old_memory_baseline_mrr",
                self.old_memory_baseline_mrr,
            ),
            (
                "old_memory_post_correction_mrr",
                self.old_memory_post_correction_mrr,
            ),
        ):
            if (
                not isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise CorrectiveMemoryExperimentError(
                    f"{label} must be finite and in [0, 1]."
                )

        for label, value in (
            (
                "reciprocal_rank_difference",
                self.reciprocal_rank_difference,
            ),
            (
                "old_memory_degradation",
                self.old_memory_degradation,
            ),
        ):
            if (
                not isfinite(float(value))
                or not -1.0 <= float(value) <= 1.0
            ):
                raise CorrectiveMemoryExperimentError(
                    f"{label} must be finite and in [-1, 1]."
                )


@dataclass(frozen=True)
class CorrectiveMemoryExperiment:
    """Complete paired H2 corrective-memory experiment."""

    target_results: tuple[CorrectiveTargetResult, ...]
    eligible_target_ids: tuple[str, ...]
    eligible_validation_event_ids: tuple[str, ...]
    excluded_validation_target_ids: tuple[str, ...]
    training_event_count: int
    validation_event_count: int
    eligible_target_count: int
    eligible_validation_event_count: int
    alpha: float
    apply_temporal_decay: bool
    top_k: int
    mean_mrr_improvement: float
    standard_deviation_mrr_improvement: float
    minimum_mrr_improvement: float
    maximum_mrr_improvement: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    bootstrap_seed: int
    bootstrap_resamples: int
    confidence_level: float
    mean_old_memory_degradation: float
    maximum_old_memory_degradation: float
    correction_success_rule_passed: bool
    old_memory_degradation_rule_passed: bool
    oracle_used: bool
    ood_tuning_used: bool
    protocol_hash: str

    def __post_init__(self) -> None:
        if len(self.target_results) != LOCKED_EXPECTED_ELIGIBLE_TARGETS:
            raise CorrectiveMemoryExperimentError(
                "Target result count differs from the locked count."
            )

        if (
            self.eligible_target_count
            != LOCKED_EXPECTED_ELIGIBLE_TARGETS
            or len(self.eligible_target_ids)
            != self.eligible_target_count
        ):
            raise CorrectiveMemoryExperimentError(
                "Eligible target count is invalid."
            )

        if (
            self.eligible_validation_event_count
            != LOCKED_EXPECTED_ELIGIBLE_EVENTS
            or len(self.eligible_validation_event_ids)
            != self.eligible_validation_event_count
        ):
            raise CorrectiveMemoryExperimentError(
                "Eligible validation-event count is invalid."
            )

        if self.training_event_count != 140:
            raise CorrectiveMemoryExperimentError(
                "Training event count must remain 140."
            )

        if self.validation_event_count != 20:
            raise CorrectiveMemoryExperimentError(
                "Validation event count must remain 20."
            )

        if self.alpha != LOCKED_ALPHA:
            raise CorrectiveMemoryExperimentError(
                "Primary correction evaluation must use alpha zero."
            )

        if (
            self.apply_temporal_decay
            is not LOCKED_APPLY_TEMPORAL_DECAY
        ):
            raise CorrectiveMemoryExperimentError(
                "Temporal decay must be disabled."
            )

        if self.top_k != LOCKED_TOP_K:
            raise CorrectiveMemoryExperimentError(
                "top_k must remain 10."
            )

        if (
            self.bootstrap_seed != LOCKED_BOOTSTRAP_SEED
            or self.bootstrap_resamples
            != LOCKED_BOOTSTRAP_RESAMPLES
            or self.confidence_level
            != LOCKED_CONFIDENCE_LEVEL
        ):
            raise CorrectiveMemoryExperimentError(
                "Bootstrap settings differ from the locked values."
            )

        if self.oracle_used is not False:
            raise CorrectiveMemoryExperimentError(
                "Oracle use is prohibited."
            )

        if self.ood_tuning_used is not False:
            raise CorrectiveMemoryExperimentError(
                "OOD tuning is prohibited."
            )

        if not self.protocol_hash:
            raise CorrectiveMemoryExperimentError(
                "protocol_hash must not be empty."
            )

        target_ids = tuple(
            result.target_item_id
            for result in self.target_results
        )

        if target_ids != self.eligible_target_ids:
            raise CorrectiveMemoryExperimentError(
                "Target results must follow locked target order."
            )


def run_corrective_memory_experiment(
    dataset: SyntheticDataset,
    *,
    evaluation_configuration: Mapping[str, Any],
    system_configuration: Mapping[str, Any],
    policy_configuration: Mapping[str, Any],
    protocol_hash: str,
    trained_at_utc: datetime,
) -> CorrectiveMemoryExperiment:
    """Run the preregistered paired controlled-correction experiment."""

    if not isinstance(dataset, SyntheticDataset):
        raise CorrectiveMemoryExperimentError(
            "dataset must be a SyntheticDataset."
        )

    if (
        trained_at_utc.tzinfo is None
        or trained_at_utc.utcoffset() is None
    ):
        raise CorrectiveMemoryExperimentError(
            "trained_at_utc must be timezone-aware."
        )

    _validate_runtime_configuration(
        evaluation_configuration
    )

    training_events = tuple(
        sorted(
            (
                event
                for event in dataset.events
                if event.split is SplitLabel.TRAIN
            ),
            key=lambda event: event.event_id,
        )
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

    training_ids = {
        event.event_id for event in training_events
    }
    validation_ids = {
        event.event_id for event in validation_events
    }

    if not training_ids.isdisjoint(validation_ids):
        raise CorrectiveMemoryExperimentError(
            "Training and validation event identifiers must be disjoint."
        )

    training_targets = {
        event.target_item_id
        for event in training_events
    }

    eligible_events = tuple(
        event
        for event in validation_events
        if event.target_item_id in training_targets
    )

    excluded_target_ids = tuple(
        sorted(
            {
                event.target_item_id
                for event in validation_events
                if event.target_item_id not in training_targets
            }
        )
    )

    events_by_target: dict[str, list[SyntheticEvent]] = (
        defaultdict(list)
    )

    for event in eligible_events:
        events_by_target[event.target_item_id].append(
            event
        )

    eligible_target_ids = tuple(
        sorted(events_by_target)
    )

    if len(eligible_events) != LOCKED_EXPECTED_ELIGIBLE_EVENTS:
        raise CorrectiveMemoryExperimentError(
            "Eligible validation-event count changed."
        )

    if (
        len(eligible_target_ids)
        != LOCKED_EXPECTED_ELIGIBLE_TARGETS
    ):
        raise CorrectiveMemoryExperimentError(
            "Eligible target count changed."
        )

    memory_ids_by_target: dict[str, tuple[str, ...]] = {}

    grouped_training_ids: dict[str, list[str]] = defaultdict(list)

    for event in training_events:
        grouped_training_ids[event.target_item_id].append(
            f"memory::{event.event_id}"
        )

    for target_id, memory_ids in grouped_training_ids.items():
        memory_ids_by_target[target_id] = tuple(
            sorted(memory_ids)
        )

    decoy_pool = tuple(sorted(training_targets))

    clean_pipeline = _fit_pipeline(
        dataset=dataset,
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=protocol_hash,
        trained_at_utc=trained_at_utc,
    )

    query_time = trained_at_utc + timedelta(days=1)
    target_results: list[CorrectiveTargetResult] = []

    for target_id in eligible_target_ids:
        decoy_id = _next_decoy(
            target_id,
            decoy_pool,
        )
        corrupted_memory_ids = memory_ids_by_target[
            target_id
        ]

        no_update_pipeline = _fit_pipeline(
            dataset=dataset,
            system_configuration=system_configuration,
            policy_configuration=policy_configuration,
            protocol_hash=protocol_hash,
            trained_at_utc=trained_at_utc,
        )
        corrected_pipeline = _fit_pipeline(
            dataset=dataset,
            system_configuration=system_configuration,
            policy_configuration=policy_configuration,
            protocol_hash=protocol_hash,
            trained_at_utc=trained_at_utc,
        )

        for index, memory_id in enumerate(
            corrupted_memory_ids
        ):
            no_update_pipeline.correct_memory(
                correction_id=(
                    f"corrupt::no-update::{target_id}::{index}"
                ),
                memory_id=memory_id,
                corrected_at_utc=trained_at_utc,
                reason=(
                    "Locked synthetic H2 controlled corruption"
                ),
                corrected_odor_item_id=decoy_id,
            )
            corrected_pipeline.correct_memory(
                correction_id=(
                    f"corrupt::corrected::{target_id}::{index}"
                ),
                memory_id=memory_id,
                corrected_at_utc=trained_at_utc,
                reason=(
                    "Locked synthetic H2 controlled corruption"
                ),
                corrected_odor_item_id=decoy_id,
            )

        restoration_audits = []

        for index, memory_id in enumerate(
            corrupted_memory_ids
        ):
            audit = corrected_pipeline.correct_memory(
                correction_id=(
                    f"restore::{target_id}::{index}"
                ),
                memory_id=memory_id,
                corrected_at_utc=query_time,
                reason=(
                    "Locked synthetic H2 corrective restoration"
                ),
                corrected_odor_item_id=target_id,
            )

            if audit.protocol_hash != protocol_hash:
                raise CorrectiveMemoryExperimentError(
                    "Correction audit protocol hash mismatch."
                )

            restoration_audits.append(audit)

        target_queries = tuple(
            sorted(
                events_by_target[target_id],
                key=lambda event: event.event_id,
            )
        )

        no_update_rankings = _rank_events(
            no_update_pipeline,
            target_queries,
            query_time=query_time,
        )
        corrected_rankings = _rank_events(
            corrected_pipeline,
            target_queries,
            query_time=query_time,
        )
        relevant_items = tuple(
            frozenset((event.target_item_id,))
            for event in target_queries
        )

        no_update_mrr = mean_reciprocal_rank(
            no_update_rankings,
            relevant_items,
        )
        corrected_mrr = mean_reciprocal_rank(
            corrected_rankings,
            relevant_items,
        )

        old_queries = tuple(
            event
            for event in eligible_events
            if event.target_item_id != target_id
        )
        old_relevant = tuple(
            frozenset((event.target_item_id,))
            for event in old_queries
        )

        clean_old_rankings = _rank_events(
            clean_pipeline,
            old_queries,
            query_time=query_time,
        )
        corrected_old_rankings = _rank_events(
            corrected_pipeline,
            old_queries,
            query_time=query_time,
        )

        baseline_old_mrr = mean_reciprocal_rank(
            clean_old_rankings,
            old_relevant,
        )
        post_old_mrr = mean_reciprocal_rank(
            corrected_old_rankings,
            old_relevant,
        )

        target_results.append(
            CorrectiveTargetResult(
                target_item_id=target_id,
                decoy_item_id=decoy_id,
                validation_event_ids=tuple(
                    event.event_id
                    for event in target_queries
                ),
                corrupted_memory_ids=corrupted_memory_ids,
                no_update_rankings=no_update_rankings,
                corrected_rankings=corrected_rankings,
                relevant_items=relevant_items,
                no_update_mean_reciprocal_rank=(
                    no_update_mrr
                ),
                corrected_mean_reciprocal_rank=(
                    corrected_mrr
                ),
                reciprocal_rank_difference=(
                    corrected_mrr - no_update_mrr
                ),
                no_update_recall_at_1=recall_at_k(
                    no_update_rankings,
                    relevant_items,
                    k=1,
                ),
                corrected_recall_at_1=recall_at_k(
                    corrected_rankings,
                    relevant_items,
                    k=1,
                ),
                no_update_recall_at_10=recall_at_k(
                    no_update_rankings,
                    relevant_items,
                    k=10,
                ),
                corrected_recall_at_10=recall_at_k(
                    corrected_rankings,
                    relevant_items,
                    k=10,
                ),
                no_update_ndcg_at_10=ndcg_at_k(
                    no_update_rankings,
                    relevant_items,
                    k=10,
                ),
                corrected_ndcg_at_10=ndcg_at_k(
                    corrected_rankings,
                    relevant_items,
                    k=10,
                ),
                old_memory_baseline_mrr=(
                    baseline_old_mrr
                ),
                old_memory_post_correction_mrr=(
                    post_old_mrr
                ),
                old_memory_degradation=(
                    baseline_old_mrr - post_old_mrr
                ),
                restoration_audit_count=len(
                    restoration_audits
                ),
            )
        )

    differences = np.asarray(
        [
            result.reciprocal_rank_difference
            for result in target_results
        ],
        dtype=np.float64,
    )

    degradations = np.asarray(
        [
            result.old_memory_degradation
            for result in target_results
        ],
        dtype=np.float64,
    )

    ci_lower, ci_upper = _paired_bootstrap_interval(
        differences,
        seed=LOCKED_BOOTSTRAP_SEED,
        resamples=LOCKED_BOOTSTRAP_RESAMPLES,
        confidence_level=LOCKED_CONFIDENCE_LEVEL,
    )

    mean_improvement = float(differences.mean())
    mean_degradation = float(degradations.mean())
    maximum_degradation = float(degradations.max())

    correction_rule_passed = (
        mean_improvement
        >= LOCKED_MINIMUM_MRR_IMPROVEMENT
        and ci_lower > 0.0
    )
    degradation_rule_passed = (
        mean_degradation
        <= LOCKED_MAXIMUM_OLD_MEMORY_DEGRADATION
    )

    return CorrectiveMemoryExperiment(
        target_results=tuple(target_results),
        eligible_target_ids=eligible_target_ids,
        eligible_validation_event_ids=tuple(
            event.event_id for event in eligible_events
        ),
        excluded_validation_target_ids=(
            excluded_target_ids
        ),
        training_event_count=len(training_events),
        validation_event_count=len(validation_events),
        eligible_target_count=len(
            eligible_target_ids
        ),
        eligible_validation_event_count=len(
            eligible_events
        ),
        alpha=LOCKED_ALPHA,
        apply_temporal_decay=(
            LOCKED_APPLY_TEMPORAL_DECAY
        ),
        top_k=LOCKED_TOP_K,
        mean_mrr_improvement=mean_improvement,
        standard_deviation_mrr_improvement=float(
            differences.std(ddof=1)
        ),
        minimum_mrr_improvement=float(
            differences.min()
        ),
        maximum_mrr_improvement=float(
            differences.max()
        ),
        bootstrap_ci_lower=ci_lower,
        bootstrap_ci_upper=ci_upper,
        bootstrap_seed=LOCKED_BOOTSTRAP_SEED,
        bootstrap_resamples=(
            LOCKED_BOOTSTRAP_RESAMPLES
        ),
        confidence_level=LOCKED_CONFIDENCE_LEVEL,
        mean_old_memory_degradation=(
            mean_degradation
        ),
        maximum_old_memory_degradation=(
            maximum_degradation
        ),
        correction_success_rule_passed=(
            correction_rule_passed
        ),
        old_memory_degradation_rule_passed=(
            degradation_rule_passed
        ),
        oracle_used=False,
        ood_tuning_used=False,
        protocol_hash=protocol_hash,
    )


def _fit_pipeline(
    *,
    dataset: SyntheticDataset,
    system_configuration: Mapping[str, Any],
    policy_configuration: Mapping[str, Any],
    protocol_hash: str,
    trained_at_utc: datetime,
) -> NOIPipeline:
    pipeline = NOIPipeline(
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=protocol_hash,
    )
    pipeline.fit(
        dataset,
        trained_at_utc=trained_at_utc,
    )
    return pipeline


def _rank_events(
    pipeline: NOIPipeline,
    events: Sequence[SyntheticEvent],
    *,
    query_time: datetime,
) -> tuple[tuple[str, ...], ...]:
    rankings = []

    for event in events:
        context = MultimodalContext(
            event_id=event.event_id,
            timestamp_utc=query_time,
            text_vector=event.text_vector,
            image_vector=event.image_vector,
            audio_vector=event.audio_vector,
            metadata={},
        )
        result = pipeline.retrieve(
            context,
            top_k=LOCKED_TOP_K,
            alpha=LOCKED_ALPHA,
            apply_temporal_decay=(
                LOCKED_APPLY_TEMPORAL_DECAY
            ),
        )
        rankings.append(
            tuple(
                candidate.item_id
                for candidate in result.candidates
            )
        )

    return tuple(rankings)


def _next_decoy(
    target_id: str,
    candidate_pool: Sequence[str],
) -> str:
    if target_id not in candidate_pool:
        raise CorrectiveMemoryExperimentError(
            "Target is absent from the decoy candidate pool."
        )

    if len(candidate_pool) < 2:
        raise CorrectiveMemoryExperimentError(
            "At least two decoy candidates are required."
        )

    index = candidate_pool.index(target_id)
    decoy = candidate_pool[
        (index + 1) % len(candidate_pool)
    ]

    if decoy == target_id:
        raise CorrectiveMemoryExperimentError(
            "Decoy selection returned the true target."
        )

    return decoy


def _paired_bootstrap_interval(
    values: FloatArray,
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> tuple[float, float]:
    if values.ndim != 1 or values.size < 2:
        raise CorrectiveMemoryExperimentError(
            "Bootstrap requires at least two paired target values."
        )

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        values.size,
        size=(resamples, values.size),
    )
    bootstrap_means = values[
        sample_indices
    ].mean(axis=1)

    tail_probability = (
        1.0 - confidence_level
    ) / 2.0

    lower = float(
        np.quantile(
            bootstrap_means,
            tail_probability,
        )
    )
    upper = float(
        np.quantile(
            bootstrap_means,
            1.0 - tail_probability,
        )
    )

    return lower, upper


def _validate_runtime_configuration(
    configuration: Mapping[str, Any],
) -> None:
    if not isinstance(configuration, Mapping):
        raise CorrectiveMemoryExperimentError(
            "evaluation_configuration must be a mapping."
        )

    eligibility = configuration.get(
        "eligibility"
    )
    retrieval = configuration.get(
        "retrieval"
    )
    statistics = configuration.get(
        "statistics"
    )

    if not isinstance(eligibility, Mapping):
        raise CorrectiveMemoryExperimentError(
            "eligibility configuration is required."
        )

    if not isinstance(retrieval, Mapping):
        raise CorrectiveMemoryExperimentError(
            "retrieval configuration is required."
        )

    if not isinstance(statistics, Mapping):
        raise CorrectiveMemoryExperimentError(
            "statistics configuration is required."
        )

    if (
        eligibility.get(
            "expected_eligible_validation_events"
        )
        != LOCKED_EXPECTED_ELIGIBLE_EVENTS
    ):
        raise CorrectiveMemoryExperimentError(
            "Eligible validation-event count is not locked."
        )

    if (
        eligibility.get(
            "expected_eligible_targets"
        )
        != LOCKED_EXPECTED_ELIGIBLE_TARGETS
    ):
        raise CorrectiveMemoryExperimentError(
            "Eligible target count is not locked."
        )

    if retrieval.get("alpha") != LOCKED_ALPHA:
        raise CorrectiveMemoryExperimentError(
            "Retrieval alpha is not locked."
        )

    if (
        retrieval.get("apply_temporal_decay")
        is not LOCKED_APPLY_TEMPORAL_DECAY
    ):
        raise CorrectiveMemoryExperimentError(
            "Temporal-decay setting is not locked."
        )

    if retrieval.get("top_k") != LOCKED_TOP_K:
        raise CorrectiveMemoryExperimentError(
            "top_k is not locked."
        )

    if (
        statistics.get("bootstrap_seed")
        != LOCKED_BOOTSTRAP_SEED
        or statistics.get("bootstrap_resamples")
        != LOCKED_BOOTSTRAP_RESAMPLES
        or statistics.get("confidence_level")
        != LOCKED_CONFIDENCE_LEVEL
    ):
        raise CorrectiveMemoryExperimentError(
            "Bootstrap settings are not locked."
        )
