"""One prespecified unseen-family Track B experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any

from src.evaluation.graded_ood import (
    GradedOODEvent,
    OODTier,
)
from src.evaluation.graded_ood_experiment import (
    GradedOODExperiment,
    run_graded_ood_experiment,
)
from src.evaluation.memory_support_calibration import (
    MemorySupportCalibration,
    MemorySupportDecision,
    calibrate_memory_support,
)
from src.evaluation.paired_ood_source_generator import (
    PairedGradedOODBundle,
)
from src.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticEvent,
)
from src.evaluation.track_b_config import (
    TrackBConfiguration,
    TrackBRun,
)
from src.models import MultimodalContext
from src.system.noi_pipeline import NOIPipeline


TOP_K = 10
Ranking = tuple[str, ...]
RelevantItems = frozenset[str]


class TrackBSeedExperimentError(ValueError):
    """Raised when one Track B seed cannot run safely."""


@dataclass(frozen=True)
class TrackBRetrievalEvaluation:
    """Retrieval results for one NOI system and OOD tier."""

    system: str
    tier: OODTier
    alpha: float
    latent_event_ids: tuple[str, ...]
    rankings: tuple[Ranking, ...]
    relevant_items: tuple[RelevantItems, ...]
    recall_at_1: float
    recall_at_10: float
    mean_reciprocal_rank: float
    ndcg_at_10: float

    def __post_init__(self) -> None:
        if self.system not in {
            "full_noi_system",
            "memory_only_diagnostic",
        }:
            raise TrackBSeedExperimentError(
                "Unknown Track B retrieval system."
            )

        if not isinstance(self.tier, OODTier):
            raise TrackBSeedExperimentError(
                "tier must be an OODTier."
            )

        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not isfinite(float(self.alpha))
            or not 0.0 <= float(self.alpha) <= 1.0
        ):
            raise TrackBSeedExperimentError(
                "alpha must be finite and in [0, 1]."
            )

        count = len(self.latent_event_ids)

        if count < 1:
            raise TrackBSeedExperimentError(
                "At least one latent event is required."
            )

        if len(set(self.latent_event_ids)) != count:
            raise TrackBSeedExperimentError(
                "latent_event_ids must be unique."
            )

        if len(self.rankings) != count:
            raise TrackBSeedExperimentError(
                "Every event must have one ranking."
            )

        if len(self.relevant_items) != count:
            raise TrackBSeedExperimentError(
                "Every event must have relevance truth."
            )

        if any(
            len(ranking) != TOP_K
            or len(set(ranking)) != TOP_K
            for ranking in self.rankings
        ):
            raise TrackBSeedExperimentError(
                "Every ranking must contain ten unique items."
            )

        if any(
            len(relevant) != 1
            for relevant in self.relevant_items
        ):
            raise TrackBSeedExperimentError(
                "Every event must have one relevant item."
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
            _validate_metric(value, name)

        if self.recall_at_1 > self.recall_at_10:
            raise TrackBSeedExperimentError(
                "Recall@1 cannot exceed Recall@10."
            )

    @property
    def event_count(self) -> int:
        return len(self.latent_event_ids)


@dataclass(frozen=True)
class TrackBSelectiveEvaluation:
    """Abstention metrics for one unseen-family tier."""

    tier: OODTier
    latent_event_ids: tuple[str, ...]
    decisions: tuple[
        MemorySupportDecision,
        ...,
    ]
    event_count: int
    supported_event_count: int
    abstained_event_count: int
    reachable_event_fraction: float
    coverage: float
    abstention_rate: float
    false_support_rate: float
    selective_error_rate: float | None
    abstention_criterion_met: bool

    def __post_init__(self) -> None:
        if not isinstance(self.tier, OODTier):
            raise TrackBSeedExperimentError(
                "tier must be an OODTier."
            )

        if (
            isinstance(self.event_count, bool)
            or not isinstance(self.event_count, int)
            or self.event_count < 1
        ):
            raise TrackBSeedExperimentError(
                "event_count must be positive."
            )

        if len(self.latent_event_ids) != self.event_count:
            raise TrackBSeedExperimentError(
                "Latent-event count is inconsistent."
            )

        if len(self.decisions) != self.event_count:
            raise TrackBSeedExperimentError(
                "Every event must have one support decision."
            )

        if (
            self.supported_event_count
            + self.abstained_event_count
            != self.event_count
        ):
            raise TrackBSeedExperimentError(
                "Selective counts are inconsistent."
            )

        for name, value in (
            (
                "reachable_event_fraction",
                self.reachable_event_fraction,
            ),
            ("coverage", self.coverage),
            (
                "abstention_rate",
                self.abstention_rate,
            ),
            (
                "false_support_rate",
                self.false_support_rate,
            ),
        ):
            _validate_metric(value, name)

        if not _close(
            self.coverage + self.abstention_rate,
            1.0,
        ):
            raise TrackBSeedExperimentError(
                "Coverage and abstention must sum to one."
            )

        if not _close(
            self.false_support_rate,
            self.coverage,
        ):
            raise TrackBSeedExperimentError(
                "All unseen-family support is false support."
            )

        if self.selective_error_rate is None:
            if self.supported_event_count != 0:
                raise TrackBSeedExperimentError(
                    "Selective error may be absent only "
                    "when coverage is zero."
                )
        else:
            _validate_metric(
                self.selective_error_rate,
                "selective_error_rate",
            )

        if not isinstance(
            self.abstention_criterion_met,
            bool,
        ):
            raise TrackBSeedExperimentError(
                "abstention_criterion_met must be boolean."
            )


@dataclass(frozen=True)
class TrackBSeedExperiment:
    """Complete result for one prespecified Track B seed."""

    run_id: str
    generator_seed: int
    ood_seed: int
    training_event_count: int
    validation_event_count: int
    latent_event_count: int
    observed_event_count: int
    reachable_target_fraction: float
    reachable_event_fraction: float
    strict_family_separation_verified: bool
    all_ood_targets_unreachable: bool
    calibration: MemorySupportCalibration
    graded_baselines: GradedOODExperiment
    full_noi_evaluations: tuple[
        TrackBRetrievalEvaluation,
        ...,
    ]
    memory_only_evaluations: tuple[
        TrackBRetrievalEvaluation,
        ...,
    ]
    selective_evaluations: tuple[
        TrackBSelectiveEvaluation,
        ...,
    ]
    selected_hybrid_alpha: float
    oracle_used: bool
    final_test_tuning_used: bool
    target_identifier_used_in_support: bool
    family_identifier_used_in_support: bool
    protocol_hash: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise TrackBSeedExperimentError(
                "run_id must be nonempty."
            )

        for name, value in (
            ("generator_seed", self.generator_seed),
            ("ood_seed", self.ood_seed),
            (
                "training_event_count",
                self.training_event_count,
            ),
            (
                "validation_event_count",
                self.validation_event_count,
            ),
            (
                "latent_event_count",
                self.latent_event_count,
            ),
            (
                "observed_event_count",
                self.observed_event_count,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise TrackBSeedExperimentError(
                    f"{name} must be a positive integer."
                )

        if self.generator_seed == self.ood_seed:
            raise TrackBSeedExperimentError(
                "Generator and OOD seed must differ."
            )

        _validate_metric(
            self.reachable_target_fraction,
            "reachable_target_fraction",
        )
        _validate_metric(
            self.reachable_event_fraction,
            "reachable_event_fraction",
        )

        if (
            self.strict_family_separation_verified
            is not True
        ):
            raise TrackBSeedExperimentError(
                "Strict family separation must pass."
            )

        if self.all_ood_targets_unreachable is not True:
            raise TrackBSeedExperimentError(
                "Every OOD target must be unreachable."
            )

        if (
            self.reachable_target_fraction != 0.0
            or self.reachable_event_fraction != 0.0
        ):
            raise TrackBSeedExperimentError(
                "Track B reachability must equal zero."
            )

        if (
            self.graded_baselines.latent_event_count
            != self.latent_event_count
        ):
            raise TrackBSeedExperimentError(
                "Baseline latent count is inconsistent."
            )

        for evaluations, system in (
            (
                self.full_noi_evaluations,
                "full_noi_system",
            ),
            (
                self.memory_only_evaluations,
                "memory_only_diagnostic",
            ),
        ):
            if len(evaluations) != len(OODTier):
                raise TrackBSeedExperimentError(
                    f"{system} must cover every tier."
                )

            if {
                evaluation.tier
                for evaluation in evaluations
            } != set(OODTier):
                raise TrackBSeedExperimentError(
                    f"{system} tiers are incomplete."
                )

            if any(
                evaluation.system != system
                for evaluation in evaluations
            ):
                raise TrackBSeedExperimentError(
                    f"{system} labels are inconsistent."
                )

        if (
            len(self.selective_evaluations)
            != len(OODTier)
            or {
                evaluation.tier
                for evaluation
                in self.selective_evaluations
            }
            != set(OODTier)
        ):
            raise TrackBSeedExperimentError(
                "Selective evaluations must cover every tier."
            )

        _validate_metric(
            self.selected_hybrid_alpha,
            "selected_hybrid_alpha",
        )

        for name, value in (
            ("oracle_used", self.oracle_used),
            (
                "final_test_tuning_used",
                self.final_test_tuning_used,
            ),
            (
                "target_identifier_used_in_support",
                self.target_identifier_used_in_support,
            ),
            (
                "family_identifier_used_in_support",
                self.family_identifier_used_in_support,
            ),
        ):
            if value is not False:
                raise TrackBSeedExperimentError(
                    f"{name} must remain false."
                )

        if (
            not isinstance(self.protocol_hash, str)
            or not self.protocol_hash.strip()
        ):
            raise TrackBSeedExperimentError(
                "protocol_hash must be nonempty."
            )


def run_track_b_seed_experiment(
    *,
    bundle: PairedGradedOODBundle,
    track_b_config: TrackBConfiguration,
    run: TrackBRun,
    system_configuration: Mapping[str, Any],
    policy_configuration: Mapping[str, Any],
    protocol_hash: str,
    trained_at_utc: datetime,
) -> TrackBSeedExperiment:
    """Run baselines, NOI retrieval, and locked abstention."""

    if not isinstance(
        bundle,
        PairedGradedOODBundle,
    ):
        raise TrackBSeedExperimentError(
            "bundle must be a PairedGradedOODBundle."
        )

    if not isinstance(
        track_b_config,
        TrackBConfiguration,
    ):
        raise TrackBSeedExperimentError(
            "track_b_config has an invalid type."
        )

    if not isinstance(run, TrackBRun):
        raise TrackBSeedExperimentError(
            "run must be a TrackBRun."
        )

    if (
        bundle.generator_seed
        != run.generator_seed
        or bundle.ood_seed != run.ood_seed
    ):
        raise TrackBSeedExperimentError(
            "Bundle seed values do not match the run."
        )

    if run not in track_b_config.runs:
        raise TrackBSeedExperimentError(
            "Run is not present in the locked protocol."
        )

    if not isinstance(system_configuration, Mapping):
        raise TrackBSeedExperimentError(
            "system_configuration must be a mapping."
        )

    if not isinstance(policy_configuration, Mapping):
        raise TrackBSeedExperimentError(
            "policy_configuration must be a mapping."
        )

    if (
        not isinstance(protocol_hash, str)
        or not protocol_hash.strip()
    ):
        raise TrackBSeedExperimentError(
            "protocol_hash must be nonempty."
        )

    if protocol_hash != (
        track_b_config.configuration_sha256
    ):
        raise TrackBSeedExperimentError(
            "protocol_hash does not match Track B."
        )

    if (
        not isinstance(trained_at_utc, datetime)
        or trained_at_utc.tzinfo is None
        or trained_at_utc.utcoffset() is None
    ):
        raise TrackBSeedExperimentError(
            "trained_at_utc must be timezone-aware."
        )

    dataset = bundle.original_dataset

    training_events = tuple(
        event
        for event in dataset.events
        if event.split is SplitLabel.TRAIN
    )
    validation_events = tuple(
        event
        for event in dataset.events
        if event.split is SplitLabel.VALIDATION
    )
    original_ood_events = tuple(
        event
        for event in dataset.events
        if event.split is SplitLabel.OOD_TEST
    )

    if (
        not training_events
        or not validation_events
        or not original_ood_events
    ):
        raise TrackBSeedExperimentError(
            "Train, validation, and OOD splits are required."
        )

    training_targets = {
        event.target_item_id
        for event in training_events
    }
    ood_targets = {
        event.target_item_id
        for event in original_ood_events
    }
    training_families = {
        event.target_family_id
        for event in training_events
    }
    ood_families = {
        event.target_family_id
        for event in original_ood_events
    }

    reachable_targets = (
        training_targets & ood_targets
    )
    reachable_ood_events = tuple(
        event
        for event in original_ood_events
        if event.target_item_id in training_targets
    )

    reachable_target_fraction = (
        len(reachable_targets)
        / len(ood_targets)
    )
    reachable_event_fraction = (
        len(reachable_ood_events)
        / len(original_ood_events)
    )

    strict_family_separation = (
        training_families.isdisjoint(
            ood_families
        )
    )
    all_unreachable = (
        not reachable_targets
        and not reachable_ood_events
    )

    if not strict_family_separation:
        raise TrackBSeedExperimentError(
            "Strict odor-family separation failed."
        )

    if not all_unreachable:
        raise TrackBSeedExperimentError(
            "All Track B OOD targets must be unreachable."
        )

    calibration = calibrate_memory_support(
        training_events=training_events,
        validation_events=validation_events,
        minimum_reachable_coverage=(
            track_b_config.minimum_reachable_coverage
        ),
    )

    graded_baselines = run_graded_ood_experiment(
        bundle,
        top_k=TOP_K,
        random_seed=run.ood_seed,
        ridge_alpha=1.0,
    )

    pipeline = NOIPipeline(
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=protocol_hash,
    )
    pipeline.fit(
        dataset,
        trained_at_utc=trained_at_utc,
    )

    selected_alpha = float(
        pipeline.selected_alpha
    )

    full_evaluations: list[
        TrackBRetrievalEvaluation
    ] = []
    memory_evaluations: list[
        TrackBRetrievalEvaluation
    ] = []
    selective_evaluations: list[
        TrackBSelectiveEvaluation
    ] = []

    for tier in OODTier:
        tier_events = (
            bundle.graded_dataset.events_for_tier(
                tier
            )
        )

        full_evaluation = _evaluate_noi_tier(
            pipeline=pipeline,
            events=tier_events,
            tier=tier,
            system="full_noi_system",
            alpha=selected_alpha,
            trained_at_utc=trained_at_utc,
        )
        memory_evaluation = _evaluate_noi_tier(
            pipeline=pipeline,
            events=tier_events,
            tier=tier,
            system="memory_only_diagnostic",
            alpha=0.0,
            trained_at_utc=trained_at_utc,
        )

        decisions = calibration.score_events(
            tier_events
        )

        selective = _summarize_selective_tier(
            tier=tier,
            events=tier_events,
            decisions=decisions,
            full_evaluation=full_evaluation,
            minimum_abstention_rate=(
                track_b_config
                .minimum_ood_abstention_rate
            ),
        )

        full_evaluations.append(full_evaluation)
        memory_evaluations.append(
            memory_evaluation
        )
        selective_evaluations.append(selective)

    return TrackBSeedExperiment(
        run_id=run.run_id,
        generator_seed=run.generator_seed,
        ood_seed=run.ood_seed,
        training_event_count=len(training_events),
        validation_event_count=len(
            validation_events
        ),
        latent_event_count=(
            bundle.source_count
        ),
        observed_event_count=(
            bundle.graded_dataset
            .observed_event_count
        ),
        reachable_target_fraction=float(
            reachable_target_fraction
        ),
        reachable_event_fraction=float(
            reachable_event_fraction
        ),
        strict_family_separation_verified=(
            strict_family_separation
        ),
        all_ood_targets_unreachable=(
            all_unreachable
        ),
        calibration=calibration,
        graded_baselines=graded_baselines,
        full_noi_evaluations=tuple(
            full_evaluations
        ),
        memory_only_evaluations=tuple(
            memory_evaluations
        ),
        selective_evaluations=tuple(
            selective_evaluations
        ),
        selected_hybrid_alpha=selected_alpha,
        oracle_used=False,
        final_test_tuning_used=False,
        target_identifier_used_in_support=False,
        family_identifier_used_in_support=False,
        protocol_hash=protocol_hash,
    )


def _evaluate_noi_tier(
    *,
    pipeline: NOIPipeline,
    events: tuple[GradedOODEvent, ...],
    tier: OODTier,
    system: str,
    alpha: float,
    trained_at_utc: datetime,
) -> TrackBRetrievalEvaluation:
    rankings: list[Ranking] = []
    relevant_items: list[RelevantItems] = []
    latent_ids: list[str] = []

    for event in events:
        context = MultimodalContext(
            event_id=event.observed_event_id,
            timestamp_utc=trained_at_utc,
            text_vector=event.text_vector,
            image_vector=event.image_vector,
            audio_vector=event.audio_vector,
            metadata={
                "evaluation_track": "track_b",
                "severity_tier": tier.value,
                "memory_reachable": False,
            },
        )

        result = pipeline.retrieve(
            context,
            top_k=TOP_K,
            alpha=alpha,
            apply_temporal_decay=False,
        )

        if result.oracle_used:
            raise TrackBSeedExperimentError(
                "NOI retrieval reported oracle use."
            )

        latent_ids.append(
            event.latent_event_id
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
    relevance_tuple = tuple(relevant_items)

    return TrackBRetrievalEvaluation(
        system=system,
        tier=tier,
        alpha=float(alpha),
        latent_event_ids=tuple(latent_ids),
        rankings=ranking_tuple,
        relevant_items=relevance_tuple,
        recall_at_1=recall_at_k(
            ranking_tuple,
            relevance_tuple,
            k=1,
        ),
        recall_at_10=recall_at_k(
            ranking_tuple,
            relevance_tuple,
            k=10,
        ),
        mean_reciprocal_rank=(
            mean_reciprocal_rank(
                ranking_tuple,
                relevance_tuple,
            )
        ),
        ndcg_at_10=ndcg_at_k(
            ranking_tuple,
            relevance_tuple,
            k=10,
        ),
    )


def _summarize_selective_tier(
    *,
    tier: OODTier,
    events: tuple[GradedOODEvent, ...],
    decisions: tuple[
        MemorySupportDecision,
        ...,
    ],
    full_evaluation: TrackBRetrievalEvaluation,
    minimum_abstention_rate: float,
) -> TrackBSelectiveEvaluation:
    if (
        len(events) != len(decisions)
        or len(events)
        != full_evaluation.event_count
    ):
        raise TrackBSeedExperimentError(
            "Selective tier inputs must align."
        )

    supported_indices = tuple(
        index
        for index, decision
        in enumerate(decisions)
        if decision.supported
    )

    supported_count = len(
        supported_indices
    )
    event_count = len(events)
    abstained_count = (
        event_count - supported_count
    )
    coverage = (
        supported_count / event_count
    )
    abstention_rate = (
        abstained_count / event_count
    )

    if supported_count == 0:
        selective_error_rate = None
    else:
        errors = sum(
            full_evaluation.rankings[index][0]
            not in full_evaluation.relevant_items[index]
            for index in supported_indices
        )
        selective_error_rate = (
            errors / supported_count
        )

    return TrackBSelectiveEvaluation(
        tier=tier,
        latent_event_ids=tuple(
            event.latent_event_id
            for event in events
        ),
        decisions=decisions,
        event_count=event_count,
        supported_event_count=supported_count,
        abstained_event_count=abstained_count,
        reachable_event_fraction=0.0,
        coverage=float(coverage),
        abstention_rate=float(
            abstention_rate
        ),
        false_support_rate=float(coverage),
        selective_error_rate=(
            None
            if selective_error_rate is None
            else float(selective_error_rate)
        ),
        abstention_criterion_met=bool(
            abstention_rate
            >= minimum_abstention_rate
        ),
    )


def _validate_metric(
    value: Any,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise TrackBSeedExperimentError(
            f"{name} must be finite and in [0, 1]."
        )


def _close(
    left: float,
    right: float,
) -> bool:
    return abs(left - right) <= 1e-12
