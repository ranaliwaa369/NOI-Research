"""Final missing-modality and temporal robustness experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from statistics import fmean
from typing import Any, Mapping

from src.evaluation.noi_ablation_experiment import (
    MultimodalContext,
    OODTier,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.paired_ood_source_generator import (
    PairedGradedOODBundle,
)
from src.evaluation.robustness_config import (
    MissingModalityCondition,
    RobustnessConfiguration,
    RobustnessRun,
)
from src.system.noi_pipeline import NOIPipeline


class FinalRobustnessExperimentError(
    ValueError
):
    """Raised when final robustness execution is invalid."""


@dataclass(frozen=True)
class FinalRobustnessEvaluation:
    """One locked robustness condition evaluation."""

    axis: str
    condition_id: str
    system: str
    tier: str
    missing_modalities: tuple[str, ...]
    temporal_displacement_days: int
    selected_alpha: float
    apply_temporal_decay: bool
    event_count: int
    latent_event_ids: tuple[str, ...]
    reciprocal_ranks: tuple[float, ...]
    recall_at_1: float
    recall_at_10: float
    mean_reciprocal_rank: float
    ndcg_at_10: float


@dataclass(frozen=True)
class FinalRobustnessExperiment:
    """All final robustness results for one seed."""

    run_id: str
    generator_seed: int
    ood_seed: int
    training_event_count: int
    validation_event_count: int
    latent_event_count: int
    observed_event_count: int
    odor_library_size: int
    selected_validation_alpha: float
    evaluations: tuple[
        FinalRobustnessEvaluation,
        ...,
    ]
    paired_analysis_unit: str
    all_ood_targets_unreachable: bool
    strict_family_separation_verified: bool
    oracle_used: bool
    ood_tuning_used: bool
    final_test_tuning_used: bool
    protocol_hash: str


@dataclass(frozen=True)
class _RawEvaluation:
    latent_event_ids: tuple[str, ...]
    reciprocal_ranks: tuple[float, ...]
    recall_at_1: float
    recall_at_10: float
    mean_reciprocal_rank: float
    ndcg_at_10: float


def run_final_robustness_experiment(
    *,
    bundle: PairedGradedOODBundle,
    configuration: RobustnessConfiguration,
    run: RobustnessRun,
    system_configuration: Mapping[str, Any],
    policy_configuration: Mapping[str, Any],
    protocol_hash: str,
    trained_at_utc: datetime,
) -> FinalRobustnessExperiment:
    """Run both locked robustness axes for one seed."""

    _validate_inputs(
        bundle=bundle,
        configuration=configuration,
        run=run,
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=protocol_hash,
        trained_at_utc=trained_at_utc,
    )

    training_events = tuple(
        event
        for event in bundle.original_dataset.events
        if event.split.value == "train"
    )
    validation_events = tuple(
        event
        for event in bundle.original_dataset.events
        if event.split.value == "validation"
    )
    ood_events = tuple(
        event
        for event in bundle.original_dataset.events
        if event.split.value == "ood_test"
    )

    training_targets = {
        event.target_item_id
        for event in training_events
    }
    training_families = {
        event.target_family_id
        for event in training_events
    }
    ood_targets = {
        event.target_item_id
        for event in ood_events
    }
    ood_families = {
        event.target_family_id
        for event in ood_events
    }

    all_targets_unreachable = (
        training_targets.isdisjoint(
            ood_targets
        )
    )
    family_separation = (
        training_families.isdisjoint(
            ood_families
        )
    )

    if not all_targets_unreachable:
        raise FinalRobustnessExperimentError(
            "All OOD targets must be unreachable."
        )

    if not family_separation:
        raise FinalRobustnessExperimentError(
            "Strict family separation failed."
        )

    pipeline = NOIPipeline(
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=protocol_hash,
    )
    pipeline.fit(
        bundle.original_dataset,
        trained_at_utc=trained_at_utc,
    )

    selected_alpha = float(
        pipeline.selected_alpha
    )

    if (
        not isfinite(selected_alpha)
        or not 0.0 <= selected_alpha <= 1.0
    ):
        raise FinalRobustnessExperimentError(
            "Validation-selected alpha is invalid."
        )

    system_runtime = {
        "ridge_only": (
            1.0,
            False,
        ),
        "memory_only": (
            0.0,
            True,
        ),
        "hybrid_without_temporal_decay": (
            selected_alpha,
            False,
        ),
        "full_hybrid": (
            selected_alpha,
            True,
        ),
    }

    if tuple(system_runtime) != (
        configuration.systems
    ):
        raise FinalRobustnessExperimentError(
            "Runtime systems differ from the lock."
        )

    events_by_tier = {
        tier.value: tuple(
            sorted(
                (
                    event
                    for event
                    in bundle.graded_dataset.events
                    if event.tier is tier
                ),
                key=lambda event: (
                    event.latent_event_id
                ),
            )
        )
        for tier in OODTier
    }

    for tier, events in events_by_tier.items():
        if len(events) != bundle.source_count:
            raise FinalRobustnessExperimentError(
                f"Tier {tier} does not contain "
                "every latent event."
            )

        latent_ids = tuple(
            event.latent_event_id
            for event in events
        )

        if len(set(latent_ids)) != len(
            latent_ids
        ):
            raise FinalRobustnessExperimentError(
                f"Tier {tier} latent IDs "
                "must be unique."
            )

    evaluations: list[
        FinalRobustnessEvaluation
    ] = []
    cache: dict[
        tuple[
            str,
            tuple[str, ...],
            int,
            float,
            bool,
        ],
        _RawEvaluation,
    ] = {}

    missing_specs = tuple(
        (
            "missing_modality",
            condition.condition_id,
            condition.missing_modalities,
            0,
        )
        for condition
        in configuration.missing_conditions
    )
    temporal_specs = tuple(
        (
            "temporal_displacement",
            f"day-{days}",
            (),
            days,
        )
        for days
        in configuration.temporal_displacement_days
    )

    for (
        axis,
        condition_id,
        missing_modalities,
        displacement_days,
    ) in missing_specs + temporal_specs:
        for system in configuration.systems:
            alpha, apply_decay = (
                system_runtime[system]
            )

            for tier in configuration.severity_tiers:
                cache_key = (
                    tier,
                    missing_modalities,
                    displacement_days,
                    alpha,
                    apply_decay,
                )

                raw = cache.get(cache_key)

                if raw is None:
                    raw = _evaluate_condition(
                        pipeline=pipeline,
                        events=events_by_tier[
                            tier
                        ],
                        missing_modalities=(
                            missing_modalities
                        ),
                        displacement_days=(
                            displacement_days
                        ),
                        trained_at_utc=(
                            trained_at_utc
                        ),
                        alpha=alpha,
                        apply_temporal_decay=(
                            apply_decay
                        ),
                        condition_id=(
                            condition_id
                        ),
                    )
                    cache[cache_key] = raw

                evaluations.append(
                    FinalRobustnessEvaluation(
                        axis=axis,
                        condition_id=condition_id,
                        system=system,
                        tier=tier,
                        missing_modalities=(
                            missing_modalities
                        ),
                        temporal_displacement_days=(
                            displacement_days
                        ),
                        selected_alpha=alpha,
                        apply_temporal_decay=(
                            apply_decay
                        ),
                        event_count=len(
                            raw.latent_event_ids
                        ),
                        latent_event_ids=(
                            raw.latent_event_ids
                        ),
                        reciprocal_ranks=(
                            raw.reciprocal_ranks
                        ),
                        recall_at_1=(
                            raw.recall_at_1
                        ),
                        recall_at_10=(
                            raw.recall_at_10
                        ),
                        mean_reciprocal_rank=(
                            raw
                            .mean_reciprocal_rank
                        ),
                        ndcg_at_10=(
                            raw.ndcg_at_10
                        ),
                    )
                )

    expected_count = (
        (
            len(
                configuration.missing_conditions
            )
            + len(
                configuration
                .temporal_displacement_days
            )
        )
        * len(configuration.systems)
        * len(configuration.severity_tiers)
    )

    if len(evaluations) != expected_count:
        raise FinalRobustnessExperimentError(
            "Robustness evaluation grid is incomplete."
        )

    return FinalRobustnessExperiment(
        run_id=run.run_id,
        generator_seed=run.generator_seed,
        ood_seed=run.ood_seed,
        training_event_count=len(
            pipeline.training_event_ids
        ),
        validation_event_count=len(
            pipeline.validation_event_ids
        ),
        latent_event_count=(
            bundle.source_count
        ),
        observed_event_count=len(
            bundle.graded_dataset.events
        ),
        odor_library_size=len(
            bundle.odor_targets
        ),
        selected_validation_alpha=(
            selected_alpha
        ),
        evaluations=tuple(evaluations),
        paired_analysis_unit=(
            "latent_event_id"
        ),
        all_ood_targets_unreachable=True,
        strict_family_separation_verified=True,
        oracle_used=False,
        ood_tuning_used=False,
        final_test_tuning_used=False,
        protocol_hash=protocol_hash,
    )


def _evaluate_condition(
    *,
    pipeline: NOIPipeline,
    events: tuple[Any, ...],
    missing_modalities: tuple[str, ...],
    displacement_days: int,
    trained_at_utc: datetime,
    alpha: float,
    apply_temporal_decay: bool,
    condition_id: str,
) -> _RawEvaluation:
    evaluation_time = (
        trained_at_utc
        + timedelta(days=displacement_days)
    )
    rankings: list[tuple[str, ...]] = []
    relevant_items: list[
        frozenset[str]
    ] = []
    latent_event_ids: list[str] = []

    missing = set(missing_modalities)

    for event in events:
        context = MultimodalContext(
            event_id=(
                f"{event.observed_event_id}|"
                f"{condition_id}|"
                f"{displacement_days}"
            ),
            timestamp_utc=evaluation_time,
            text_vector=(
                None
                if "text" in missing
                else event.text_vector
            ),
            image_vector=(
                None
                if "image" in missing
                else event.image_vector
            ),
            audio_vector=(
                None
                if "audio" in missing
                else event.audio_vector
            ),
            metadata={
                "latent_event_id": (
                    event.latent_event_id
                ),
                "condition_id": condition_id,
                "missing_modalities": (
                    missing_modalities
                ),
                "temporal_displacement_days": (
                    displacement_days
                ),
            },
        )

        result = pipeline.retrieve(
            context,
            top_k=10,
            alpha=alpha,
            apply_temporal_decay=(
                apply_temporal_decay
            ),
        )

        if result.oracle_used is not False:
            raise FinalRobustnessExperimentError(
                "Pipeline reported prohibited "
                "oracle use."
            )

        ranking = tuple(
            candidate.item_id
            for candidate in result.candidates
        )

        rankings.append(ranking)
        relevant_items.append(
            frozenset(
                (event.target_item_id,)
            )
        )
        latent_event_ids.append(
            event.latent_event_id
        )

    ranking_tuple = tuple(rankings)
    relevant_tuple = tuple(
        relevant_items
    )
    reciprocal_ranks = tuple(
        _reciprocal_rank(
            ranking,
            relevant,
        )
        for ranking, relevant in zip(
            ranking_tuple,
            relevant_tuple,
            strict=True,
        )
    )

    return _RawEvaluation(
        latent_event_ids=tuple(
            latent_event_ids
        ),
        reciprocal_ranks=reciprocal_ranks,
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


def _reciprocal_rank(
    ranking: tuple[str, ...],
    relevant: frozenset[str],
) -> float:
    for rank, item_id in enumerate(
        ranking,
        start=1,
    ):
        if item_id in relevant:
            return 1.0 / rank

    return 0.0


def _validate_inputs(
    *,
    bundle: PairedGradedOODBundle,
    configuration: RobustnessConfiguration,
    run: RobustnessRun,
    system_configuration: Mapping[str, Any],
    policy_configuration: Mapping[str, Any],
    protocol_hash: str,
    trained_at_utc: datetime,
) -> None:
    if not isinstance(
        bundle,
        PairedGradedOODBundle,
    ):
        raise FinalRobustnessExperimentError(
            "bundle must be a paired OOD bundle."
        )

    if not isinstance(
        configuration,
        RobustnessConfiguration,
    ):
        raise FinalRobustnessExperimentError(
            "configuration is invalid."
        )

    if run not in configuration.runs:
        raise FinalRobustnessExperimentError(
            "Run is not in the locked configuration."
        )

    if (
        bundle.generator_seed
        != run.generator_seed
        or bundle.ood_seed
        != run.ood_seed
    ):
        raise FinalRobustnessExperimentError(
            "Bundle seed does not match "
            "the locked run seed."
        )

    if protocol_hash != (
        configuration.configuration_sha256
    ):
        raise FinalRobustnessExperimentError(
            "Robustness protocol hash "
            "does not match."
        )

    if not isinstance(
        system_configuration,
        Mapping,
    ):
        raise FinalRobustnessExperimentError(
            "System configuration is required."
        )

    if not isinstance(
        policy_configuration,
        Mapping,
    ):
        raise FinalRobustnessExperimentError(
            "Policy configuration is required."
        )

    if (
        not isinstance(
            trained_at_utc,
            datetime,
        )
        or trained_at_utc.tzinfo is None
        or trained_at_utc.utcoffset()
        is None
    ):
        raise FinalRobustnessExperimentError(
            "trained_at_utc must be "
            "timezone-aware."
        )

    if configuration.event_count_per_run != (
        10000
    ):
        raise FinalRobustnessExperimentError(
            "Final event count lock is invalid."
        )
