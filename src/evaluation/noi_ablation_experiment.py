"""Paired graded-OOD ablation evaluation for the integrated NOI system.

The experiment uses the same latent events across every system, OOD tier,
and temporal displacement. Model fitting and hybrid-alpha selection occur
before OOD evaluation. No OOD oracle or OOD tuning is permitted.

Results are exploratory synthetic computational evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite
from typing import Any, Mapping

from src.evaluation.graded_ood import (
    GradedOODEvent,
    OODTier,
)
from src.evaluation.paired_ood_source_generator import (
    PairedGradedOODBundle,
)
from src.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from src.models import MultimodalContext
from src.system.noi_pipeline import NOIPipeline


LOCKED_TOP_K = 10
LOCKED_TEMPORAL_DISPLACEMENTS = (0, 1, 7, 30, 90)
LOCKED_PAIRED_ANALYSIS_UNIT = "latent_event_id"


class NOIAblationExperimentError(ValueError):
    """Raised when the locked NOI ablation cannot be evaluated."""


class NOIAblationSystem(str, Enum):
    """Prespecified integrated-system ablations."""

    RIDGE_ONLY = "ridge_only"
    MEMORY_ONLY = "memory_only"
    HYBRID_WITHOUT_TEMPORAL_DECAY = (
        "hybrid_without_temporal_decay"
    )
    FULL_HYBRID = "full_hybrid"


@dataclass(frozen=True)
class NOIAblationEvaluation:
    """Metrics and rankings for one system, tier, and time condition."""

    system: NOIAblationSystem
    tier: OODTier
    temporal_displacement_days: int
    selected_alpha: float
    apply_temporal_decay: bool
    latent_event_ids: tuple[str, ...]
    rankings: tuple[tuple[str, ...], ...]
    relevant_items: tuple[frozenset[str], ...]
    top_k: int
    recall_at_1: float
    recall_at_10: float
    mean_reciprocal_rank: float
    ndcg_at_10: float

    def __post_init__(self) -> None:
        if not isinstance(self.system, NOIAblationSystem):
            raise NOIAblationExperimentError(
                "system must be a NOIAblationSystem."
            )

        if not isinstance(self.tier, OODTier):
            raise NOIAblationExperimentError(
                "tier must be an OODTier."
            )

        if (
            isinstance(self.temporal_displacement_days, bool)
            or not isinstance(
                self.temporal_displacement_days,
                int,
            )
            or self.temporal_displacement_days < 0
        ):
            raise NOIAblationExperimentError(
                "temporal_displacement_days must be nonnegative."
            )

        if (
            isinstance(self.selected_alpha, bool)
            or not isinstance(
                self.selected_alpha,
                (int, float),
            )
            or not isfinite(float(self.selected_alpha))
            or not 0.0 <= float(self.selected_alpha) <= 1.0
        ):
            raise NOIAblationExperimentError(
                "selected_alpha must be finite and in [0, 1]."
            )

        if not isinstance(self.apply_temporal_decay, bool):
            raise NOIAblationExperimentError(
                "apply_temporal_decay must be boolean."
            )

        count = len(self.latent_event_ids)

        if count < 1:
            raise NOIAblationExperimentError(
                "At least one latent event is required."
            )

        if (
            len(set(self.latent_event_ids)) != count
            or len(self.rankings) != count
            or len(self.relevant_items) != count
        ):
            raise NOIAblationExperimentError(
                "Paired evaluation arrays must align uniquely."
            )

        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 1
        ):
            raise NOIAblationExperimentError(
                "top_k must be a positive integer."
            )

        for ranking in self.rankings:
            if not ranking or len(ranking) > self.top_k:
                raise NOIAblationExperimentError(
                    "Each ranking must be nonempty and respect top_k."
                )

            if len(set(ranking)) != len(ranking):
                raise NOIAblationExperimentError(
                    "Rankings cannot contain duplicate items."
                )

        for relevant in self.relevant_items:
            if not relevant:
                raise NOIAblationExperimentError(
                    "Every event must have a relevant item."
                )

        for label, value in (
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
                raise NOIAblationExperimentError(
                    f"{label} must be finite and in [0, 1]."
                )


@dataclass(frozen=True)
class NOIAblationExperiment:
    """Complete paired graded-OOD ablation experiment."""

    evaluations: tuple[NOIAblationEvaluation, ...]
    latent_event_count: int
    odor_library_size: int
    training_event_count: int
    validation_event_count: int
    selected_validation_alpha: float
    top_k: int
    temporal_displacements: tuple[int, ...]
    paired_analysis_unit: str
    oracle_used: bool
    ood_tuning_used: bool
    protocol_hash: str

    def __post_init__(self) -> None:
        expected_count = (
            len(NOIAblationSystem)
            * len(OODTier)
            * len(self.temporal_displacements)
        )

        if len(self.evaluations) != expected_count:
            raise NOIAblationExperimentError(
                "The experiment does not contain every locked condition."
            )

        if (
            self.latent_event_count < 1
            or self.odor_library_size < 1
            or self.training_event_count < 1
            or self.validation_event_count < 1
        ):
            raise NOIAblationExperimentError(
                "Experiment counts must be positive."
            )

        if self.top_k != LOCKED_TOP_K:
            raise NOIAblationExperimentError(
                "top_k differs from the locked value."
            )

        if (
            self.temporal_displacements
            != LOCKED_TEMPORAL_DISPLACEMENTS
        ):
            raise NOIAblationExperimentError(
                "Temporal displacements differ from the locked values."
            )

        if (
            self.paired_analysis_unit
            != LOCKED_PAIRED_ANALYSIS_UNIT
        ):
            raise NOIAblationExperimentError(
                "Paired analysis unit must be latent_event_id."
            )

        if self.oracle_used is not False:
            raise NOIAblationExperimentError(
                "OOD oracle use is prohibited."
            )

        if self.ood_tuning_used is not False:
            raise NOIAblationExperimentError(
                "OOD tuning is prohibited."
            )

        if not self.protocol_hash:
            raise NOIAblationExperimentError(
                "protocol_hash must not be empty."
            )

        keys = {
            (
                evaluation.system,
                evaluation.tier,
                evaluation.temporal_displacement_days,
            )
            for evaluation in self.evaluations
        }

        if len(keys) != expected_count:
            raise NOIAblationExperimentError(
                "Experiment conditions must be unique."
            )

    def get(
        self,
        *,
        system: NOIAblationSystem,
        tier: OODTier,
        temporal_displacement_days: int,
    ) -> NOIAblationEvaluation:
        """Return one exact ablation evaluation."""

        if not isinstance(system, NOIAblationSystem):
            raise NOIAblationExperimentError(
                "system must be a NOIAblationSystem."
            )

        if not isinstance(tier, OODTier):
            raise NOIAblationExperimentError(
                "tier must be an OODTier."
            )

        for evaluation in self.evaluations:
            if (
                evaluation.system is system
                and evaluation.tier is tier
                and evaluation.temporal_displacement_days
                == temporal_displacement_days
            ):
                return evaluation

        raise NOIAblationExperimentError(
            "Requested ablation evaluation was not found."
        )


def run_noi_ablation_experiment(
    bundle: PairedGradedOODBundle,
    *,
    system_configuration: Mapping[str, Any],
    policy_configuration: Mapping[str, Any],
    protocol_hash: str,
    trained_at_utc: datetime,
    top_k: int = LOCKED_TOP_K,
) -> NOIAblationExperiment:
    """Fit NOI without OOD access and evaluate every locked ablation."""

    if not isinstance(bundle, PairedGradedOODBundle):
        raise NOIAblationExperimentError(
            "bundle must be a PairedGradedOODBundle."
        )

    if bundle.severe_reference_verified is not True:
        raise NOIAblationExperimentError(
            "Severe reference replay must be verified."
        )

    if (
        trained_at_utc.tzinfo is None
        or trained_at_utc.utcoffset() is None
    ):
        raise NOIAblationExperimentError(
            "trained_at_utc must be timezone-aware."
        )

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k != LOCKED_TOP_K
    ):
        raise NOIAblationExperimentError(
            "top_k must equal the locked value 10."
        )

    ablation = system_configuration.get(
        "ablation_evaluation"
    )

    if not isinstance(ablation, Mapping):
        raise NOIAblationExperimentError(
            "ablation_evaluation configuration is required."
        )

    displacements = tuple(
        ablation.get(
            "temporal_displacement_days",
            (),
        )
    )

    if displacements != LOCKED_TEMPORAL_DISPLACEMENTS:
        raise NOIAblationExperimentError(
            "Temporal displacement configuration is not locked."
        )

    systems_configuration = ablation.get("systems")

    if not isinstance(systems_configuration, Mapping):
        raise NOIAblationExperimentError(
            "Locked ablation systems are required."
        )

    if tuple(systems_configuration) != tuple(
        system.value for system in NOIAblationSystem
    ):
        raise NOIAblationExperimentError(
            "Ablation systems differ from the locked order."
        )

    if ablation.get("ood_alpha_tuning_prohibited") is not True:
        raise NOIAblationExperimentError(
            "OOD alpha tuning must be prohibited."
        )

    if ablation.get("ood_model_fitting_prohibited") is not True:
        raise NOIAblationExperimentError(
            "OOD model fitting must be prohibited."
        )

    if ablation.get("oracle_used") is not False:
        raise NOIAblationExperimentError(
            "OOD oracle must be disabled."
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

    selected_alpha = pipeline.selected_alpha

    system_runtime = {
        NOIAblationSystem.RIDGE_ONLY: (
            1.0,
            False,
        ),
        NOIAblationSystem.MEMORY_ONLY: (
            0.0,
            True,
        ),
        NOIAblationSystem.HYBRID_WITHOUT_TEMPORAL_DECAY: (
            selected_alpha,
            False,
        ),
        NOIAblationSystem.FULL_HYBRID: (
            selected_alpha,
            True,
        ),
    }

    graded_events = tuple(bundle.graded_dataset.events)
    evaluations: list[NOIAblationEvaluation] = []

    for system in NOIAblationSystem:
        alpha, apply_temporal_decay = system_runtime[system]

        for tier in OODTier:
            tier_events = tuple(
                sorted(
                    (
                        event
                        for event in graded_events
                        if event.tier is tier
                    ),
                    key=lambda event: event.latent_event_id,
                )
            )

            if len(tier_events) != bundle.source_count:
                raise NOIAblationExperimentError(
                    "Each tier must contain every latent event."
                )

            latent_ids = tuple(
                event.latent_event_id
                for event in tier_events
            )

            if len(set(latent_ids)) != len(latent_ids):
                raise NOIAblationExperimentError(
                    "Tier latent_event_id values must be unique."
                )

            for displacement_days in displacements:
                evaluation_time = (
                    trained_at_utc
                    + timedelta(days=displacement_days)
                )

                rankings: list[tuple[str, ...]] = []
                relevant_items: list[frozenset[str]] = []

                for event in tier_events:
                    context = _context_from_graded_event(
                        event,
                        timestamp_utc=evaluation_time,
                    )

                    result = pipeline.retrieve(
                        context,
                        top_k=top_k,
                        alpha=alpha,
                        apply_temporal_decay=(
                            apply_temporal_decay
                        ),
                    )

                    if result.oracle_used is not False:
                        raise NOIAblationExperimentError(
                            "Pipeline reported prohibited oracle use."
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
                    NOIAblationEvaluation(
                        system=system,
                        tier=tier,
                        temporal_displacement_days=(
                            displacement_days
                        ),
                        selected_alpha=alpha,
                        apply_temporal_decay=(
                            apply_temporal_decay
                        ),
                        latent_event_ids=latent_ids,
                        rankings=ranking_tuple,
                        relevant_items=relevant_tuple,
                        top_k=top_k,
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

    return NOIAblationExperiment(
        evaluations=tuple(evaluations),
        latent_event_count=bundle.source_count,
        odor_library_size=len(bundle.odor_targets),
        training_event_count=len(
            pipeline.training_event_ids
        ),
        validation_event_count=len(
            pipeline.validation_event_ids
        ),
        selected_validation_alpha=selected_alpha,
        top_k=top_k,
        temporal_displacements=displacements,
        paired_analysis_unit=LOCKED_PAIRED_ANALYSIS_UNIT,
        oracle_used=False,
        ood_tuning_used=False,
        protocol_hash=protocol_hash,
    )


def _context_from_graded_event(
    event: GradedOODEvent,
    *,
    timestamp_utc: datetime,
) -> MultimodalContext:
    """Convert a graded event into an integrated runtime context."""

    return MultimodalContext(
        event_id=event.observed_event_id,
        timestamp_utc=timestamp_utc,
        text_vector=event.text_vector,
        image_vector=event.image_vector,
        audio_vector=event.audio_vector,
        metadata={
            "latent_event_id": event.latent_event_id,
            "ood_tier": event.tier.value,
            "temporal_displacement_days": (
                timestamp_utc.isoformat()
            ),
        },
    )
