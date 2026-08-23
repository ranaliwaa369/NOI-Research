"""Prespecified baseline evaluation across paired graded-OOD tiers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from src.baselines.retrieval_baselines import (
    BaselineKind,
    RidgeFusionRetriever,
    _random_ranking,
    build_odor_library,
    mean_fuse_event,
)
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
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticEvent,
)
from src.retrieval.cosine_retriever import (
    CosineOdorRetriever,
    OdorLibraryItem,
)


Ranking = tuple[str, ...]
RelevantItems = frozenset[str]
LOCKED_TOP_K = 10


class GradedOODExperimentError(ValueError):
    """Raised when graded-OOD baseline evaluation is invalid."""


@dataclass(frozen=True)
class GradedOODEvaluation:
    """Results for one baseline on one graded-OOD tier."""

    baseline: BaselineKind
    tier: OODTier
    latent_event_ids: tuple[str, ...]
    rankings: tuple[Ranking, ...]
    relevant_items: tuple[RelevantItems, ...]
    top_k: int
    training_event_count: int
    recall_at_1: float
    recall_at_10: float
    mean_reciprocal_rank: float
    ndcg_at_10: float

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, BaselineKind):
            raise GradedOODExperimentError(
                "baseline must be a BaselineKind."
            )

        if not isinstance(self.tier, OODTier):
            raise GradedOODExperimentError(
                "tier must be an OODTier."
            )

        count = len(self.latent_event_ids)

        if count < 1:
            raise GradedOODExperimentError(
                "At least one latent event is required."
            )

        if len(set(self.latent_event_ids)) != count:
            raise GradedOODExperimentError(
                "latent_event_ids must be unique."
            )

        if len(self.rankings) != count:
            raise GradedOODExperimentError(
                "Every latent event must have one ranking."
            )

        if len(self.relevant_items) != count:
            raise GradedOODExperimentError(
                "Every latent event must have one relevance set."
            )

        if self.top_k != LOCKED_TOP_K:
            raise GradedOODExperimentError(
                "top_k must remain locked at 10."
            )

        if any(
            len(ranking) != self.top_k
            for ranking in self.rankings
        ):
            raise GradedOODExperimentError(
                "Every ranking must contain exactly top_k items."
            )

        if any(
            len(set(ranking)) != len(ranking)
            for ranking in self.rankings
        ):
            raise GradedOODExperimentError(
                "A ranking cannot contain duplicate item identifiers."
            )

        if any(
            not relevant
            for relevant in self.relevant_items
        ):
            raise GradedOODExperimentError(
                "Every relevance set must be nonempty."
            )

        if (
            isinstance(self.training_event_count, bool)
            or not isinstance(self.training_event_count, int)
            or self.training_event_count < 1
        ):
            raise GradedOODExperimentError(
                "training_event_count must be a positive integer."
            )

        metrics = {
            "recall_at_1": self.recall_at_1,
            "recall_at_10": self.recall_at_10,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "ndcg_at_10": self.ndcg_at_10,
        }

        for name, value in metrics.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise GradedOODExperimentError(
                    f"{name} must be finite and in [0, 1]."
                )

        if self.recall_at_1 > self.recall_at_10:
            raise GradedOODExperimentError(
                "recall_at_1 cannot exceed recall_at_10."
            )

    @property
    def event_count(self) -> int:
        """Return the number of independent latent evaluation units."""

        return len(self.latent_event_ids)


@dataclass(frozen=True)
class GradedOODExperiment:
    """Complete four-baseline by three-tier evaluation."""

    evaluations: tuple[GradedOODEvaluation, ...]
    latent_event_count: int
    odor_library_size: int
    top_k: int
    random_seed: int
    ridge_alpha: float
    oracle_used: bool
    paired_analysis_unit: str

    def __post_init__(self) -> None:
        expected_count = len(BaselineKind) * len(OODTier)

        if len(self.evaluations) != expected_count:
            raise GradedOODExperimentError(
                f"The experiment must contain {expected_count} evaluations."
            )

        keys = [
            (evaluation.tier, evaluation.baseline)
            for evaluation in self.evaluations
        ]

        if len(set(keys)) != expected_count:
            raise GradedOODExperimentError(
                "Every tier-baseline pair must appear exactly once."
            )

        if (
            isinstance(self.latent_event_count, bool)
            or not isinstance(self.latent_event_count, int)
            or self.latent_event_count < 1
        ):
            raise GradedOODExperimentError(
                "latent_event_count must be a positive integer."
            )

        if self.odor_library_size < self.top_k:
            raise GradedOODExperimentError(
                "odor_library_size cannot be smaller than top_k."
            )

        if self.top_k != LOCKED_TOP_K:
            raise GradedOODExperimentError(
                "Experiment top_k must remain locked at 10."
            )

        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise GradedOODExperimentError(
                "random_seed must be a nonnegative integer."
            )

        if (
            isinstance(self.ridge_alpha, bool)
            or not isinstance(self.ridge_alpha, (int, float))
            or not isfinite(float(self.ridge_alpha))
            or float(self.ridge_alpha) < 0.0
        ):
            raise GradedOODExperimentError(
                "ridge_alpha must be finite and nonnegative."
            )

        if self.oracle_used is not False:
            raise GradedOODExperimentError(
                "OOD oracle use is prohibited in baseline evaluation."
            )

        if self.paired_analysis_unit != "latent_event_id":
            raise GradedOODExperimentError(
                "paired_analysis_unit must be latent_event_id."
            )

        if any(
            evaluation.event_count != self.latent_event_count
            for evaluation in self.evaluations
        ):
            raise GradedOODExperimentError(
                "All evaluations must use the same latent-event count."
            )

    def get(
        self,
        *,
        tier: OODTier,
        baseline: BaselineKind,
    ) -> GradedOODEvaluation:
        """Return one uniquely identified tier-baseline evaluation."""

        if not isinstance(tier, OODTier):
            raise GradedOODExperimentError(
                "tier must be an OODTier."
            )

        if not isinstance(baseline, BaselineKind):
            raise GradedOODExperimentError(
                "baseline must be a BaselineKind."
            )

        for evaluation in self.evaluations:
            if (
                evaluation.tier is tier
                and evaluation.baseline is baseline
            ):
                return evaluation

        raise GradedOODExperimentError(
            f"Missing evaluation for {tier.value}/{baseline.value}."
        )


def run_graded_ood_experiment(
    bundle: PairedGradedOODBundle,
    *,
    top_k: int = LOCKED_TOP_K,
    random_seed: int = 2026,
    ridge_alpha: float = 1.0,
) -> GradedOODExperiment:
    """Run all prespecified baselines without OOD-label calibration."""

    if not isinstance(bundle, PairedGradedOODBundle):
        raise GradedOODExperimentError(
            "bundle must be a PairedGradedOODBundle."
        )

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k != LOCKED_TOP_K
    ):
        raise GradedOODExperimentError(
            "top_k must remain locked at 10."
        )

    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise GradedOODExperimentError(
            "random_seed must be a nonnegative integer."
        )

    if (
        isinstance(ridge_alpha, bool)
        or not isinstance(ridge_alpha, (int, float))
        or not isfinite(float(ridge_alpha))
        or float(ridge_alpha) < 0.0
    ):
        raise GradedOODExperimentError(
            "ridge_alpha must be finite and nonnegative."
        )

    library = build_odor_library(
        bundle.original_dataset
    )

    if len(library) < top_k:
        raise GradedOODExperimentError(
            "The odor library is smaller than top_k."
        )

    training_count = sum(
        event.split is SplitLabel.TRAIN
        for event in bundle.original_dataset.events
    )

    if training_count < 1:
        raise GradedOODExperimentError(
            "At least one training event is required."
        )

    cosine_retriever = CosineOdorRetriever(library)

    ridge_retriever = RidgeFusionRetriever(
        alpha=float(ridge_alpha)
    ).fit(bundle.original_dataset)

    if ridge_retriever.training_event_count != training_count:
        raise GradedOODExperimentError(
            "Ridge training count does not match the locked train split."
        )

    evaluations: list[GradedOODEvaluation] = []

    for tier in OODTier:
        tier_events = bundle.graded_dataset.events_for_tier(
            tier
        )

        if len(tier_events) != bundle.source_count:
            raise GradedOODExperimentError(
                f"Tier {tier.value} does not contain every latent event."
            )

        for baseline in BaselineKind:
            evaluation = _evaluate_tier_baseline(
                tier_events=tier_events,
                tier=tier,
                baseline=baseline,
                library=library,
                cosine_retriever=cosine_retriever,
                ridge_retriever=ridge_retriever,
                top_k=top_k,
                random_seed=random_seed,
                training_event_count=training_count,
            )
            evaluations.append(evaluation)

    return GradedOODExperiment(
        evaluations=tuple(evaluations),
        latent_event_count=bundle.source_count,
        odor_library_size=len(library),
        top_k=top_k,
        random_seed=random_seed,
        ridge_alpha=float(ridge_alpha),
        oracle_used=False,
        paired_analysis_unit="latent_event_id",
    )


def _evaluate_tier_baseline(
    *,
    tier_events: tuple[GradedOODEvent, ...],
    tier: OODTier,
    baseline: BaselineKind,
    library: tuple[OdorLibraryItem, ...],
    cosine_retriever: CosineOdorRetriever,
    ridge_retriever: RidgeFusionRetriever,
    top_k: int,
    random_seed: int,
    training_event_count: int,
) -> GradedOODEvaluation:
    """Evaluate one baseline on one complete paired tier."""

    rankings: list[Ranking] = []
    relevant_items: list[RelevantItems] = []
    latent_ids: list[str] = []

    for graded_event in tier_events:
        event = _as_synthetic_event(
            graded_event
        )

        if baseline is BaselineKind.RANDOM:
            ranking = _random_ranking(
                library,
                event_id=graded_event.latent_event_id,
                top_k=top_k,
                random_seed=random_seed,
            )

        elif baseline is BaselineKind.TEXT_ONLY_COSINE:
            candidates = cosine_retriever.retrieve(
                event.text_vector,
                top_k=top_k,
            )
            ranking = tuple(
                candidate.item_id
                for candidate in candidates
            )

        elif baseline is BaselineKind.MEAN_FUSION_COSINE:
            candidates = cosine_retriever.retrieve(
                mean_fuse_event(event),
                top_k=top_k,
            )
            ranking = tuple(
                candidate.item_id
                for candidate in candidates
            )

        elif baseline is BaselineKind.RIDGE_FUSION:
            ranking = ridge_retriever.retrieve(
                event,
                top_k=top_k,
            )

        else:
            raise GradedOODExperimentError(
                f"Unsupported baseline: {baseline}"
            )

        latent_ids.append(
            graded_event.latent_event_id
        )
        rankings.append(ranking)
        relevant_items.append(
            frozenset((graded_event.target_item_id,))
        )

    ranking_tuple = tuple(rankings)
    relevance_tuple = tuple(relevant_items)

    return GradedOODEvaluation(
        baseline=baseline,
        tier=tier,
        latent_event_ids=tuple(latent_ids),
        rankings=ranking_tuple,
        relevant_items=relevance_tuple,
        top_k=top_k,
        training_event_count=training_event_count,
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
        mean_reciprocal_rank=mean_reciprocal_rank(
            ranking_tuple,
            relevance_tuple,
        ),
        ndcg_at_10=ndcg_at_k(
            ranking_tuple,
            relevance_tuple,
            k=10,
        ),
    )


def _as_synthetic_event(
    event: GradedOODEvent,
) -> SyntheticEvent:
    """Convert one graded view for reuse by locked baseline code."""

    if not isinstance(event, GradedOODEvent):
        raise GradedOODExperimentError(
            "event must be a GradedOODEvent."
        )

    return SyntheticEvent(
        event_id=event.observed_event_id,
        split=SplitLabel.OOD_TEST,
        template_id=event.template_id,
        target_item_id=event.target_item_id,
        target_family_id=event.target_family_id,
        text_vector=event.text_vector,
        image_vector=event.image_vector,
        audio_vector=event.audio_vector,
    )
