"""Final leakage-resistant NOI v0.2 seen-item experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from src.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.seen_item_memory_experiment import (
    SeenItemEvaluation,
    SeenItemSystem,
)
from src.evaluation.seen_item_partition import (
    SeenItemPartitionConfig,
    create_seen_item_partition,
)
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
)
from src.models import MultimodalContext
from src.system.noi_pipeline import NOIPipeline


TOP_K = 10


class SeenItemFinalExperimentError(ValueError):
    """Raised when the final Track A experiment cannot run."""


@dataclass(frozen=True, slots=True)
class SeenItemFinalExperiment:
    """Final Track A results from a held-out template group."""

    training_event_count: int
    calibration_event_count: int
    final_test_event_count: int
    raw_final_test_event_count: int
    reachable_event_fraction: float
    calibration_template_ids: tuple[int, ...]
    final_test_template_ids: tuple[int, ...]
    final_test_event_ids: tuple[str, ...]
    selected_hybrid_alpha: float
    evaluations: tuple[SeenItemEvaluation, ...]
    oracle_used: bool
    final_test_tuning_used: bool
    protocol_hash: str

    def __post_init__(self) -> None:
        for name, value in (
            (
                "training_event_count",
                self.training_event_count,
            ),
            (
                "calibration_event_count",
                self.calibration_event_count,
            ),
            (
                "final_test_event_count",
                self.final_test_event_count,
            ),
            (
                "raw_final_test_event_count",
                self.raw_final_test_event_count,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise SeenItemFinalExperimentError(
                    f"{name} must be a positive integer."
                )

        if (
            self.final_test_event_count
            > self.raw_final_test_event_count
        ):
            raise SeenItemFinalExperimentError(
                "Final reachable count exceeds the raw count."
            )

        expected_fraction = (
            self.final_test_event_count
            / self.raw_final_test_event_count
        )

        if self.reachable_event_fraction != expected_fraction:
            raise SeenItemFinalExperimentError(
                "Reachability fraction is inconsistent."
            )

        calibration_templates = set(
            self.calibration_template_ids
        )
        final_templates = set(
            self.final_test_template_ids
        )

        if not calibration_templates or not final_templates:
            raise SeenItemFinalExperimentError(
                "Template groups must not be empty."
            )

        if calibration_templates & final_templates:
            raise SeenItemFinalExperimentError(
                "Calibration and final templates must be disjoint."
            )

        if (
            len(self.final_test_event_ids)
            != self.final_test_event_count
            or len(set(self.final_test_event_ids))
            != len(self.final_test_event_ids)
        ):
            raise SeenItemFinalExperimentError(
                "Final event identifiers are inconsistent."
            )

        if not 0.0 <= self.selected_hybrid_alpha <= 1.0:
            raise SeenItemFinalExperimentError(
                "Selected hybrid alpha must be in [0, 1]."
            )

        systems = tuple(
            evaluation.system
            for evaluation in self.evaluations
        )

        if set(systems) != set(SeenItemSystem):
            raise SeenItemFinalExperimentError(
                "Every prespecified system must be evaluated."
            )

        if len(systems) != len(set(systems)):
            raise SeenItemFinalExperimentError(
                "Every system may have only one evaluation."
            )

        if any(
            evaluation.event_ids
            != self.final_test_event_ids
            for evaluation in self.evaluations
        ):
            raise SeenItemFinalExperimentError(
                "Systems must use identical final events."
            )

        if self.oracle_used is not False:
            raise SeenItemFinalExperimentError(
                "Oracle use is prohibited."
            )

        if self.final_test_tuning_used is not False:
            raise SeenItemFinalExperimentError(
                "Final-test tuning is prohibited."
            )

        if (
            not isinstance(self.protocol_hash, str)
            or not self.protocol_hash.strip()
        ):
            raise SeenItemFinalExperimentError(
                "protocol_hash must not be empty."
            )

    def for_system(
        self,
        system: SeenItemSystem,
    ) -> SeenItemEvaluation:
        """Return the result for one prespecified system."""

        if not isinstance(system, SeenItemSystem):
            raise SeenItemFinalExperimentError(
                "system must be a SeenItemSystem."
            )

        for evaluation in self.evaluations:
            if evaluation.system is system:
                return evaluation

        raise SeenItemFinalExperimentError(
            f"No result exists for {system.value}."
        )


def run_seen_item_final_experiment(
    *,
    dataset: SyntheticDataset,
    partition_config: SeenItemPartitionConfig,
    system_configuration: Mapping[str, Any],
    policy_configuration: Mapping[str, Any],
    protocol_hash: str,
    trained_at_utc: datetime,
) -> SeenItemFinalExperiment:
    """Fit on train/calibration and evaluate the held-out Track A set."""

    if not isinstance(dataset, SyntheticDataset):
        raise SeenItemFinalExperimentError(
            "dataset must be a SyntheticDataset."
        )

    if not isinstance(
        partition_config,
        SeenItemPartitionConfig,
    ):
        raise SeenItemFinalExperimentError(
            "partition_config has an invalid type."
        )

    if not isinstance(system_configuration, Mapping):
        raise SeenItemFinalExperimentError(
            "system_configuration must be a mapping."
        )

    if not isinstance(policy_configuration, Mapping):
        raise SeenItemFinalExperimentError(
            "policy_configuration must be a mapping."
        )

    if (
        not isinstance(protocol_hash, str)
        or not protocol_hash.strip()
    ):
        raise SeenItemFinalExperimentError(
            "protocol_hash must not be empty."
        )

    if (
        not isinstance(trained_at_utc, datetime)
        or trained_at_utc.tzinfo is None
        or trained_at_utc.utcoffset() is None
    ):
        raise SeenItemFinalExperimentError(
            "trained_at_utc must be timezone-aware."
        )

    partition = create_seen_item_partition(
        dataset,
        partition_config,
    )

    pipeline = NOIPipeline(
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=protocol_hash,
    )

    pipeline.fit(
        partition.fit_dataset,
        trained_at_utc=trained_at_utc,
    )

    final_events = partition.seen_item_test_events

    if not final_events:
        raise SeenItemFinalExperimentError(
            "The final seen-item test set is empty."
        )

    final_event_ids = tuple(
        event.event_id
        for event in final_events
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

        for event in final_events:
            context = MultimodalContext(
                event_id=event.event_id,
                timestamp_utc=trained_at_utc,
                text_vector=event.text_vector,
                image_vector=event.image_vector,
                audio_vector=event.audio_vector,
                metadata={
                    "evaluation_track": (
                        "seen_item_final"
                    ),
                    "template_id": event.template_id,
                    "memory_reachable": True,
                },
            )

            result = pipeline.retrieve(
                context,
                top_k=TOP_K,
                alpha=alpha,
                apply_temporal_decay=True,
            )

            if result.oracle_used:
                raise SeenItemFinalExperimentError(
                    "Retrieval reported forbidden oracle use."
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
                event_ids=final_event_ids,
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

    training_event_count = sum(
        event.split is SplitLabel.TRAIN
        for event in partition.fit_dataset.events
    )

    return SeenItemFinalExperiment(
        training_event_count=training_event_count,
        calibration_event_count=len(
            partition.calibration_events
        ),
        final_test_event_count=len(final_events),
        raw_final_test_event_count=(
            partition.raw_seen_item_test_event_count
        ),
        reachable_event_fraction=(
            len(final_events)
            / partition.raw_seen_item_test_event_count
        ),
        calibration_template_ids=(
            partition.calibration_template_ids
        ),
        final_test_template_ids=(
            partition.seen_item_test_template_ids
        ),
        final_test_event_ids=final_event_ids,
        selected_hybrid_alpha=float(
            pipeline.selected_alpha
        ),
        evaluations=tuple(evaluations),
        oracle_used=False,
        final_test_tuning_used=False,
        protocol_hash=protocol_hash,
    )
