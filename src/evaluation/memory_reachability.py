"""Audit whether evaluation targets are represented in memory."""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
)


class MemoryReachabilityError(ValueError):
    """Raised when memory reachability cannot be audited."""


@dataclass(frozen=True, slots=True)
class MemoryReachabilitySummary:
    """Reachability summary for one evaluation split."""

    split: SplitLabel
    reachable_target_ids: tuple[str, ...]
    unreachable_target_ids: tuple[str, ...]
    reachable_event_ids: tuple[str, ...]
    unreachable_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split is SplitLabel.TRAIN:
            raise MemoryReachabilityError(
                "A reachability summary requires an evaluation split."
            )

        reachable_targets = set(self.reachable_target_ids)
        unreachable_targets = set(self.unreachable_target_ids)
        reachable_events = set(self.reachable_event_ids)
        unreachable_events = set(self.unreachable_event_ids)

        if reachable_targets & unreachable_targets:
            raise MemoryReachabilityError(
                "Reachable and unreachable targets must be disjoint."
            )

        if reachable_events & unreachable_events:
            raise MemoryReachabilityError(
                "Reachable and unreachable events must be disjoint."
            )

    @property
    def target_count(self) -> int:
        """Return the number of unique targets in this split."""

        return (
            len(self.reachable_target_ids)
            + len(self.unreachable_target_ids)
        )

    @property
    def event_count(self) -> int:
        """Return the number of events in this split."""

        return (
            len(self.reachable_event_ids)
            + len(self.unreachable_event_ids)
        )

    @property
    def reachable_target_fraction(self) -> float:
        """Return the fraction of targets represented during training."""

        if self.target_count == 0:
            return 0.0

        return len(self.reachable_target_ids) / self.target_count

    @property
    def reachable_event_fraction(self) -> float:
        """Return the fraction of events whose targets are reachable."""

        if self.event_count == 0:
            return 0.0

        return len(self.reachable_event_ids) / self.event_count


@dataclass(frozen=True, slots=True)
class MemoryReachabilityAudit:
    """Training support and evaluation reachability summaries."""

    training_target_ids: tuple[str, ...]
    summaries: tuple[MemoryReachabilitySummary, ...]

    def __post_init__(self) -> None:
        if not self.training_target_ids:
            raise MemoryReachabilityError(
                "At least one training target is required."
            )

        splits = tuple(summary.split for summary in self.summaries)

        if len(splits) != len(set(splits)):
            raise MemoryReachabilityError(
                "Every evaluation split may have only one summary."
            )

    def for_split(
        self,
        split: SplitLabel,
    ) -> MemoryReachabilitySummary:
        """Return the reachability summary for one evaluation split."""

        if not isinstance(split, SplitLabel):
            raise MemoryReachabilityError(
                "split must be a SplitLabel."
            )

        if split is SplitLabel.TRAIN:
            raise MemoryReachabilityError(
                "Training is not an evaluation split."
            )

        for summary in self.summaries:
            if summary.split is split:
                return summary

        raise MemoryReachabilityError(
            f"No reachability summary exists for {split.value}."
        )


def audit_memory_reachability(
    dataset: SyntheticDataset,
) -> MemoryReachabilityAudit:
    """Audit target reachability using training memory support only."""

    if not isinstance(dataset, SyntheticDataset):
        raise MemoryReachabilityError(
            "dataset must be a SyntheticDataset."
        )

    training_target_ids = tuple(
        sorted(
            {
                event.target_item_id
                for event in dataset.events
                if event.split is SplitLabel.TRAIN
            }
        )
    )

    if not training_target_ids:
        raise MemoryReachabilityError(
            "At least one training target is required."
        )

    training_targets = set(training_target_ids)
    summaries = []

    for split in (
        SplitLabel.VALIDATION,
        SplitLabel.OOD_TEST,
    ):
        events = tuple(
            event
            for event in dataset.events
            if event.split is split
        )

        target_ids = {
            event.target_item_id
            for event in events
        }

        reachable_target_ids = tuple(
            sorted(target_ids & training_targets)
        )
        unreachable_target_ids = tuple(
            sorted(target_ids - training_targets)
        )

        reachable_event_ids = tuple(
            sorted(
                event.event_id
                for event in events
                if event.target_item_id in training_targets
            )
        )
        unreachable_event_ids = tuple(
            sorted(
                event.event_id
                for event in events
                if event.target_item_id not in training_targets
            )
        )

        summaries.append(
            MemoryReachabilitySummary(
                split=split,
                reachable_target_ids=reachable_target_ids,
                unreachable_target_ids=unreachable_target_ids,
                reachable_event_ids=reachable_event_ids,
                unreachable_event_ids=unreachable_event_ids,
            )
        )

    return MemoryReachabilityAudit(
        training_target_ids=training_target_ids,
        summaries=tuple(summaries),
    )
