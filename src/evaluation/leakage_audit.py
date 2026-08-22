"""Leakage audits for synthetic NOI evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
    SyntheticEvent,
)


class LeakageAuditError(ValueError):
    """Raised when a dataset violates prespecified leakage safeguards."""


@dataclass(frozen=True)
class LeakageAuditReport:
    """Immutable summary of leakage-audit results."""

    event_count: int
    target_count: int
    split_counts: tuple[tuple[str, int], ...]
    duplicate_event_ids: tuple[str, ...]
    cross_split_feature_duplicates: tuple[str, ...]
    ood_family_overlap: tuple[int, ...]
    template_overlap: tuple[int, ...]
    inconsistent_target_families: tuple[str, ...]
    missing_splits: tuple[str, ...]
    passed: bool

    def require_pass(self) -> None:
        """Raise an exception when any leakage safeguard failed."""

        if self.passed:
            return

        failures: list[str] = []

        if self.duplicate_event_ids:
            failures.append(
                "duplicate event identifiers: "
                f"{list(self.duplicate_event_ids)}"
            )

        if self.cross_split_feature_duplicates:
            failures.append(
                "identical feature records occur across splits: "
                f"{list(self.cross_split_feature_duplicates)}"
            )

        if self.ood_family_overlap:
            failures.append(
                "OOD odor families overlap development families: "
                f"{list(self.ood_family_overlap)}"
            )

        if self.template_overlap:
            failures.append(
                "context templates overlap evaluation partitions: "
                f"{list(self.template_overlap)}"
            )

        if self.inconsistent_target_families:
            failures.append(
                "event target-family labels are inconsistent: "
                f"{list(self.inconsistent_target_families)}"
            )

        if self.missing_splits:
            failures.append(
                "required splits are missing: "
                f"{list(self.missing_splits)}"
            )

        raise LeakageAuditError("; ".join(failures))


def audit_synthetic_dataset(
    dataset: SyntheticDataset,
    *,
    required_splits: Iterable[SplitLabel] = (
        SplitLabel.TRAIN,
        SplitLabel.VALIDATION,
        SplitLabel.OOD_TEST,
    ),
    raise_on_failure: bool = True,
) -> LeakageAuditReport:
    """Audit a synthetic dataset for prespecified forms of leakage.

    The audit checks identifier uniqueness, separation of held-out odor
    families and context templates, cross-split feature duplication, and
    consistency between event labels and the independently generated odor
    library.

    Passing this audit establishes only conformance with these implemented
    safeguards. It does not establish perceptual validity or real-world
    generalization.
    """

    if not isinstance(dataset, SyntheticDataset):
        raise LeakageAuditError(
            "dataset must be an instance of SyntheticDataset."
        )

    required = tuple(required_splits)

    if not required:
        raise LeakageAuditError(
            "At least one required split must be specified."
        )

    if any(not isinstance(split, SplitLabel) for split in required):
        raise LeakageAuditError(
            "Every required split must be a SplitLabel."
        )

    if len(set(required)) != len(required):
        raise LeakageAuditError(
            "required_splits cannot contain duplicates."
        )

    events_by_split: dict[SplitLabel, list[SyntheticEvent]] = {
        split: [] for split in SplitLabel
    }

    for event in dataset.events:
        events_by_split[event.split].append(event)

    split_counts = tuple(
        sorted(
            (
                split.value,
                len(events),
            )
            for split, events in events_by_split.items()
        )
    )

    missing_splits = tuple(
        sorted(
            split.value
            for split in required
            if not events_by_split[split]
        )
    )

    duplicate_event_ids = _find_duplicates(
        event.event_id for event in dataset.events
    )

    target_family_map = {
        target.item_id: target.family_id
        for target in dataset.odor_targets
    }

    inconsistent_target_families = tuple(
        sorted(
            event.event_id
            for event in dataset.events
            if (
                event.target_item_id not in target_family_map
                or target_family_map[event.target_item_id]
                != event.target_family_id
            )
        )
    )

    development_families = {
        event.target_family_id
        for split in (SplitLabel.TRAIN, SplitLabel.VALIDATION)
        for event in events_by_split[split]
    }

    ood_families = {
        event.target_family_id
        for event in events_by_split[SplitLabel.OOD_TEST]
    }

    ood_family_overlap = tuple(
        sorted(development_families & ood_families)
    )

    templates_by_split = {
        split: {
            event.template_id
            for event in events
        }
        for split, events in events_by_split.items()
    }

    template_overlap_values: set[int] = set()

    split_list = list(SplitLabel)

    for index, left_split in enumerate(split_list):
        for right_split in split_list[index + 1:]:
            template_overlap_values.update(
                templates_by_split[left_split]
                & templates_by_split[right_split]
            )

    template_overlap = tuple(sorted(template_overlap_values))

    cross_split_feature_duplicates = (
        _find_cross_split_feature_duplicates(dataset.events)
    )

    passed = not any(
        (
            duplicate_event_ids,
            cross_split_feature_duplicates,
            ood_family_overlap,
            template_overlap,
            inconsistent_target_families,
            missing_splits,
        )
    )

    report = LeakageAuditReport(
        event_count=len(dataset.events),
        target_count=len(dataset.odor_targets),
        split_counts=split_counts,
        duplicate_event_ids=duplicate_event_ids,
        cross_split_feature_duplicates=cross_split_feature_duplicates,
        ood_family_overlap=ood_family_overlap,
        template_overlap=template_overlap,
        inconsistent_target_families=inconsistent_target_families,
        missing_splits=missing_splits,
        passed=passed,
    )

    if raise_on_failure:
        report.require_pass()

    return report


def _find_duplicates(values: Iterable[str]) -> tuple[str, ...]:
    """Return sorted values that occur more than once."""

    observed: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in observed:
            duplicates.add(value)
        else:
            observed.add(value)

    return tuple(sorted(duplicates))


def _find_cross_split_feature_duplicates(
    events: Iterable[SyntheticEvent],
) -> tuple[str, ...]:
    """Return fingerprints reused by events from different splits."""

    fingerprint_splits: dict[str, set[SplitLabel]] = {}

    for event in events:
        fingerprint = _feature_fingerprint(event)
        fingerprint_splits.setdefault(fingerprint, set()).add(event.split)

    return tuple(
        sorted(
            fingerprint
            for fingerprint, splits in fingerprint_splits.items()
            if len(splits) > 1
        )
    )


def _feature_fingerprint(event: SyntheticEvent) -> str:
    """Create a stable fingerprint without using target identifiers."""

    payload = repr(
        (
            event.template_id,
            tuple(event.text_vector),
            tuple(event.image_vector),
            tuple(event.audio_vector),
        )
    ).encode("utf-8")

    return sha256(payload).hexdigest()