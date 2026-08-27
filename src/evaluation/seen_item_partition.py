"""Deterministic template-group partition for NOI Track A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
    SyntheticEvent,
)


class SeenItemPartitionError(ValueError):
    """Raised when a valid seen-item partition cannot be created."""


@dataclass(frozen=True, slots=True)
class SeenItemPartitionConfig:
    """Locked configuration for the template-group partition."""

    total_event_count: int
    calibration_fraction: float
    partition_seed: int
    grouping_field: str
    prohibit_template_overlap: bool
    prohibit_random_row_split: bool
    require_training_target_reachability: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_event_count, bool)
            or not isinstance(self.total_event_count, int)
            or self.total_event_count < 1
        ):
            raise SeenItemPartitionError(
                "total_event_count must be a positive integer."
            )

        if (
            isinstance(self.calibration_fraction, bool)
            or not isinstance(
                self.calibration_fraction,
                (int, float),
            )
            or not 0.0
            < float(self.calibration_fraction)
            < 1.0
        ):
            raise SeenItemPartitionError(
                "calibration_fraction must be between 0 and 1."
            )

        if (
            isinstance(self.partition_seed, bool)
            or not isinstance(self.partition_seed, int)
        ):
            raise SeenItemPartitionError(
                "partition_seed must be an integer."
            )

        if self.grouping_field != "template_id":
            raise SeenItemPartitionError(
                "The grouping field must be template_id."
            )

        boolean_fields = {
            "prohibit_template_overlap": (
                self.prohibit_template_overlap
            ),
            "prohibit_random_row_split": (
                self.prohibit_random_row_split
            ),
            "require_training_target_reachability": (
                self.require_training_target_reachability
            ),
        }

        for name, value in boolean_fields.items():
            if not isinstance(value, bool):
                raise SeenItemPartitionError(
                    f"{name} must be boolean."
                )

        if not self.prohibit_template_overlap:
            raise SeenItemPartitionError(
                "Template overlap must be prohibited."
            )

        if not self.prohibit_random_row_split:
            raise SeenItemPartitionError(
                "Random row splitting must be prohibited."
            )

        if not self.require_training_target_reachability:
            raise SeenItemPartitionError(
                "Final-test targets must be memory-reachable."
            )


@dataclass(frozen=True, slots=True)
class SeenItemPartition:
    """Calibration and final seen-item groups with a fit dataset."""

    calibration_template_ids: tuple[int, ...]
    seen_item_test_template_ids: tuple[int, ...]
    calibration_events: tuple[SyntheticEvent, ...]
    seen_item_test_events: tuple[SyntheticEvent, ...]
    unreachable_test_event_ids: tuple[str, ...]
    raw_seen_item_test_event_count: int
    training_target_ids: tuple[str, ...]
    fit_dataset: SyntheticDataset
    partition_seed: int

    def __post_init__(self) -> None:
        calibration_templates = set(
            self.calibration_template_ids
        )
        test_templates = set(
            self.seen_item_test_template_ids
        )

        if not calibration_templates or not test_templates:
            raise SeenItemPartitionError(
                "Both template partitions must be nonempty."
            )

        if calibration_templates & test_templates:
            raise SeenItemPartitionError(
                "Calibration and test templates must be disjoint."
            )

        calibration_ids = {
            event.event_id
            for event in self.calibration_events
        }
        test_ids = {
            event.event_id
            for event in self.seen_item_test_events
        }
        unreachable_ids = set(
            self.unreachable_test_event_ids
        )

        if not calibration_ids:
            raise SeenItemPartitionError(
                "Calibration must contain events."
            )

        if not test_ids:
            raise SeenItemPartitionError(
                "Seen-item test must contain reachable events."
            )

        if (
            calibration_ids & test_ids
            or calibration_ids & unreachable_ids
            or test_ids & unreachable_ids
        ):
            raise SeenItemPartitionError(
                "Partition event identifiers must be disjoint."
            )

        if (
            len(test_ids) + len(unreachable_ids)
            != self.raw_seen_item_test_event_count
        ):
            raise SeenItemPartitionError(
                "Raw test-event accounting is inconsistent."
            )

        training_targets = set(self.training_target_ids)

        if not training_targets:
            raise SeenItemPartitionError(
                "Training targets must not be empty."
            )

        if any(
            event.target_item_id not in training_targets
            for event in self.seen_item_test_events
        ):
            raise SeenItemPartitionError(
                "Every retained test target must be reachable."
            )

        if any(
            event.template_id not in calibration_templates
            for event in self.calibration_events
        ):
            raise SeenItemPartitionError(
                "Calibration event has an invalid template."
            )

        if any(
            event.template_id not in test_templates
            for event in self.seen_item_test_events
        ):
            raise SeenItemPartitionError(
                "Test event has an invalid template."
            )

        fit_ids = {
            event.event_id
            for event in self.fit_dataset.events
        }

        if fit_ids & test_ids:
            raise SeenItemPartitionError(
                "Final test events cannot enter fit_dataset."
            )

        if any(
            event.split is SplitLabel.OOD_TEST
            for event in self.fit_dataset.events
        ):
            raise SeenItemPartitionError(
                "OOD events cannot enter fit_dataset."
            )


def load_seen_item_partition_config(
    path: str | Path,
) -> SeenItemPartitionConfig:
    """Load and validate the locked Track A partition settings."""

    config_path = Path(path)

    if not config_path.is_file():
        raise SeenItemPartitionError(
            f"Configuration file not found: {config_path}"
        )

    try:
        raw = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        raise SeenItemPartitionError(
            "Unable to read the partition configuration."
        ) from exc

    if not isinstance(raw, Mapping):
        raise SeenItemPartitionError(
            "Configuration root must be a mapping."
        )

    try:
        protocol = _mapping(raw, "protocol")
        dataset = _mapping(raw, "dataset")
        template_partition = _mapping(
            raw,
            "template_partition",
        )
        seen_item_test = _mapping(
            raw,
            "seen_item_test",
        )

        if protocol.get("version") != "0.2.0":
            raise SeenItemPartitionError(
                "Protocol version must be 0.2.0."
            )

        if (
            template_partition.get("method")
            != "deterministic_group_partition"
        ):
            raise SeenItemPartitionError(
                "Partition method must be deterministic."
            )

        return SeenItemPartitionConfig(
            total_event_count=dataset["total_event_count"],
            calibration_fraction=template_partition[
                "calibration_fraction"
            ],
            partition_seed=template_partition[
                "partition_seed"
            ],
            grouping_field=template_partition[
                "grouping_field"
            ],
            prohibit_template_overlap=template_partition[
                "prohibit_template_overlap"
            ],
            prohibit_random_row_split=template_partition[
                "prohibit_random_row_split"
            ],
            require_training_target_reachability=(
                seen_item_test[
                    "require_training_target_reachability"
                ]
            ),
        )
    except KeyError as exc:
        raise SeenItemPartitionError(
            f"Missing configuration field: {exc.args[0]}"
        ) from exc


def create_seen_item_partition(
    dataset: SyntheticDataset,
    config: SeenItemPartitionConfig,
) -> SeenItemPartition:
    """Partition validation templates into calibration and final test."""

    if not isinstance(dataset, SyntheticDataset):
        raise SeenItemPartitionError(
            "dataset must be a SyntheticDataset."
        )

    if not isinstance(config, SeenItemPartitionConfig):
        raise SeenItemPartitionError(
            "config must be a SeenItemPartitionConfig."
        )

    if len(dataset.events) != config.total_event_count:
        raise SeenItemPartitionError(
            "Dataset event count differs from the locked protocol."
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

    if not training_events or not validation_events:
        raise SeenItemPartitionError(
            "Training and validation events are required."
        )

    training_target_ids = tuple(
        sorted(
            {
                event.target_item_id
                for event in training_events
            }
        )
    )
    training_targets = set(training_target_ids)

    validation_template_ids = tuple(
        sorted(
            {
                event.template_id
                for event in validation_events
            }
        )
    )

    if len(validation_template_ids) < 2:
        raise SeenItemPartitionError(
            "At least two validation templates are required."
        )

    rng = np.random.default_rng(config.partition_seed)
    permuted_templates = tuple(
        int(value)
        for value in rng.permutation(
            validation_template_ids
        )
    )

    calibration_count = int(
        round(
            len(validation_template_ids)
            * config.calibration_fraction
        )
    )

    if (
        calibration_count < 1
        or calibration_count >= len(validation_template_ids)
    ):
        raise SeenItemPartitionError(
            "Calibration fraction creates an empty partition."
        )

    calibration_template_ids = tuple(
        sorted(permuted_templates[:calibration_count])
    )
    seen_item_test_template_ids = tuple(
        sorted(permuted_templates[calibration_count:])
    )

    calibration_templates = set(
        calibration_template_ids
    )
    test_templates = set(
        seen_item_test_template_ids
    )

    calibration_events = tuple(
        event
        for event in validation_events
        if event.template_id in calibration_templates
    )
    raw_test_events = tuple(
        event
        for event in validation_events
        if event.template_id in test_templates
    )

    seen_item_test_events = tuple(
        event
        for event in raw_test_events
        if event.target_item_id in training_targets
    )
    unreachable_test_event_ids = tuple(
        event.event_id
        for event in raw_test_events
        if event.target_item_id not in training_targets
    )

    fit_dataset = SyntheticDataset(
        odor_targets=dataset.odor_targets,
        events=training_events + calibration_events,
        generator_version=dataset.generator_version,
        generator_seed=dataset.generator_seed,
        ood_seed=dataset.ood_seed,
    )

    return SeenItemPartition(
        calibration_template_ids=(
            calibration_template_ids
        ),
        seen_item_test_template_ids=(
            seen_item_test_template_ids
        ),
        calibration_events=calibration_events,
        seen_item_test_events=seen_item_test_events,
        unreachable_test_event_ids=(
            unreachable_test_event_ids
        ),
        raw_seen_item_test_event_count=len(
            raw_test_events
        ),
        training_target_ids=training_target_ids,
        fit_dataset=fit_dataset,
        partition_seed=config.partition_seed,
    )


def _mapping(
    parent: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = parent.get(key)

    if not isinstance(value, Mapping):
        raise SeenItemPartitionError(
            f"{key} must be a mapping."
        )

    return value
