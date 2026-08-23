"""Dataset assembly for paired graded-OOD synthetic evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.evaluation.graded_ood import (
    GradedOODError,
    GradedOODEvent,
    OODTier,
    create_paired_graded_ood_views,
)
from src.evaluation.synthetic_records import SyntheticEvent


class GradedOODGeneratorError(ValueError):
    """Raised when a paired graded-OOD dataset cannot be assembled."""


@dataclass(frozen=True)
class PairedOODSource:
    """Identity and severe observations for one latent evaluation unit."""

    latent_event_id: str
    identity_event: SyntheticEvent
    severe_event: SyntheticEvent

    def __post_init__(self) -> None:
        if (
            not isinstance(self.latent_event_id, str)
            or not self.latent_event_id.strip()
        ):
            raise GradedOODGeneratorError(
                "latent_event_id must be a nonempty string."
            )

        if not isinstance(self.identity_event, SyntheticEvent):
            raise GradedOODGeneratorError(
                "identity_event must be a SyntheticEvent."
            )

        if not isinstance(self.severe_event, SyntheticEvent):
            raise GradedOODGeneratorError(
                "severe_event must be a SyntheticEvent."
            )

        if (
            self.identity_event.target_item_id
            != self.severe_event.target_item_id
        ):
            raise GradedOODGeneratorError(
                "Paired source events must preserve target_item_id."
            )

        if (
            self.identity_event.target_family_id
            != self.severe_event.target_family_id
        ):
            raise GradedOODGeneratorError(
                "Paired source events must preserve target_family_id."
            )

        if (
            self.identity_event.template_id
            != self.severe_event.template_id
        ):
            raise GradedOODGeneratorError(
                "Paired source events must preserve template_id."
            )


@dataclass(frozen=True)
class GradedOODDataset:
    """Validated collection of paired mild, moderate, and severe views."""

    events: tuple[GradedOODEvent, ...]
    latent_event_count: int
    observed_event_count: int
    views_per_latent_event: int
    paired_analysis_unit: str
    amendment_id: str
    amendment_version: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.latent_event_count, bool)
            or not isinstance(self.latent_event_count, int)
            or self.latent_event_count < 1
        ):
            raise GradedOODGeneratorError(
                "latent_event_count must be a positive integer."
            )

        if (
            isinstance(self.observed_event_count, bool)
            or not isinstance(self.observed_event_count, int)
            or self.observed_event_count < 1
        ):
            raise GradedOODGeneratorError(
                "observed_event_count must be a positive integer."
            )

        if self.views_per_latent_event != 3:
            raise GradedOODGeneratorError(
                "views_per_latent_event must be 3."
            )

        expected_count = (
            self.latent_event_count
            * self.views_per_latent_event
        )

        if self.observed_event_count != expected_count:
            raise GradedOODGeneratorError(
                "observed_event_count must equal "
                "latent_event_count multiplied by 3."
            )

        if len(self.events) != self.observed_event_count:
            raise GradedOODGeneratorError(
                "The event tuple length must match observed_event_count."
            )

        if self.paired_analysis_unit != "latent_event_id":
            raise GradedOODGeneratorError(
                "paired_analysis_unit must be latent_event_id."
            )

        if self.amendment_id != "NOI-PROTOCOL-AMENDMENT-0.2":
            raise GradedOODGeneratorError(
                "Unexpected amendment_id."
            )

        if self.amendment_version != "0.2.0":
            raise GradedOODGeneratorError(
                "Unexpected amendment_version."
            )

        observed_ids = [
            event.observed_event_id
            for event in self.events
        ]

        if len(set(observed_ids)) != len(observed_ids):
            raise GradedOODGeneratorError(
                "Observed event identifiers must be unique."
            )

        grouped: dict[str, list[GradedOODEvent]] = {}

        for event in self.events:
            if not isinstance(event, GradedOODEvent):
                raise GradedOODGeneratorError(
                    "Every dataset event must be a GradedOODEvent."
                )

            grouped.setdefault(
                event.latent_event_id,
                [],
            ).append(event)

        if len(grouped) != self.latent_event_count:
            raise GradedOODGeneratorError(
                "The number of latent groups is inconsistent."
            )

        expected_tiers = {
            OODTier.MILD,
            OODTier.MODERATE,
            OODTier.SEVERE,
        }

        for latent_id, group in grouped.items():
            if len(group) != 3:
                raise GradedOODGeneratorError(
                    f"Latent event {latent_id} must have exactly 3 views."
                )

            tiers = {
                event.tier
                for event in group
            }

            if tiers != expected_tiers:
                raise GradedOODGeneratorError(
                    f"Latent event {latent_id} must contain "
                    "mild, moderate, and severe views."
                )

            target_items = {
                event.target_item_id
                for event in group
            }
            target_families = {
                event.target_family_id
                for event in group
            }
            templates = {
                event.template_id
                for event in group
            }

            if len(target_items) != 1:
                raise GradedOODGeneratorError(
                    f"Latent event {latent_id} changes target_item_id."
                )

            if len(target_families) != 1:
                raise GradedOODGeneratorError(
                    f"Latent event {latent_id} changes target_family_id."
                )

            if len(templates) != 1:
                raise GradedOODGeneratorError(
                    f"Latent event {latent_id} changes template_id."
                )

    @property
    def tier_counts(self) -> dict[str, int]:
        """Return observed row counts by severity tier."""

        counts = Counter(
            event.tier.value
            for event in self.events
        )

        return {
            tier.value: counts[tier.value]
            for tier in OODTier
        }

    @property
    def latent_event_ids(self) -> tuple[str, ...]:
        """Return unique latent IDs in deterministic order."""

        return tuple(
            sorted({
                event.latent_event_id
                for event in self.events
            })
        )

    def events_for_tier(
        self,
        tier: OODTier,
    ) -> tuple[GradedOODEvent, ...]:
        """Return one tier in deterministic latent-event order."""

        if not isinstance(tier, OODTier):
            raise GradedOODGeneratorError(
                "tier must be an OODTier."
            )

        return tuple(
            event
            for event in self.events
            if event.tier is tier
        )

    def events_for_latent_id(
        self,
        latent_event_id: str,
    ) -> tuple[GradedOODEvent, ...]:
        """Return all three paired views for one latent unit."""

        if (
            not isinstance(latent_event_id, str)
            or not latent_event_id.strip()
        ):
            raise GradedOODGeneratorError(
                "latent_event_id must be a nonempty string."
            )

        selected = tuple(
            event
            for event in self.events
            if event.latent_event_id == latent_event_id
        )

        if not selected:
            raise GradedOODGeneratorError(
                f"Unknown latent_event_id: {latent_event_id}"
            )

        return selected


def generate_graded_ood_dataset(
    sources: Iterable[PairedOODSource],
    amendment: dict[str, Any],
) -> GradedOODDataset:
    """Generate a deterministic paired graded-OOD dataset.

    The function requires explicit identity/severe source pairs. It never
    invents identity references from target labels, preventing target leakage.
    """

    if isinstance(sources, (str, bytes)):
        raise GradedOODGeneratorError(
            "sources must contain PairedOODSource records."
        )

    try:
        source_tuple = tuple(sources)
    except TypeError as error:
        raise GradedOODGeneratorError(
            "sources must be iterable."
        ) from error

    if not source_tuple:
        raise GradedOODGeneratorError(
            "At least one paired OOD source is required."
        )

    if not isinstance(amendment, dict):
        raise GradedOODGeneratorError(
            "amendment must be a dictionary."
        )

    try:
        amendment_identity = amendment["amendment"]
        ood_policy = amendment[
            "confirmatory_evaluation"
        ]["ood_evaluation"]
    except (KeyError, TypeError) as error:
        raise GradedOODGeneratorError(
            "The amendment lacks required graded-OOD metadata."
        ) from error

    if not isinstance(amendment_identity, dict):
        raise GradedOODGeneratorError(
            "The amendment identity must be a mapping."
        )

    if not isinstance(ood_policy, dict):
        raise GradedOODGeneratorError(
            "The OOD evaluation policy must be a mapping."
        )

    if (
        ood_policy.get("paired_across_severity_tiers")
        is not True
    ):
        raise GradedOODGeneratorError(
            "The amendment must require paired severity evaluation."
        )

    if (
        ood_policy.get("analysis_unit", "latent_event_id")
        != "latent_event_id"
    ):
        raise GradedOODGeneratorError(
            "The analysis unit must be latent_event_id."
        )

    if (
        ood_policy.get("severity_views_per_latent_event")
        != 3
    ):
        raise GradedOODGeneratorError(
            "The amendment must require 3 severity views."
        )

    for source in source_tuple:
        if not isinstance(source, PairedOODSource):
            raise GradedOODGeneratorError(
                "Every source must be a PairedOODSource."
            )

    sorted_sources = tuple(
        sorted(
            source_tuple,
            key=lambda source: source.latent_event_id,
        )
    )

    latent_ids = [
        source.latent_event_id
        for source in sorted_sources
    ]

    if len(set(latent_ids)) != len(latent_ids):
        raise GradedOODGeneratorError(
            "Source latent_event_id values must be unique."
        )

    events: list[GradedOODEvent] = []

    for source in sorted_sources:
        try:
            views = create_paired_graded_ood_views(
                source.identity_event,
                source.severe_event,
                amendment,
                latent_event_id=source.latent_event_id,
            )
        except GradedOODError as error:
            raise GradedOODGeneratorError(
                f"Could not generate latent event "
                f"{source.latent_event_id}: {error}"
            ) from error

        events.extend(views)

    generated_events = tuple(events)
    latent_count = len(sorted_sources)

    return GradedOODDataset(
        events=generated_events,
        latent_event_count=latent_count,
        observed_event_count=len(generated_events),
        views_per_latent_event=3,
        paired_analysis_unit="latent_event_id",
        amendment_id=amendment_identity.get("id"),
        amendment_version=amendment_identity.get("version"),
    )
