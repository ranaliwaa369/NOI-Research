"""Tests for paired graded-OOD dataset assembly."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.evaluation.amendment_config import (
    load_amendment_configuration,
)
from src.evaluation.graded_ood import OODTier
from src.evaluation.graded_ood_generator import (
    GradedOODDataset,
    GradedOODGeneratorError,
    PairedOODSource,
    generate_graded_ood_dataset,
)
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticEvent,
)


AMENDMENT_PATH = Path("configs/protocol_amendment_v0.2.yaml")
PROTOCOL_PATH = Path("configs/research_protocol.yaml")


@pytest.fixture(scope="module")
def amendment() -> dict:
    """Load the validated amendment once."""

    return load_amendment_configuration(
        AMENDMENT_PATH,
        PROTOCOL_PATH,
    )


def make_source(
    index: int,
    *,
    latent_event_id: str | None = None,
) -> PairedOODSource:
    """Create one deterministic matched identity/severe source."""

    target_item_id = f"odor-{index:04d}"
    target_family_id = index % 5
    template_id = index

    identity = SyntheticEvent(
        event_id=f"identity-{index:04d}",
        split=SplitLabel.VALIDATION,
        template_id=template_id,
        target_item_id=target_item_id,
        target_family_id=target_family_id,
        text_vector=(1.0, 0.0, 0.0),
        image_vector=(0.0, 1.0, 0.0),
        audio_vector=(0.0, 0.0, 1.0),
    )

    severe = SyntheticEvent(
        event_id=f"severe-{index:04d}",
        split=SplitLabel.OOD_TEST,
        template_id=template_id,
        target_item_id=target_item_id,
        target_family_id=target_family_id,
        text_vector=(0.0, 1.0, 0.0),
        image_vector=(0.0, 0.0, 1.0),
        audio_vector=(1.0, 0.0, 0.0),
    )

    return PairedOODSource(
        latent_event_id=(
            f"latent-{index:04d}"
            if latent_event_id is None
            else latent_event_id
        ),
        identity_event=identity,
        severe_event=severe,
    )


@pytest.fixture
def sources() -> tuple[PairedOODSource, ...]:
    """Return deliberately unsorted paired sources."""

    return (
        make_source(3),
        make_source(1),
        make_source(2),
    )


@pytest.fixture
def dataset(
    sources: tuple[PairedOODSource, ...],
    amendment: dict,
) -> GradedOODDataset:
    """Generate a small deterministic graded-OOD dataset."""

    return generate_graded_ood_dataset(
        sources,
        amendment,
    )


def test_dataset_has_expected_counts(
    dataset: GradedOODDataset,
) -> None:
    """Three latent units must produce nine observed rows."""

    assert dataset.latent_event_count == 3
    assert dataset.observed_event_count == 9
    assert dataset.views_per_latent_event == 3
    assert len(dataset.events) == 9


def test_dataset_records_amendment_identity(
    dataset: GradedOODDataset,
) -> None:
    """Generated data must record the governing amendment."""

    assert (
        dataset.amendment_id
        == "NOI-PROTOCOL-AMENDMENT-0.2"
    )
    assert dataset.amendment_version == "0.2.0"
    assert dataset.paired_analysis_unit == "latent_event_id"


def test_tier_counts_are_balanced(
    dataset: GradedOODDataset,
) -> None:
    """Each latent unit must contribute once to every tier."""

    assert dataset.tier_counts == {
        "mild": 3,
        "moderate": 3,
        "severe": 3,
    }


def test_latent_identifiers_are_sorted(
    dataset: GradedOODDataset,
) -> None:
    """Latent identifiers must be exposed in deterministic order."""

    assert dataset.latent_event_ids == (
        "latent-0001",
        "latent-0002",
        "latent-0003",
    )


def test_generated_rows_use_deterministic_order(
    dataset: GradedOODDataset,
) -> None:
    """Rows must sort first by latent ID and then locked tier order."""

    assert tuple(
        event.observed_event_id
        for event in dataset.events
    ) == (
        "latent-0001::mild",
        "latent-0001::moderate",
        "latent-0001::severe",
        "latent-0002::mild",
        "latent-0002::moderate",
        "latent-0002::severe",
        "latent-0003::mild",
        "latent-0003::moderate",
        "latent-0003::severe",
    )


@pytest.mark.parametrize(
    ("tier", "expected_count"),
    (
        (OODTier.MILD, 3),
        (OODTier.MODERATE, 3),
        (OODTier.SEVERE, 3),
    ),
)
def test_events_for_tier_returns_complete_partition(
    dataset: GradedOODDataset,
    tier: OODTier,
    expected_count: int,
) -> None:
    """Tier selection must return one row per latent unit."""

    selected = dataset.events_for_tier(tier)

    assert len(selected) == expected_count
    assert all(event.tier is tier for event in selected)
    assert tuple(
        event.latent_event_id
        for event in selected
    ) == dataset.latent_event_ids


def test_events_for_latent_id_returns_three_views(
    dataset: GradedOODDataset,
) -> None:
    """A latent lookup must return its paired severity triplet."""

    selected = dataset.events_for_latent_id(
        "latent-0002"
    )

    assert len(selected) == 3
    assert tuple(event.tier for event in selected) == (
        OODTier.MILD,
        OODTier.MODERATE,
        OODTier.SEVERE,
    )
    assert {
        event.target_item_id
        for event in selected
    } == {"odor-0002"}


def test_unknown_latent_id_is_rejected(
    dataset: GradedOODDataset,
) -> None:
    """A missing latent unit must not return an empty silent result."""

    with pytest.raises(
        GradedOODGeneratorError,
        match="Unknown latent_event_id",
    ):
        dataset.events_for_latent_id(
            "latent-missing"
        )


def test_invalid_tier_lookup_is_rejected(
    dataset: GradedOODDataset,
) -> None:
    """Tier access requires an OODTier value."""

    with pytest.raises(
        GradedOODGeneratorError,
        match="OODTier",
    ):
        dataset.events_for_tier(
            "mild"  # type: ignore[arg-type]
        )


def test_generation_is_deterministic(
    sources: tuple[PairedOODSource, ...],
    amendment: dict,
    dataset: GradedOODDataset,
) -> None:
    """Repeated assembly must produce exactly equal datasets."""

    repeated = generate_graded_ood_dataset(
        reversed(sources),
        amendment,
    )

    assert repeated == dataset


def test_dataset_is_immutable(
    dataset: GradedOODDataset,
) -> None:
    """Generated dataset metadata cannot be silently modified."""

    with pytest.raises(FrozenInstanceError):
        dataset.latent_event_count = 10  # type: ignore[misc]


def test_empty_sources_are_rejected(
    amendment: dict,
) -> None:
    """At least one explicit paired source is required."""

    with pytest.raises(
        GradedOODGeneratorError,
        match="At least one",
    ):
        generate_graded_ood_dataset(
            (),
            amendment,
        )


@pytest.mark.parametrize(
    "invalid_sources",
    (
        "invalid",
        42,
        None,
    ),
)
def test_invalid_source_collection_is_rejected(
    amendment: dict,
    invalid_sources,
) -> None:
    """Source input must be an iterable of paired records."""

    with pytest.raises(
        GradedOODGeneratorError,
        match="sources",
    ):
        generate_graded_ood_dataset(
            invalid_sources,  # type: ignore[arg-type]
            amendment,
        )


def test_non_source_member_is_rejected(
    amendment: dict,
) -> None:
    """Every input member must be a PairedOODSource."""

    with pytest.raises(
        GradedOODGeneratorError,
        match="PairedOODSource",
    ):
        generate_graded_ood_dataset(
            (make_source(1), "invalid"),  # type: ignore[arg-type]
            amendment,
        )


def test_duplicate_latent_ids_are_rejected(
    amendment: dict,
) -> None:
    """Two source pairs cannot claim the same latent unit."""

    sources = (
        make_source(
            1,
            latent_event_id="duplicate",
        ),
        make_source(
            2,
            latent_event_id="duplicate",
        ),
    )

    with pytest.raises(
        GradedOODGeneratorError,
        match="must be unique",
    ):
        generate_graded_ood_dataset(
            sources,
            amendment,
        )


def test_empty_latent_source_id_is_rejected() -> None:
    """A source must have an auditable latent identifier."""

    source = make_source(1)

    with pytest.raises(
        GradedOODGeneratorError,
        match="nonempty string",
    ):
        PairedOODSource(
            latent_event_id=" ",
            identity_event=source.identity_event,
            severe_event=source.severe_event,
        )


def test_mismatched_source_target_is_rejected() -> None:
    """Identity and severe observations must preserve the target."""

    source = make_source(1)

    mismatched_severe = SyntheticEvent(
        event_id="severe-mismatch",
        split=SplitLabel.OOD_TEST,
        template_id=source.severe_event.template_id,
        target_item_id="odor-different",
        target_family_id=source.severe_event.target_family_id,
        text_vector=source.severe_event.text_vector,
        image_vector=source.severe_event.image_vector,
        audio_vector=source.severe_event.audio_vector,
    )

    with pytest.raises(
        GradedOODGeneratorError,
        match="target_item_id",
    ):
        PairedOODSource(
            latent_event_id="latent-mismatch",
            identity_event=source.identity_event,
            severe_event=mismatched_severe,
        )


def test_mismatched_source_family_is_rejected() -> None:
    """Identity and severe observations must preserve family truth."""

    source = make_source(1)

    mismatched_severe = SyntheticEvent(
        event_id="severe-mismatch",
        split=SplitLabel.OOD_TEST,
        template_id=source.severe_event.template_id,
        target_item_id=source.severe_event.target_item_id,
        target_family_id=99,
        text_vector=source.severe_event.text_vector,
        image_vector=source.severe_event.image_vector,
        audio_vector=source.severe_event.audio_vector,
    )

    with pytest.raises(
        GradedOODGeneratorError,
        match="target_family_id",
    ):
        PairedOODSource(
            latent_event_id="latent-mismatch",
            identity_event=source.identity_event,
            severe_event=mismatched_severe,
        )


def test_mismatched_source_template_is_rejected() -> None:
    """Identity and severe observations must preserve template truth."""

    source = make_source(1)

    mismatched_severe = SyntheticEvent(
        event_id="severe-mismatch",
        split=SplitLabel.OOD_TEST,
        template_id=999,
        target_item_id=source.severe_event.target_item_id,
        target_family_id=source.severe_event.target_family_id,
        text_vector=source.severe_event.text_vector,
        image_vector=source.severe_event.image_vector,
        audio_vector=source.severe_event.audio_vector,
    )

    with pytest.raises(
        GradedOODGeneratorError,
        match="template_id",
    ):
        PairedOODSource(
            latent_event_id="latent-mismatch",
            identity_event=source.identity_event,
            severe_event=mismatched_severe,
        )


def test_invalid_amendment_type_is_rejected(
    sources: tuple[PairedOODSource, ...],
) -> None:
    """Dataset generation requires amendment metadata."""

    with pytest.raises(
        GradedOODGeneratorError,
        match="dictionary",
    ):
        generate_graded_ood_dataset(
            sources,
            "invalid",  # type: ignore[arg-type]
        )


def test_unpaired_amendment_policy_is_rejected(
    sources: tuple[PairedOODSource, ...],
    amendment: dict,
) -> None:
    """The generator cannot run if paired analysis is disabled."""

    invalid = deepcopy(amendment)
    invalid[
        "confirmatory_evaluation"
    ]["ood_evaluation"][
        "paired_across_severity_tiers"
    ] = False

    with pytest.raises(
        GradedOODGeneratorError,
        match="paired severity",
    ):
        generate_graded_ood_dataset(
            sources,
            invalid,
        )


def test_wrong_analysis_unit_is_rejected(
    sources: tuple[PairedOODSource, ...],
    amendment: dict,
) -> None:
    """Observed rows cannot replace latent events as analysis units."""

    invalid = deepcopy(amendment)
    invalid[
        "confirmatory_evaluation"
    ]["ood_evaluation"][
        "analysis_unit"
    ] = "observed_row"

    with pytest.raises(
        GradedOODGeneratorError,
        match="latent_event_id",
    ):
        generate_graded_ood_dataset(
            sources,
            invalid,
        )


def test_wrong_view_count_policy_is_rejected(
    sources: tuple[PairedOODSource, ...],
    amendment: dict,
) -> None:
    """The amendment must continue to require three severity views."""

    invalid = deepcopy(amendment)
    invalid[
        "confirmatory_evaluation"
    ]["ood_evaluation"][
        "severity_views_per_latent_event"
    ] = 2

    with pytest.raises(
        GradedOODGeneratorError,
        match="3 severity views",
    ):
        generate_graded_ood_dataset(
            sources,
            invalid,
        )


def test_target_truth_never_enters_feature_vectors(
    dataset: GradedOODDataset,
) -> None:
    """Generated vectors must not contain string target identifiers."""

    for event in dataset.events:
        for vector in (
            event.text_vector,
            event.image_vector,
            event.audio_vector,
        ):
            assert all(
                isinstance(value, float)
                for value in vector
            )
            assert event.target_item_id not in vector
