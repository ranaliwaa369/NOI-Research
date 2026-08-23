"""Tests for paired graded-OOD transformations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.amendment_config import (
    load_amendment_configuration,
)
from src.evaluation.graded_ood import (
    EXPECTED_TIER_SEEDS,
    EXPECTED_TIER_STRENGTHS,
    GradedOODError,
    GradedOODEvent,
    OODTier,
    OODTierSpecification,
    create_paired_graded_ood_views,
    tier_specifications_from_amendment,
)
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticEvent,
)


AMENDMENT_PATH = Path("configs/protocol_amendment_v0.2.yaml")
PROTOCOL_PATH = Path("configs/research_protocol.yaml")


@pytest.fixture(scope="module")
def amendment() -> dict:
    """Load the validated preregistered amendment."""

    return load_amendment_configuration(
        AMENDMENT_PATH,
        PROTOCOL_PATH,
    )


@pytest.fixture
def identity_event() -> SyntheticEvent:
    """Return a deterministic identity-distribution observation."""

    return SyntheticEvent(
        event_id="latent-001-identity",
        split=SplitLabel.VALIDATION,
        template_id=17,
        target_item_id="odor-0042",
        target_family_id=4,
        text_vector=(1.0, 0.0, 0.0),
        image_vector=(0.0, 1.0, 0.0),
        audio_vector=(0.0, 0.0, 1.0),
    )


@pytest.fixture
def severe_event() -> SyntheticEvent:
    """Return the paired severe-OOD observation."""

    return SyntheticEvent(
        event_id="latent-001-severe",
        split=SplitLabel.OOD_TEST,
        template_id=17,
        target_item_id="odor-0042",
        target_family_id=4,
        text_vector=(0.0, 1.0, 0.0),
        image_vector=(0.0, 0.0, 1.0),
        audio_vector=(1.0, 0.0, 0.0),
    )


@pytest.fixture
def views(
    identity_event: SyntheticEvent,
    severe_event: SyntheticEvent,
    amendment: dict,
) -> tuple[GradedOODEvent, ...]:
    """Create one complete set of paired OOD views."""

    return create_paired_graded_ood_views(
        identity_event,
        severe_event,
        amendment,
        latent_event_id="latent-001",
    )


def test_locked_tier_specifications_load(
    amendment: dict,
) -> None:
    """The amendment must yield the three locked tiers."""

    specifications = tier_specifications_from_amendment(
        amendment
    )

    assert tuple(
        specification.tier
        for specification in specifications
    ) == (
        OODTier.MILD,
        OODTier.MODERATE,
        OODTier.SEVERE,
    )

    assert tuple(
        specification.shift_strength
        for specification in specifications
    ) == (
        0.25,
        0.50,
        1.00,
    )

    assert tuple(
        specification.seed
        for specification in specifications
    ) == (
        7001,
        8001,
        9001,
    )


def test_three_paired_views_are_created(
    views: tuple[GradedOODEvent, ...],
) -> None:
    """Each latent event must produce exactly three severity views."""

    assert len(views) == 3
    assert tuple(view.tier for view in views) == (
        OODTier.MILD,
        OODTier.MODERATE,
        OODTier.SEVERE,
    )


def test_views_preserve_locked_ground_truth(
    views: tuple[GradedOODEvent, ...],
) -> None:
    """All tiers must preserve target and context ground truth."""

    assert {
        view.latent_event_id
        for view in views
    } == {"latent-001"}

    assert {
        view.target_item_id
        for view in views
    } == {"odor-0042"}

    assert {
        view.target_family_id
        for view in views
    } == {4}

    assert {
        view.template_id
        for view in views
    } == {17}


def test_observed_identifiers_are_unique_and_auditable(
    views: tuple[GradedOODEvent, ...],
) -> None:
    """Observed IDs must explicitly include their severity tier."""

    assert tuple(
        view.observed_event_id
        for view in views
    ) == (
        "latent-001::mild",
        "latent-001::moderate",
        "latent-001::severe",
    )

    assert len({
        view.observed_event_id
        for view in views
    }) == 3


def test_tier_metadata_matches_preregistration(
    views: tuple[GradedOODEvent, ...],
) -> None:
    """Every generated view must retain its locked metadata."""

    for view in views:
        assert (
            view.shift_strength
            == EXPECTED_TIER_STRENGTHS[view.tier.value]
        )
        assert (
            view.tier_seed
            == EXPECTED_TIER_SEEDS[view.tier.value]
        )


def test_all_generated_vectors_are_unit_normalized(
    views: tuple[GradedOODEvent, ...],
) -> None:
    """Every transformed modality must have unit L2 norm."""

    for view in views:
        for vector in (
            view.text_vector,
            view.image_vector,
            view.audio_vector,
        ):
            assert np.linalg.norm(vector) == pytest.approx(
                1.0,
                abs=1e-12,
            )


def test_severe_view_matches_normalized_severe_observation(
    views: tuple[GradedOODEvent, ...],
    severe_event: SyntheticEvent,
) -> None:
    """Strength 1.0 must preserve the original severe direction."""

    severe_view = views[2]

    assert severe_view.text_vector == pytest.approx(
        severe_event.text_vector
    )
    assert severe_view.image_vector == pytest.approx(
        severe_event.image_vector
    )
    assert severe_view.audio_vector == pytest.approx(
        severe_event.audio_vector
    )


def test_mild_text_vector_uses_locked_interpolation(
    views: tuple[GradedOODEvent, ...],
) -> None:
    """The mild view must apply the registered 0.25 blend."""

    expected = np.asarray(
        (0.75, 0.25, 0.0),
        dtype=np.float64,
    )
    expected = expected / np.linalg.norm(expected)

    assert views[0].text_vector == pytest.approx(
        expected
    )


def test_moderate_text_vector_uses_locked_interpolation(
    views: tuple[GradedOODEvent, ...],
) -> None:
    """The moderate view must apply the registered 0.50 blend."""

    expected = np.asarray(
        (0.50, 0.50, 0.0),
        dtype=np.float64,
    )
    expected = expected / np.linalg.norm(expected)

    assert views[1].text_vector == pytest.approx(
        expected
    )


def test_generation_is_deterministic(
    identity_event: SyntheticEvent,
    severe_event: SyntheticEvent,
    amendment: dict,
    views: tuple[GradedOODEvent, ...],
) -> None:
    """Repeated generation must produce identical immutable views."""

    repeated = create_paired_graded_ood_views(
        identity_event,
        severe_event,
        amendment,
        latent_event_id="latent-001",
    )

    assert repeated == views


def test_default_latent_id_uses_identity_event_id(
    identity_event: SyntheticEvent,
    severe_event: SyntheticEvent,
    amendment: dict,
) -> None:
    """The identity event ID is the deterministic default latent ID."""

    generated = create_paired_graded_ood_views(
        identity_event,
        severe_event,
        amendment,
    )

    assert all(
        view.latent_event_id == identity_event.event_id
        for view in generated
    )


def test_generated_views_are_immutable(
    views: tuple[GradedOODEvent, ...],
) -> None:
    """A generated OOD record cannot be silently modified."""

    with pytest.raises(FrozenInstanceError):
        views[0].shift_strength = 0.9  # type: ignore[misc]


@pytest.mark.parametrize(
    ("tier", "invalid_strength", "seed"),
    (
        (OODTier.MILD, 0.10, 7001),
        (OODTier.MODERATE, 0.75, 8001),
        (OODTier.SEVERE, 0.90, 9001),
        (OODTier.MILD, float("nan"), 7001),
        (OODTier.MODERATE, float("inf"), 8001),
        (OODTier.SEVERE, True, 9001),
    ),
)
def test_unregistered_strength_is_rejected(
    tier: OODTier,
    invalid_strength,
    seed: int,
) -> None:
    """Only preregistered transformation strengths are valid."""

    with pytest.raises(
        GradedOODError,
        match="shift_strength",
    ):
        OODTierSpecification(
            tier=tier,
            shift_strength=invalid_strength,
            seed=seed,
        )


@pytest.mark.parametrize(
    ("tier", "strength", "invalid_seed"),
    (
        (OODTier.MILD, 0.25, 1),
        (OODTier.MODERATE, 0.50, 2),
        (OODTier.SEVERE, 1.00, 3),
        (OODTier.MILD, 0.25, -1),
        (OODTier.MODERATE, 0.50, True),
        (OODTier.SEVERE, 1.00, 9001.0),
    ),
)
def test_unregistered_seed_is_rejected(
    tier: OODTier,
    strength: float,
    invalid_seed,
) -> None:
    """Only preregistered tier seeds are valid."""

    with pytest.raises(
        GradedOODError,
        match="seed",
    ):
        OODTierSpecification(
            tier=tier,
            shift_strength=strength,
            seed=invalid_seed,
        )


def test_mismatched_target_item_is_rejected(
    identity_event: SyntheticEvent,
    severe_event: SyntheticEvent,
    amendment: dict,
) -> None:
    """Paired views cannot refer to different odor targets."""

    invalid = SyntheticEvent(
        event_id=severe_event.event_id,
        split=severe_event.split,
        template_id=severe_event.template_id,
        target_item_id="odor-0099",
        target_family_id=severe_event.target_family_id,
        text_vector=severe_event.text_vector,
        image_vector=severe_event.image_vector,
        audio_vector=severe_event.audio_vector,
    )

    with pytest.raises(
        GradedOODError,
        match="target_item_id",
    ):
        create_paired_graded_ood_views(
            identity_event,
            invalid,
            amendment,
        )


def test_mismatched_target_family_is_rejected(
    identity_event: SyntheticEvent,
    severe_event: SyntheticEvent,
    amendment: dict,
) -> None:
    """Paired views cannot refer to different odor families."""

    invalid = SyntheticEvent(
        event_id=severe_event.event_id,
        split=severe_event.split,
        template_id=severe_event.template_id,
        target_item_id=severe_event.target_item_id,
        target_family_id=9,
        text_vector=severe_event.text_vector,
        image_vector=severe_event.image_vector,
        audio_vector=severe_event.audio_vector,
    )

    with pytest.raises(
        GradedOODError,
        match="target_family_id",
    ):
        create_paired_graded_ood_views(
            identity_event,
            invalid,
            amendment,
        )


def test_mismatched_template_is_rejected(
    identity_event: SyntheticEvent,
    severe_event: SyntheticEvent,
    amendment: dict,
) -> None:
    """Paired views must preserve the context template."""

    invalid = SyntheticEvent(
        event_id=severe_event.event_id,
        split=severe_event.split,
        template_id=99,
        target_item_id=severe_event.target_item_id,
        target_family_id=severe_event.target_family_id,
        text_vector=severe_event.text_vector,
        image_vector=severe_event.image_vector,
        audio_vector=severe_event.audio_vector,
    )

    with pytest.raises(
        GradedOODError,
        match="template_id",
    ):
        create_paired_graded_ood_views(
            identity_event,
            invalid,
            amendment,
        )


def test_mismatched_vector_dimension_is_rejected(
    identity_event: SyntheticEvent,
    severe_event: SyntheticEvent,
    amendment: dict,
) -> None:
    """Paired modality dimensions must remain identical."""

    invalid = SyntheticEvent(
        event_id=severe_event.event_id,
        split=severe_event.split,
        template_id=severe_event.template_id,
        target_item_id=severe_event.target_item_id,
        target_family_id=severe_event.target_family_id,
        text_vector=(0.0, 1.0, 0.0, 0.0),
        image_vector=(0.0, 0.0, 1.0, 0.0),
        audio_vector=(1.0, 0.0, 0.0, 0.0),
    )

    with pytest.raises(
        GradedOODError,
        match="dimensions must match",
    ):
        create_paired_graded_ood_views(
            identity_event,
            invalid,
            amendment,
        )


def test_empty_latent_identifier_is_rejected(
    identity_event: SyntheticEvent,
    severe_event: SyntheticEvent,
    amendment: dict,
) -> None:
    """Every paired unit must have an auditable latent ID."""

    with pytest.raises(
        GradedOODError,
        match="nonempty string",
    ):
        create_paired_graded_ood_views(
            identity_event,
            severe_event,
            amendment,
            latent_event_id=" ",
        )


def test_zero_norm_blend_is_rejected(
    amendment: dict,
) -> None:
    """Opposite vectors cannot produce a zero-norm moderate view."""

    identity = SyntheticEvent(
        event_id="identity",
        split=SplitLabel.VALIDATION,
        template_id=1,
        target_item_id="odor-0001",
        target_family_id=1,
        text_vector=(1.0, 0.0),
        image_vector=(1.0, 0.0),
        audio_vector=(1.0, 0.0),
    )

    severe = SyntheticEvent(
        event_id="severe",
        split=SplitLabel.OOD_TEST,
        template_id=1,
        target_item_id="odor-0001",
        target_family_id=1,
        text_vector=(-1.0, 0.0),
        image_vector=(-1.0, 0.0),
        audio_vector=(-1.0, 0.0),
    )

    with pytest.raises(
        GradedOODError,
        match="zero or invalid norm",
    ):
        create_paired_graded_ood_views(
            identity,
            severe,
            amendment,
        )


def test_invalid_amendment_type_is_rejected() -> None:
    """Tier extraction requires a validated dictionary."""

    with pytest.raises(
        GradedOODError,
        match="dictionary",
    ):
        tier_specifications_from_amendment(
            "invalid"  # type: ignore[arg-type]
        )
