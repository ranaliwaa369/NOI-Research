"""Tests for NOI v0.3 split and latent-event generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import inspect
import math

import pytest

from src.evaluation.multisensory_records import (
    LatentMultisensoryEvent,
    MultisensorySplit,
    MultisensoryTarget,
    SupportRegime,
)
from src.evaluation.noi_v0_3_generator import (
    NOIV03GenerationConfig,
    NOIV03GenerationError,
    NOIV03GenerationProvenance,
    NOIV03GenerationResult,
    ReachabilityMetadata,
    generate_noi_v0_3_events,
)


def make_config(
    **changes: object,
) -> NOIV03GenerationConfig:
    """Return one reduced-size configuration preserving protocol ratios."""

    values: dict[str, object] = {
        "seed": 1301,
        "train_event_count": 70,
        "validation_event_count": 10,
        "final_test_event_count": 20,
        "validation_seen_item_count": 4,
        "validation_known_family_unseen_item_count": 3,
        "validation_unseen_family_count": 3,
        "final_seen_item_count": 8,
        "final_known_family_unseen_item_count": 6,
        "final_unseen_family_count": 6,
        "known_family_count": 4,
        "training_items_per_family": 4,
        "withheld_items_per_known_family": 2,
        "validation_unknown_family_count": 2,
        "final_unknown_family_count": 2,
        "items_per_unknown_family": 3,
        "generator_version": "0.3.0-feasibility",
        "feasibility_only": True,
    }
    values.update(changes)

    return NOIV03GenerationConfig(**values)


def events_for(
    result: NOIV03GenerationResult,
    split: MultisensorySplit,
) -> tuple[LatentMultisensoryEvent, ...]:
    """Return all latent events in one split."""

    return tuple(
        event
        for event in result.latent_events
        if event.split is split
    )


def support_counts(
    events: tuple[LatentMultisensoryEvent, ...],
) -> Counter[SupportRegime]:
    """Count support regimes in one event collection."""

    return Counter(event.support_regime for event in events)


def test_same_configuration_is_exactly_deterministic() -> None:
    """One seed and configuration produce identical records."""

    config = make_config()

    first = generate_noi_v0_3_events(config)
    second = generate_noi_v0_3_events(config)

    assert first == second


def test_different_seed_changes_generated_content() -> None:
    """Independent feasibility seeds produce different vectors."""

    first = generate_noi_v0_3_events(make_config(seed=1301))
    second = generate_noi_v0_3_events(make_config(seed=1302))

    assert first.latent_events != second.latent_events


def test_reduced_split_allocation_is_exact() -> None:
    """Feasibility mode preserves the declared 70/10/20 allocation."""

    result = generate_noi_v0_3_events(make_config())

    assert len(events_for(result, MultisensorySplit.TRAIN)) == 70
    assert len(events_for(result, MultisensorySplit.VALIDATION)) == 10
    assert len(events_for(result, MultisensorySplit.FINAL_TEST)) == 20


def test_full_protocol_allocation_can_be_declared() -> None:
    """The production configuration accepts 7000/1000/2000 events."""

    config = replace(
        make_config(),
        train_event_count=7000,
        validation_event_count=1000,
        final_test_event_count=2000,
        validation_seen_item_count=400,
        validation_known_family_unseen_item_count=300,
        validation_unseen_family_count=300,
        final_seen_item_count=800,
        final_known_family_unseen_item_count=600,
        final_unseen_family_count=600,
        generator_version="0.3.0",
        feasibility_only=False,
    )

    assert config.train_event_count == 7000
    assert config.validation_event_count == 1000
    assert config.final_test_event_count == 2000
    assert config.final_seen_item_count == 800
    assert config.final_known_family_unseen_item_count == 600
    assert config.final_unseen_family_count == 600


def test_final_test_support_allocation_is_exact() -> None:
    """Final-test support follows the scaled 8/6/6 allocation."""

    result = generate_noi_v0_3_events(make_config())
    final_events = events_for(
        result,
        MultisensorySplit.FINAL_TEST,
    )

    assert support_counts(final_events) == Counter(
        {
            SupportRegime.SEEN_ITEM: 8,
            SupportRegime.KNOWN_FAMILY_UNSEEN_ITEM: 6,
            SupportRegime.UNSEEN_FAMILY: 6,
        },
    )


def test_validation_support_allocation_is_exact() -> None:
    """Validation contains each support regime for threshold derivation."""

    result = generate_noi_v0_3_events(make_config())
    validation_events = events_for(
        result,
        MultisensorySplit.VALIDATION,
    )

    assert support_counts(validation_events) == Counter(
        {
            SupportRegime.SEEN_ITEM: 4,
            SupportRegime.KNOWN_FAMILY_UNSEEN_ITEM: 3,
            SupportRegime.UNSEEN_FAMILY: 3,
        },
    )


def test_training_events_use_development_support_only() -> None:
    """Training records cannot carry confirmatory support labels."""

    result = generate_noi_v0_3_events(make_config())
    training_events = events_for(
        result,
        MultisensorySplit.TRAIN,
    )

    assert {
        event.support_regime
        for event in training_events
    } == {SupportRegime.DEVELOPMENT}


def test_seen_item_events_are_exactly_reachable_from_training() -> None:
    """Every seen-item evaluation target occurred in training."""

    result = generate_noi_v0_3_events(make_config())
    training_item_ids = {
        event.target_item_id
        for event in events_for(
            result,
            MultisensorySplit.TRAIN,
        )
    }

    evaluation_seen_events = tuple(
        event
        for event in result.latent_events
        if event.support_regime is SupportRegime.SEEN_ITEM
    )

    assert evaluation_seen_events
    assert all(
        event.target_item_id in training_item_ids
        for event in evaluation_seen_events
    )


def test_known_family_unseen_items_have_no_exact_item_leakage() -> None:
    """Known-family targets share families but never training item IDs."""

    result = generate_noi_v0_3_events(make_config())
    training_events = events_for(
        result,
        MultisensorySplit.TRAIN,
    )
    training_item_ids = {
        event.target_item_id
        for event in training_events
    }
    training_family_ids = {
        event.target_family_id
        for event in training_events
    }

    withheld_events = tuple(
        event
        for event in result.latent_events
        if (
            event.support_regime
            is SupportRegime.KNOWN_FAMILY_UNSEEN_ITEM
        )
    )

    assert withheld_events
    assert all(
        event.target_item_id not in training_item_ids
        for event in withheld_events
    )
    assert all(
        event.target_family_id in training_family_ids
        for event in withheld_events
    )


def test_unseen_family_events_have_no_training_family_leakage() -> None:
    """Unknown-family evaluation events use entirely held-out families."""

    result = generate_noi_v0_3_events(make_config())
    training_family_ids = {
        event.target_family_id
        for event in events_for(
            result,
            MultisensorySplit.TRAIN,
        )
    }

    unseen_events = tuple(
        event
        for event in result.latent_events
        if event.support_regime is SupportRegime.UNSEEN_FAMILY
    )

    assert unseen_events
    assert all(
        event.target_family_id not in training_family_ids
        for event in unseen_events
    )


def test_validation_and_final_unknown_families_are_disjoint() -> None:
    """Validation threshold selection cannot inspect final unknown families."""

    result = generate_noi_v0_3_events(make_config())

    validation_unknown = {
        event.target_family_id
        for event in events_for(
            result,
            MultisensorySplit.VALIDATION,
        )
        if event.support_regime is SupportRegime.UNSEEN_FAMILY
    }
    final_unknown = {
        event.target_family_id
        for event in events_for(
            result,
            MultisensorySplit.FINAL_TEST,
        )
        if event.support_regime is SupportRegime.UNSEEN_FAMILY
    }

    assert validation_unknown
    assert final_unknown
    assert validation_unknown.isdisjoint(final_unknown)


def test_all_identifiers_are_unique() -> None:
    """Target and latent-event identifiers are deterministic and unique."""

    result = generate_noi_v0_3_events(make_config())

    target_ids = [
        target.item_id
        for target in result.targets
    ]
    event_ids = [
        event.latent_event_id
        for event in result.latent_events
    ]

    assert len(target_ids) == len(set(target_ids))
    assert len(event_ids) == len(set(event_ids))


def test_all_event_targets_exist_and_family_assignments_match() -> None:
    """Every event resolves to one target with unchanged ground truth."""

    result = generate_noi_v0_3_events(make_config())
    targets = {
        target.item_id: target
        for target in result.targets
    }

    for event in result.latent_events:
        assert event.target_item_id in targets
        assert (
            targets[event.target_item_id].family_id
            == event.target_family_id
        )


def test_vectors_have_protocol_dimensions_and_finite_values() -> None:
    """Generated latent evidence uses 16 odor and 8 touch dimensions."""

    result = generate_noi_v0_3_events(make_config())

    assert all(
        isinstance(target, MultisensoryTarget)
        for target in result.targets
    )

    for target in result.targets:
        assert len(target.olfactory_prototype) == 16
        assert len(target.tactile_prototype) == 8
        assert all(
            math.isfinite(value)
            for value in target.olfactory_prototype
        )
        assert all(
            math.isfinite(value)
            for value in target.tactile_prototype
        )

    for event in result.latent_events:
        assert len(event.olfactory_vector) == 16
        assert len(event.tactile_vector) == 8
        assert all(
            math.isfinite(value)
            for value in event.olfactory_vector
        )
        assert all(
            math.isfinite(value)
            for value in event.tactile_vector
        )


def test_reachability_metadata_matches_generated_records() -> None:
    """Exported reachability metadata is auditable against events."""

    result = generate_noi_v0_3_events(make_config())
    metadata = result.reachability

    assert isinstance(metadata, ReachabilityMetadata)

    training_events = events_for(
        result,
        MultisensorySplit.TRAIN,
    )

    assert set(metadata.training_item_ids) == {
        event.target_item_id
        for event in training_events
    }
    assert set(metadata.training_family_ids) == {
        event.target_family_id
        for event in training_events
    }

    assert set(
        metadata.validation_unknown_family_ids,
    ).isdisjoint(
        metadata.final_test_unknown_family_ids,
    )


def test_provenance_reports_all_allocations() -> None:
    """Generation provenance exposes the prespecified split counts."""

    config = make_config()
    result = generate_noi_v0_3_events(config)
    provenance = result.provenance

    assert isinstance(provenance, NOIV03GenerationProvenance)
    assert provenance.generator_version == config.generator_version
    assert provenance.seed == config.seed
    assert provenance.train_event_count == 70
    assert provenance.validation_event_count == 10
    assert provenance.final_test_event_count == 20
    assert provenance.final_support_allocation == (8, 6, 6)
    assert provenance.feasibility_only is True
    assert provenance.test_labels_used_for_generation is False


def test_result_is_an_explicit_immutable_record() -> None:
    """The generator returns a stable immutable research artifact."""

    result = generate_noi_v0_3_events(make_config())

    assert isinstance(result, NOIV03GenerationResult)

    with pytest.raises(AttributeError):
        result.targets = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", True),
        ("seed", -1),
        ("train_event_count", 0),
        ("validation_event_count", 0),
        ("final_test_event_count", 0),
        ("known_family_count", 0),
        ("training_items_per_family", 0),
        ("withheld_items_per_known_family", 0),
        ("validation_unknown_family_count", 0),
        ("final_unknown_family_count", 0),
        ("items_per_unknown_family", 0),
    ),
)
def test_invalid_integer_configuration_is_rejected(
    field: str,
    value: object,
) -> None:
    """Seeds and allocation parameters obey strict integer contracts."""

    with pytest.raises(NOIV03GenerationError):
        make_config(**{field: value})


def test_validation_support_counts_must_sum_to_validation_total() -> None:
    """Validation allocation cannot silently omit conditions."""

    with pytest.raises(
        NOIV03GenerationError,
        match="validation support allocation",
    ):
        make_config(
            validation_seen_item_count=5,
        )


def test_final_support_counts_must_sum_to_final_total() -> None:
    """Final allocation is completely prespecified."""

    with pytest.raises(
        NOIV03GenerationError,
        match="final-test support allocation",
    ):
        make_config(
            final_seen_item_count=9,
        )


def test_production_mode_requires_exact_protocol_split_counts() -> None:
    """Non-feasibility execution cannot use scaled allocations."""

    with pytest.raises(
        NOIV03GenerationError,
        match="7000/1000/2000",
    ):
        make_config(
            feasibility_only=False,
            generator_version="0.3.0",
        )


def test_production_mode_requires_exact_final_support_counts() -> None:
    """Production final test must preserve the registered 800/600/600."""

    with pytest.raises(
        NOIV03GenerationError,
        match="800/600/600",
    ):
        make_config(
            train_event_count=7000,
            validation_event_count=1000,
            final_test_event_count=2000,
            validation_seen_item_count=400,
            validation_known_family_unseen_item_count=300,
            validation_unseen_family_count=300,
            final_seen_item_count=801,
            final_known_family_unseen_item_count=599,
            final_unseen_family_count=600,
            feasibility_only=False,
            generator_version="0.3.0",
        )


def test_generator_version_is_required() -> None:
    """Every generated artifact has explicit version provenance."""

    with pytest.raises(
        NOIV03GenerationError,
        match="generator_version",
    ):
        make_config(generator_version=" ")


def test_configuration_revalidation_cannot_be_bypassed() -> None:
    """Dataclass replacement re-applies allocation validation."""

    config = make_config()

    with pytest.raises(NOIV03GenerationError):
        replace(
            config,
            final_unseen_family_count=7,
        )


def test_generator_interface_accepts_configuration_only() -> None:
    """No test labels or target IDs may enter through the public API."""

    signature = inspect.signature(generate_noi_v0_3_events)

    assert set(signature.parameters) == {"config"}
