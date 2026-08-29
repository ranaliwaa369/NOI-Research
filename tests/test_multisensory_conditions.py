"""Tests for NOI v0.3 paired multisensory stress views."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import inspect
import math

import pytest

from src.evaluation.multisensory_conditions import (
    ConditionGenerationConfig,
    ConditionGenerationError,
    ConditionGenerationProvenance,
    ConditionGenerationResult,
    ConflictSource,
    generate_multisensory_condition_views,
)
from src.evaluation.multisensory_records import (
    ConditionLabel,
    MultisensoryConditionView,
    MultisensorySplit,
)
from src.evaluation.noi_v0_3_generator import (
    NOIV03GenerationConfig,
    generate_noi_v0_3_events,
)


def make_event_result():
    """Return a small deterministic v0.3 latent dataset."""

    return generate_noi_v0_3_events(
        NOIV03GenerationConfig(
            seed=1301,
            train_event_count=70,
            validation_event_count=10,
            final_test_event_count=20,
            validation_seen_item_count=4,
            validation_known_family_unseen_item_count=3,
            validation_unseen_family_count=3,
            final_seen_item_count=8,
            final_known_family_unseen_item_count=6,
            final_unseen_family_count=6,
            known_family_count=4,
            training_items_per_family=4,
            withheld_items_per_known_family=2,
            validation_unknown_family_count=2,
            final_unknown_family_count=2,
            items_per_unknown_family=3,
            generator_version="0.3.0-feasibility",
            feasibility_only=True,
        ),
    )


def make_config(
    **changes: object,
) -> ConditionGenerationConfig:
    """Return one locked feasibility condition configuration."""

    values: dict[str, object] = {
        "seed": 1301,
        "odor_noise_scale": 0.25,
        "tactile_noise_scale": 0.20,
        "degraded_quality": 0.40,
        "locked_temporal_offset_steps": 2,
        "generator_version": "0.3.0-feasibility",
    }
    values.update(changes)

    return ConditionGenerationConfig(**values)


def make_result() -> ConditionGenerationResult:
    """Generate paired views for final-test latent events."""

    generated = make_event_result()
    final_events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )

    return generate_multisensory_condition_views(
        latent_events=final_events,
        targets=generated.targets,
        config=make_config(),
    )


def grouped_views(
    result: ConditionGenerationResult,
) -> dict[str, dict[ConditionLabel, MultisensoryConditionView]]:
    """Index views by latent event and condition."""

    grouped: dict[
        str,
        dict[ConditionLabel, MultisensoryConditionView],
    ] = defaultdict(dict)

    for view in result.views:
        grouped[view.latent_event_id][view.condition] = view

    return dict(grouped)


def test_same_inputs_produce_identical_views() -> None:
    """Paired condition generation is deterministic."""

    generated = make_event_result()
    events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )
    config = make_config()

    first = generate_multisensory_condition_views(
        latent_events=events,
        targets=generated.targets,
        config=config,
    )
    second = generate_multisensory_condition_views(
        latent_events=events,
        targets=generated.targets,
        config=config,
    )

    assert first == second


def test_different_condition_seed_changes_stochastic_views() -> None:
    """Noise and conflict selection depend on the condition seed."""

    generated = make_event_result()
    events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )

    first = generate_multisensory_condition_views(
        latent_events=events,
        targets=generated.targets,
        config=make_config(seed=1301),
    )
    second = generate_multisensory_condition_views(
        latent_events=events,
        targets=generated.targets,
        config=make_config(seed=1302),
    )

    assert first.views != second.views


def test_every_latent_event_has_exactly_seven_views() -> None:
    """All prespecified conditions are paired on every latent event."""

    result = make_result()
    grouped = grouped_views(result)

    assert grouped
    assert all(
        set(condition_views) == set(ConditionLabel)
        for condition_views in grouped.values()
    )
    assert all(
        len(condition_views) == 7
        for condition_views in grouped.values()
    )


def test_total_view_count_is_seven_times_event_count() -> None:
    """No condition is silently added or removed."""

    generated = make_event_result()
    final_events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )

    result = generate_multisensory_condition_views(
        latent_events=final_events,
        targets=generated.targets,
        config=make_config(),
    )

    assert len(result.views) == 7 * len(final_events)


def test_each_condition_has_equal_support() -> None:
    """The paired design has one observation per event per condition."""

    result = make_result()
    counts = Counter(view.condition for view in result.views)

    assert set(counts.values()) == {20}


def test_clean_view_preserves_both_modalities_exactly() -> None:
    """Clean observations equal the latent event evidence."""

    generated = make_event_result()
    final_events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )
    event_lookup = {
        event.latent_event_id: event
        for event in final_events
    }

    result = generate_multisensory_condition_views(
        latent_events=final_events,
        targets=generated.targets,
        config=make_config(),
    )

    for view in result.views:
        if view.condition is not ConditionLabel.CLEAN:
            continue

        event = event_lookup[view.latent_event_id]

        assert view.olfactory_vector == event.olfactory_vector
        assert view.tactile_vector == event.tactile_vector
        assert view.olfactory_quality == 1.0
        assert view.tactile_quality == 1.0
        assert view.modality_conflict is False
        assert view.temporal_offset_steps == 0


def test_degraded_odor_changes_only_odor() -> None:
    """Odor degradation cannot alter tactile evidence."""

    result = make_result()
    grouped = grouped_views(result)

    for condition_views in grouped.values():
        clean = condition_views[ConditionLabel.CLEAN]
        degraded = condition_views[ConditionLabel.DEGRADED_ODOR]

        assert degraded.olfactory_vector != clean.olfactory_vector
        assert degraded.tactile_vector == clean.tactile_vector
        assert degraded.olfactory_quality == 0.40
        assert degraded.tactile_quality == 1.0


def test_degraded_touch_changes_only_touch() -> None:
    """Tactile degradation cannot alter olfactory evidence."""

    result = make_result()
    grouped = grouped_views(result)

    for condition_views in grouped.values():
        clean = condition_views[ConditionLabel.CLEAN]
        degraded = condition_views[ConditionLabel.DEGRADED_TOUCH]

        assert degraded.olfactory_vector == clean.olfactory_vector
        assert degraded.tactile_vector != clean.tactile_vector
        assert degraded.olfactory_quality == 1.0
        assert degraded.tactile_quality == 0.40


def test_missing_touch_is_encoded_as_absent() -> None:
    """Missing touch is not represented by a fabricated zero vector."""

    result = make_result()

    for view in result.views:
        if view.condition is ConditionLabel.MISSING_TOUCH:
            assert view.olfactory_vector is not None
            assert view.tactile_vector is None
            assert view.olfactory_quality == 1.0
            assert view.tactile_quality == 0.0


def test_missing_odor_is_encoded_as_absent() -> None:
    """Missing odor is not represented by a fabricated zero vector."""

    result = make_result()

    for view in result.views:
        if view.condition is ConditionLabel.MISSING_ODOR:
            assert view.olfactory_vector is None
            assert view.tactile_vector is not None
            assert view.olfactory_quality == 0.0
            assert view.tactile_quality == 1.0


def test_conflict_uses_touch_from_a_different_family() -> None:
    """Contradictory touch comes from a deterministic cross-family target."""

    generated = make_event_result()
    target_lookup = {
        target.item_id: target
        for target in generated.targets
    }
    final_events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )
    event_lookup = {
        event.latent_event_id: event
        for event in final_events
    }

    result = generate_multisensory_condition_views(
        latent_events=final_events,
        targets=generated.targets,
        config=make_config(),
    )

    source_lookup = {
        source.view_id: source
        for source in result.conflict_sources
    }

    for view in result.views:
        if view.condition is not ConditionLabel.CONTRADICTORY_MODALITIES:
            continue

        event = event_lookup[view.latent_event_id]
        source = source_lookup[view.view_id]
        donor = target_lookup[source.source_item_id]

        assert isinstance(source, ConflictSource)
        assert source.source_family_id != event.target_family_id
        assert donor.family_id == source.source_family_id
        assert view.olfactory_vector == event.olfactory_vector
        assert view.tactile_vector == donor.tactile_prototype
        assert view.modality_conflict is True


def test_temporal_condition_uses_locked_offset_only() -> None:
    """Temporal stress preserves vectors and applies the registered offset."""

    result = make_result()
    grouped = grouped_views(result)

    for condition_views in grouped.values():
        clean = condition_views[ConditionLabel.CLEAN]
        temporal = condition_views[
            ConditionLabel.TEMPORAL_MISALIGNMENT
        ]

        assert temporal.olfactory_vector == clean.olfactory_vector
        assert temporal.tactile_vector == clean.tactile_vector
        assert temporal.olfactory_quality == 1.0
        assert temporal.tactile_quality == 1.0
        assert temporal.temporal_offset_steps == 2


def test_ground_truth_is_preserved_across_all_views() -> None:
    """A stress transformation cannot redefine the prediction target."""

    generated = make_event_result()
    final_events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )
    event_lookup = {
        event.latent_event_id: event
        for event in final_events
    }

    result = generate_multisensory_condition_views(
        latent_events=final_events,
        targets=generated.targets,
        config=make_config(),
    )

    for view in result.views:
        event = event_lookup[view.latent_event_id]

        assert view.target_item_id == event.target_item_id
        assert view.target_family_id == event.target_family_id


def test_support_regimes_are_exported_without_change() -> None:
    """Views cannot change or manufacture support-regime labels."""

    generated = make_event_result()
    final_events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )
    expected = tuple(
        (
            event.latent_event_id,
            event.support_regime,
        )
        for event in final_events
    )

    result = generate_multisensory_condition_views(
        latent_events=final_events,
        targets=generated.targets,
        config=make_config(),
    )

    assert result.support_regimes == expected


def test_view_identifiers_and_pairs_are_unique() -> None:
    """Every latent-event/condition pair has one unique view."""

    result = make_result()

    view_ids = [
        view.view_id
        for view in result.views
    ]
    pairs = [
        (
            view.latent_event_id,
            view.condition,
        )
        for view in result.views
    ]

    assert len(view_ids) == len(set(view_ids))
    assert len(pairs) == len(set(pairs))


def test_all_generated_vectors_are_finite_and_bounded() -> None:
    """Stress transformations retain valid numerical representations."""

    result = make_result()

    for view in result.views:
        for vector in (
            view.olfactory_vector,
            view.tactile_vector,
        ):
            if vector is None:
                continue

            assert all(math.isfinite(value) for value in vector)
            assert all(-1.0 <= value <= 1.0 for value in vector)


def test_provenance_records_locked_condition_parameters() -> None:
    """The paired-view artifact reports its transformation settings."""

    result = make_result()
    provenance = result.provenance

    assert isinstance(provenance, ConditionGenerationProvenance)
    assert provenance.generator_version == "0.3.0-feasibility"
    assert provenance.seed == 1301
    assert provenance.conditions == tuple(ConditionLabel)
    assert provenance.odor_noise_scale == 0.25
    assert provenance.tactile_noise_scale == 0.20
    assert provenance.degraded_quality == 0.40
    assert provenance.locked_temporal_offset_steps == 2
    assert provenance.ground_truth_changed is False
    assert provenance.support_regimes_changed is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", True),
        ("seed", -1),
        ("locked_temporal_offset_steps", 0),
        ("locked_temporal_offset_steps", True),
    ),
)
def test_invalid_integer_configuration_is_rejected(
    field: str,
    value: object,
) -> None:
    """Seed and temporal offset obey strict integer contracts."""

    with pytest.raises(ConditionGenerationError):
        make_config(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("odor_noise_scale", 0.0),
        ("odor_noise_scale", 1.1),
        ("odor_noise_scale", float("nan")),
        ("tactile_noise_scale", 0.0),
        ("tactile_noise_scale", 1.1),
        ("tactile_noise_scale", float("inf")),
        ("degraded_quality", 0.0),
        ("degraded_quality", 1.0),
    ),
)
def test_invalid_fraction_is_rejected(
    field: str,
    value: object,
) -> None:
    """Noise and degraded quality must be strictly between 0 and 1."""

    with pytest.raises(
        ConditionGenerationError,
        match="between 0 and 1",
    ):
        make_config(**{field: value})


def test_generator_version_is_required() -> None:
    """Every condition artifact carries explicit version provenance."""

    with pytest.raises(
        ConditionGenerationError,
        match="generator_version",
    ):
        make_config(generator_version=" ")


def test_at_least_two_target_families_are_required() -> None:
    """Cross-family conflicts require a valid donor family."""

    generated = make_event_result()
    one_family_targets = tuple(
        target
        for target in generated.targets
        if target.family_id == 0
    )
    one_family_events = tuple(
        event
        for event in generated.latent_events
        if event.target_family_id == 0
    )[:1]

    with pytest.raises(
        ConditionGenerationError,
        match="two target families",
    ):
        generate_multisensory_condition_views(
            latent_events=one_family_events,
            targets=one_family_targets,
            config=make_config(),
        )


def test_unknown_event_target_is_rejected() -> None:
    """Every latent event must resolve against the supplied target registry."""

    generated = make_event_result()
    event = next(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )
    incomplete_targets = tuple(
        target
        for target in generated.targets
        if target.item_id != event.target_item_id
    )

    with pytest.raises(
        ConditionGenerationError,
        match="event target",
    ):
        generate_multisensory_condition_views(
            latent_events=(event,),
            targets=incomplete_targets,
            config=make_config(),
        )


def test_configuration_revalidates_after_replace() -> None:
    """Dataclass replacement cannot bypass locked controls."""

    config = make_config()

    with pytest.raises(ConditionGenerationError):
        replace(
            config,
            degraded_quality=1.0,
        )


def test_public_generator_has_only_declared_inputs() -> None:
    """Condition generation receives events, targets, and locked config only."""

    signature = inspect.signature(
        generate_multisensory_condition_views,
    )

    assert set(signature.parameters) == {
        "latent_events",
        "targets",
        "config",
    }
