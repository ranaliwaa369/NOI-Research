"""Tests for seedwise NOI v0.3 validation-lock derivation."""

from __future__ import annotations

import inspect
import math

from src.evaluation.multisensory_conditions import (
    ConditionGenerationConfig,
)
from src.evaluation.multisensory_records import (
    MultisensorySplit,
)
from src.evaluation.noi_v0_3_generator import (
    NOIV03GenerationConfig,
    generate_noi_v0_3_events,
)
from src.evaluation.noi_v0_3_validation_lock import (
    build_seed_validation_input,
    derive_seed_validation_lock,
)


def generation_config(
    seed: int = 1301,
) -> NOIV03GenerationConfig:
    """Return one scaled seedwise lock configuration."""

    return NOIV03GenerationConfig(
        seed=seed,
        train_event_count=70,
        validation_event_count=100,
        final_test_event_count=20,
        validation_seen_item_count=40,
        validation_known_family_unseen_item_count=30,
        validation_unseen_family_count=30,
        final_seen_item_count=8,
        final_known_family_unseen_item_count=6,
        final_unseen_family_count=6,
        known_family_count=4,
        training_items_per_family=4,
        withheld_items_per_known_family=2,
        validation_unknown_family_count=2,
        final_unknown_family_count=2,
        items_per_unknown_family=3,
        generator_version="0.3.1-lock-test",
        feasibility_only=True,
    )


def condition_config(
    seed: int = 1301,
) -> ConditionGenerationConfig:
    """Return registered condition mechanics for lock tests."""

    return ConditionGenerationConfig(
        seed=seed,
        odor_noise_scale=0.10,
        tactile_noise_scale=0.10,
        degraded_quality=0.40,
        locked_temporal_offset_steps=3,
        generator_version="0.3.1-lock-test",
    )


def make_input(seed: int = 1301):
    """Build one leakage-controlled seed input."""

    generated = generate_noi_v0_3_events(
        generation_config(seed)
    )

    return (
        generated,
        build_seed_validation_input(
            generated=generated,
            condition_config=condition_config(seed),
        ),
    )


def test_input_builder_interface_is_explicit() -> None:
    """Only generated records and locked condition controls enter."""

    signature = inspect.signature(
        build_seed_validation_input
    )

    assert set(signature.parameters) == {
        "generated",
        "condition_config",
    }


def test_derivation_interface_has_only_registered_controls() -> None:
    """No final-test labels or results can enter through the public API."""

    signature = inspect.signature(
        derive_seed_validation_lock
    )

    assert set(signature.parameters) == {
        "lock_input",
        "support_bootstrap_seed",
        "support_bootstrap_resamples",
        "confidence_level",
        "maximum_false_known_rate",
        "maximum_false_conflict_rate",
    }


def test_input_contains_training_and_validation_only() -> None:
    """Final-test records must not survive input construction."""

    _, lock_input = make_input()

    assert len(lock_input.training_events) == 70
    assert len(lock_input.validation_events) == 100
    assert all(
        event.split is MultisensorySplit.TRAIN
        for event in lock_input.training_events
    )
    assert all(
        event.split is MultisensorySplit.VALIDATION
        for event in lock_input.validation_events
    )
    assert lock_input.final_test_event_count_exposed == 0


def test_permitted_targets_exclude_final_only_targets() -> None:
    """Validation stress generation cannot access final-only prototypes."""

    generated, lock_input = make_input()

    permitted_ids = {
        target.item_id
        for target in lock_input.permitted_targets
    }
    development_ids = {
        event.target_item_id
        for event in (
            lock_input.training_events
            + lock_input.validation_events
        )
    }
    final_ids = {
        event.target_item_id
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    }
    final_only_ids = final_ids - development_ids

    assert permitted_ids == development_ids
    assert permitted_ids.isdisjoint(final_only_ids)


def test_seed_lock_contains_all_five_finite_values() -> None:
    """One seed must emit the complete registered threshold lock."""

    _, lock_input = make_input()

    result = derive_seed_validation_lock(
        lock_input=lock_input,
        support_bootstrap_seed=4242,
        support_bootstrap_resamples=200,
        confidence_level=0.95,
        maximum_false_known_rate=0.05,
        maximum_false_conflict_rate=0.05,
    )

    values = (
        result.support_threshold,
        result.support_uncertainty_lower,
        result.support_uncertainty_upper,
        result.reliability_threshold,
        result.conflict_threshold,
    )

    assert all(math.isfinite(value) for value in values)
    assert (
        result.support_uncertainty_lower
        <= result.support_threshold
        <= result.support_uncertainty_upper
    )


def test_seed_lock_satisfies_validation_safety_caps() -> None:
    """Both registered false-positive constraints must hold."""

    _, lock_input = make_input()

    result = derive_seed_validation_lock(
        lock_input=lock_input,
        support_bootstrap_seed=4242,
        support_bootstrap_resamples=200,
        confidence_level=0.95,
        maximum_false_known_rate=0.05,
        maximum_false_conflict_rate=0.05,
    )

    assert result.validation_false_known_rate <= 0.05
    assert result.validation_false_conflict_rate <= 0.05


def test_seed_lock_records_no_final_test_use() -> None:
    """The output audit must explicitly deny final-test access."""

    _, lock_input = make_input()

    result = derive_seed_validation_lock(
        lock_input=lock_input,
        support_bootstrap_seed=4242,
        support_bootstrap_resamples=200,
        confidence_level=0.95,
        maximum_false_known_rate=0.05,
        maximum_false_conflict_rate=0.05,
    )

    assert result.final_test_events_used == 0
    assert result.final_test_labels_used is False
    assert result.condition_metadata_used_as_model_input is False
    assert result.target_labels_used_as_inference_input is False


def test_seed_lock_preserves_registered_condition_controls() -> None:
    """Validation views must use protocol noise and temporal offset."""

    _, lock_input = make_input()

    result = derive_seed_validation_lock(
        lock_input=lock_input,
        support_bootstrap_seed=4242,
        support_bootstrap_resamples=200,
        confidence_level=0.95,
        maximum_false_known_rate=0.05,
        maximum_false_conflict_rate=0.05,
    )

    assert result.odor_noise_scale == 0.10
    assert result.tactile_noise_scale == 0.10
    assert result.temporal_offset_steps == 3
    assert result.quality_metadata_used_as_model_input is False


def test_seed_lock_is_deterministic() -> None:
    """Identical seed input and controls reproduce exactly."""

    _, lock_input = make_input()

    arguments = {
        "lock_input": lock_input,
        "support_bootstrap_seed": 4242,
        "support_bootstrap_resamples": 200,
        "confidence_level": 0.95,
        "maximum_false_known_rate": 0.05,
        "maximum_false_conflict_rate": 0.05,
    }

    first = derive_seed_validation_lock(**arguments)
    second = derive_seed_validation_lock(**arguments)

    assert first == second
