"""Tests for deterministic NOI v0.3 tactile generation."""

from __future__ import annotations

from dataclasses import replace
import inspect
import math

import pytest

from src.evaluation.tactile_generator import (
    TACTILE_FEATURE_LAYOUT,
    TactileGenerationConfig,
    TactileGenerationError,
    TactileGenerationProvenance,
    TactileGenerationResult,
    generate_tactile_prototypes,
)


def make_config(
    **changes: object,
) -> TactileGenerationConfig:
    """Return one small feasibility-only generator configuration."""

    values: dict[str, object] = {
        "seed": 1301,
        "family_count": 4,
        "items_per_family": 5,
        "family_scale": 0.65,
        "item_residual_scale": 0.12,
        "generator_version": "0.3.0-feasibility",
    }
    values.update(changes)

    return TactileGenerationConfig(**values)


def mean_absolute_distance(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    """Return mean absolute coordinate distance."""

    return sum(
        abs(a - b)
        for a, b in zip(left, right, strict=True)
    ) / len(left)


def test_feature_layout_is_explicit_and_eight_dimensional() -> None:
    """Every tactile coordinate has one prespecified interpretation."""

    assert TACTILE_FEATURE_LAYOUT == (
        "roughness",
        "stiffness",
        "friction",
        "contact_geometry_1",
        "contact_geometry_2",
        "pressure_response_1",
        "pressure_response_2",
        "pressure_response_3",
    )


def test_same_seed_produces_identical_result() -> None:
    """Repeated generation with one configuration is exact."""

    config = make_config()

    first = generate_tactile_prototypes(config)
    second = generate_tactile_prototypes(config)

    assert first == second


def test_different_seed_changes_prototypes() -> None:
    """The feasibility seed controls stochastic realization."""

    first = generate_tactile_prototypes(make_config(seed=1301))
    second = generate_tactile_prototypes(make_config(seed=1302))

    assert first.prototypes != second.prototypes


def test_expected_number_of_prototypes_is_generated() -> None:
    """The result contains one vector for every requested item."""

    config = make_config(
        family_count=3,
        items_per_family=7,
    )

    result = generate_tactile_prototypes(config)

    assert len(result.prototypes) == 21
    assert len(result.family_indices) == 21
    assert len(result.item_indices) == 21


def test_every_prototype_is_finite_bounded_and_eight_dimensional() -> None:
    """Generated touch features respect the protocol representation."""

    result = generate_tactile_prototypes(make_config())

    for vector in result.prototypes:
        assert len(vector) == 8
        assert all(math.isfinite(value) for value in vector)
        assert all(-1.0 <= value <= 1.0 for value in vector)


def test_family_and_item_indices_follow_generation_order() -> None:
    """Structural indices are positional metadata, not encoded labels."""

    result = generate_tactile_prototypes(
        make_config(
            family_count=2,
            items_per_family=3,
        ),
    )

    assert result.family_indices == (0, 0, 0, 1, 1, 1)
    assert result.item_indices == (0, 1, 2, 0, 1, 2)


def test_family_structure_exceeds_item_residual_structure() -> None:
    """Items in one family remain closer than cross-family items."""

    result = generate_tactile_prototypes(
        make_config(
            family_count=5,
            items_per_family=8,
            family_scale=0.75,
            item_residual_scale=0.05,
        ),
    )

    within: list[float] = []
    between: list[float] = []

    for left_index, left in enumerate(result.prototypes):
        for right_index in range(left_index + 1, len(result.prototypes)):
            distance = mean_absolute_distance(
                left,
                result.prototypes[right_index],
            )

            if (
                result.family_indices[left_index]
                == result.family_indices[right_index]
            ):
                within.append(distance)
            else:
                between.append(distance)

    assert sum(within) / len(within) < sum(between) / len(between)


def test_provenance_is_explicit_and_complete() -> None:
    """Every generated batch reports its construction metadata."""

    config = make_config()
    result = generate_tactile_prototypes(config)

    assert isinstance(result.provenance, TactileGenerationProvenance)
    assert result.provenance.generator_version == config.generator_version
    assert result.provenance.seed == config.seed
    assert result.provenance.dimension == 8
    assert result.provenance.feature_layout == TACTILE_FEATURE_LAYOUT
    assert result.provenance.family_scale == config.family_scale
    assert (
        result.provenance.item_residual_scale
        == config.item_residual_scale
    )
    assert result.provenance.feasibility_only is True


def test_result_is_an_explicit_immutable_record() -> None:
    """The public result has a stable typed representation."""

    result = generate_tactile_prototypes(make_config())

    assert isinstance(result, TactileGenerationResult)

    with pytest.raises(AttributeError):
        result.prototypes = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", True),
        ("seed", -1),
        ("family_count", 0),
        ("family_count", True),
        ("items_per_family", 0),
        ("items_per_family", True),
    ),
)
def test_invalid_integer_configuration_is_rejected(
    field: str,
    value: object,
) -> None:
    """Counts and seeds must be valid protocol integers."""

    with pytest.raises(TactileGenerationError):
        make_config(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("family_scale", 0.0),
        ("family_scale", 1.1),
        ("family_scale", float("nan")),
        ("item_residual_scale", 0.0),
        ("item_residual_scale", 1.1),
        ("item_residual_scale", float("inf")),
    ),
)
def test_invalid_scale_is_rejected(
    field: str,
    value: object,
) -> None:
    """Generator scales must be finite values in (0, 1]."""

    with pytest.raises(
        TactileGenerationError,
        match="between 0 and 1",
    ):
        make_config(**{field: value})


def test_item_residual_must_be_smaller_than_family_scale() -> None:
    """Family structure must dominate the item-specific residual."""

    with pytest.raises(
        TactileGenerationError,
        match="smaller than family_scale",
    ):
        make_config(
            family_scale=0.20,
            item_residual_scale=0.20,
        )


def test_empty_generator_version_is_rejected() -> None:
    """Provenance requires a nonempty generator version."""

    with pytest.raises(
        TactileGenerationError,
        match="generator_version",
    ):
        make_config(generator_version=" ")


def test_configuration_is_immutable() -> None:
    """A validated configuration cannot silently change after creation."""

    config = make_config()

    with pytest.raises(AttributeError):
        config.seed = 1302  # type: ignore[misc]


def test_generator_interface_prohibits_label_inputs() -> None:
    """The generator cannot receive IDs, splits, or support labels."""

    signature = inspect.signature(generate_tactile_prototypes)
    parameter_names = set(signature.parameters)

    forbidden = {
        "item_id",
        "item_ids",
        "family_id",
        "family_ids",
        "split",
        "split_label",
        "support",
        "support_label",
        "support_regime",
    }

    assert parameter_names == {"config"}
    assert parameter_names.isdisjoint(forbidden)


def test_positional_indices_do_not_directly_appear_as_features() -> None:
    """No coordinate is an exact normalized family or item label."""

    config = make_config(
        family_count=4,
        items_per_family=5,
    )
    result = generate_tactile_prototypes(config)

    family_denominator = max(config.family_count - 1, 1)
    item_denominator = max(config.items_per_family - 1, 1)

    for vector, family_index, item_index in zip(
        result.prototypes,
        result.family_indices,
        result.item_indices,
        strict=True,
    ):
        normalized_family = family_index / family_denominator
        normalized_item = item_index / item_denominator

        assert vector != (normalized_family,) * 8
        assert vector != (normalized_item,) * 8


def test_replace_revalidates_configuration() -> None:
    """Dataclass replacement cannot bypass protocol validation."""

    config = make_config()

    with pytest.raises(TactileGenerationError):
        replace(
            config,
            item_residual_scale=config.family_scale,
        )
