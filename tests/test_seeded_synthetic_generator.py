"""Tests for explicit repeated-run synthetic seeds."""

import pytest

from src.evaluation.synthetic_generator import (
    SyntheticGenerationError,
    generate_synthetic_pilot,
    generate_synthetic_pilot_with_seeds,
)


CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"


def test_locked_original_seed_reproduces_original_dataset() -> None:
    original = generate_synthetic_pilot(
        CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )

    explicit = generate_synthetic_pilot_with_seeds(
        CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
        generator_seed=1001,
        ood_seed=9001,
    )

    assert explicit == original


def test_new_seeds_are_recorded_and_change_dataset() -> None:
    original = generate_synthetic_pilot(
        CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )

    repeated = generate_synthetic_pilot_with_seeds(
        CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
        generator_seed=1101,
        ood_seed=9101,
    )

    assert repeated.generator_seed == 1101
    assert repeated.ood_seed == 9101
    assert repeated != original
    assert repeated.events != original.events


def test_explicit_seed_generation_is_deterministic() -> None:
    first = generate_synthetic_pilot_with_seeds(
        CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
        generator_seed=1201,
        ood_seed=9201,
    )
    second = generate_synthetic_pilot_with_seeds(
        CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
        generator_seed=1201,
        ood_seed=9201,
    )

    assert first == second


@pytest.mark.parametrize(
    ("generator_seed", "ood_seed"),
    (
        (1001, 1001),
        (True, 9001),
        (1001, False),
        (-1, 9001),
        (1001, -1),
        (1001.0, 9001),
        (1001, 9001.0),
    ),
)
def test_invalid_explicit_seeds_are_rejected(
    generator_seed,
    ood_seed,
) -> None:
    with pytest.raises(
        SyntheticGenerationError,
        match="seeds",
    ):
        generate_synthetic_pilot_with_seeds(
            CONFIG_PATH,
            PROTOCOL_PATH,
            event_count=200,
            generator_seed=generator_seed,
            ood_seed=ood_seed,
        )
