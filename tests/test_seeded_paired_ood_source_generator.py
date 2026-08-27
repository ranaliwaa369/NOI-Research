"""Tests for prespecified seeds in paired OOD generation."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.paired_ood_source_generator import (
    PairedOODSourceGenerationError,
    generate_paired_graded_ood_bundle,
)
from src.evaluation.synthetic_records import (
    SplitLabel,
)


SYNTHETIC_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"
AMENDMENT_PATH = (
    "configs/protocol_amendment_v0.2.yaml"
)
DEFINITION_PATH = (
    "configs/graded_ood_generation.yaml"
)


def generate(
    *,
    generator_seed: int,
    ood_seed: int,
):
    return generate_paired_graded_ood_bundle(
        SYNTHETIC_PATH,
        PROTOCOL_PATH,
        AMENDMENT_PATH,
        DEFINITION_PATH,
        event_count=200,
        generator_seed=generator_seed,
        ood_seed=ood_seed,
    )


def test_prespecified_seeds_are_recorded() -> None:
    bundle = generate(
        generator_seed=1101,
        ood_seed=9101,
    )

    assert bundle.generator_seed == 1101
    assert bundle.ood_seed == 9101
    assert (
        bundle.original_dataset.generator_seed
        == 1101
    )
    assert bundle.original_dataset.ood_seed == 9101


def test_seeded_generation_is_deterministic() -> None:
    first = generate(
        generator_seed=1101,
        ood_seed=9101,
    )
    second = generate(
        generator_seed=1101,
        ood_seed=9101,
    )

    assert first == second


def test_different_seeds_change_generated_data() -> None:
    first = generate(
        generator_seed=1101,
        ood_seed=9101,
    )
    second = generate(
        generator_seed=1201,
        ood_seed=9201,
    )

    assert (
        first.original_dataset
        != second.original_dataset
    )
    assert (
        first.graded_dataset
        != second.graded_dataset
    )


def test_seeded_severe_replay_remains_verified() -> None:
    bundle = generate(
        generator_seed=1301,
        ood_seed=9301,
    )

    assert bundle.severe_reference_verified is True

    original_ood = tuple(
        event
        for event in bundle.original_dataset.events
        if event.split is SplitLabel.OOD_TEST
    )
    severe = bundle.graded_dataset.events_for_tier(
        next(
            event.tier
            for event in bundle.graded_dataset.events
            if event.tier.value == "severe"
        )
    )

    assert len(original_ood) == len(severe)

    for original, replayed in zip(
        original_ood,
        severe,
        strict=True,
    ):
        assert (
            original.target_item_id
            == replayed.target_item_id
        )
        assert (
            original.target_family_id
            == replayed.target_family_id
        )
        assert (
            original.template_id
            == replayed.template_id
        )
        np.testing.assert_allclose(
            original.text_vector,
            replayed.text_vector,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            original.image_vector,
            replayed.image_vector,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            original.audio_vector,
            replayed.audio_vector,
            rtol=0.0,
            atol=1e-15,
        )


def test_ood_families_remain_held_out() -> None:
    bundle = generate(
        generator_seed=1401,
        ood_seed=9401,
    )

    training_families = {
        event.target_family_id
        for event in bundle.original_dataset.events
        if event.split is SplitLabel.TRAIN
    }
    ood_families = {
        event.target_family_id
        for event in bundle.original_dataset.events
        if event.split is SplitLabel.OOD_TEST
    }

    assert training_families
    assert ood_families
    assert training_families.isdisjoint(
        ood_families
    )


@pytest.mark.parametrize(
    ("generator_seed", "ood_seed"),
    (
        (1001, 1001),
        (-1, 9001),
        (1001, -1),
        (True, 9001),
        (1001, True),
    ),
)
def test_invalid_seed_pair_is_rejected(
    generator_seed,
    ood_seed,
) -> None:
    with pytest.raises(
        PairedOODSourceGenerationError,
        match="seed",
    ):
        generate(
            generator_seed=generator_seed,
            ood_seed=ood_seed,
        )
