"""Tests for the independent synthetic NOI pilot generator."""

from collections import Counter

import numpy as np
import pytest

from src.evaluation.synthetic_generator import (
    SyntheticGenerationError,
    generate_synthetic_pilot,
)
from src.evaluation.synthetic_records import SplitLabel


CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"


@pytest.fixture(scope="module")
def pilot():
    return generate_synthetic_pilot(
        CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )


def test_pilot_has_prespecified_counts(pilot) -> None:
    assert len(pilot.odor_targets) == 200
    assert len(pilot.events) == 200


def test_pilot_split_counts(pilot) -> None:
    counts = Counter(
        event.split
        for event in pilot.events
    )

    assert counts == {
        SplitLabel.TRAIN: 140,
        SplitLabel.VALIDATION: 20,
        SplitLabel.OOD_TEST: 40,
    }


def test_same_configuration_is_deterministic(pilot) -> None:
    repeated = generate_synthetic_pilot(
        CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )

    assert repeated == pilot


def test_ood_families_do_not_appear_in_development(pilot) -> None:
    development_families = {
        event.target_family_id
        for event in pilot.events
        if event.split in {
            SplitLabel.TRAIN,
            SplitLabel.VALIDATION,
        }
    }

    ood_families = {
        event.target_family_id
        for event in pilot.events
        if event.split is SplitLabel.OOD_TEST
    }

    assert development_families
    assert ood_families
    assert not development_families & ood_families


def test_template_sets_are_disjoint(pilot) -> None:
    templates_by_split = {
        split: {
            event.template_id
            for event in pilot.events
            if event.split is split
        }
        for split in SplitLabel
    }

    assert not (
        templates_by_split[SplitLabel.TRAIN]
        & templates_by_split[SplitLabel.VALIDATION]
    )
    assert not (
        templates_by_split[SplitLabel.TRAIN]
        & templates_by_split[SplitLabel.OOD_TEST]
    )
    assert not (
        templates_by_split[SplitLabel.VALIDATION]
        & templates_by_split[SplitLabel.OOD_TEST]
    )


def test_all_generated_vectors_have_unit_norm(pilot) -> None:
    for target in pilot.odor_targets:
        assert np.linalg.norm(
            target.odor_vector
        ) == pytest.approx(1.0)

    for event in pilot.events:
        for vector in (
            event.text_vector,
            event.image_vector,
            event.audio_vector,
        ):
            assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_modalities_are_not_identical_copies(pilot) -> None:
    for event in pilot.events[:20]:
        assert not np.allclose(
            event.text_vector,
            event.image_vector,
        )
        assert not np.allclose(
            event.text_vector,
            event.audio_vector,
        )
        assert not np.allclose(
            event.image_vector,
            event.audio_vector,
        )


def test_too_small_pilot_is_rejected() -> None:
    with pytest.raises(
        SyntheticGenerationError,
        match="between 10",
    ):
        generate_synthetic_pilot(
            CONFIG_PATH,
            PROTOCOL_PATH,
            event_count=9,
        )