"""Tests for replay-safe paired OOD source generation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.graded_ood import OODTier
from src.evaluation.paired_ood_source_generator import (
    EXPECTED_AMENDMENT_SHA256,
    EXPECTED_GENERATION_DEFINITION_SHA256,
    PairedGradedOODBundle,
    PairedOODSourceGenerationError,
    generate_paired_graded_ood_bundle,
)
from src.evaluation.synthetic_records import SplitLabel


SYNTHETIC_PATH = Path("configs/synthetic_data.yaml")
PROTOCOL_PATH = Path("configs/research_protocol.yaml")
AMENDMENT_PATH = Path("configs/protocol_amendment_v0.2.yaml")
DEFINITION_PATH = Path("configs/graded_ood_generation.yaml")


@pytest.fixture(scope="module")
def bundle() -> PairedGradedOODBundle:
    """Generate the deterministic 200-event paired pilot."""

    return generate_paired_graded_ood_bundle(
        SYNTHETIC_PATH,
        PROTOCOL_PATH,
        AMENDMENT_PATH,
        DEFINITION_PATH,
        event_count=200,
    )


def test_bundle_preserves_original_library(
    bundle: PairedGradedOODBundle,
) -> None:
    """The paired bundle must reuse the original 200 odor targets."""

    assert len(bundle.odor_targets) == 200
    assert bundle.odor_targets == bundle.original_dataset.odor_targets


def test_bundle_has_locked_paired_counts(
    bundle: PairedGradedOODBundle,
) -> None:
    """Forty latent OOD events must produce 120 observed rows."""

    assert bundle.source_count == 40
    assert bundle.graded_dataset.latent_event_count == 40
    assert bundle.graded_dataset.observed_event_count == 120
    assert bundle.graded_dataset.views_per_latent_event == 3


def test_tier_counts_are_balanced(
    bundle: PairedGradedOODBundle,
) -> None:
    """Every latent event must appear once in every severity tier."""

    assert bundle.graded_dataset.tier_counts == {
        "mild": 40,
        "moderate": 40,
        "severe": 40,
    }


def test_severe_replay_verification_passes(
    bundle: PairedGradedOODBundle,
) -> None:
    """The replayed severe observations must match the original pilot."""

    assert bundle.severe_reference_verified is True


def test_bundle_records_locked_hashes(
    bundle: PairedGradedOODBundle,
) -> None:
    """The bundle must record both governing configuration hashes."""

    assert (
        bundle.generation_definition_sha256
        == EXPECTED_GENERATION_DEFINITION_SHA256
    )
    assert bundle.amendment_sha256 == EXPECTED_AMENDMENT_SHA256


def test_bundle_records_original_seeds(
    bundle: PairedGradedOODBundle,
) -> None:
    """Original generator provenance must remain attached."""

    assert bundle.generator_seed == 1001
    assert bundle.ood_seed == 9001
    assert bundle.generator_version == "0.1.0"


def test_original_split_counts_remain_unchanged(
    bundle: PairedGradedOODBundle,
) -> None:
    """Pair generation must not modify the original pilot dataset."""

    counts = {
        split: sum(
            event.split is split
            for event in bundle.original_dataset.events
        )
        for split in SplitLabel
    }

    assert counts == {
        SplitLabel.TRAIN: 140,
        SplitLabel.VALIDATION: 20,
        SplitLabel.OOD_TEST: 40,
    }


def test_severe_vectors_equal_original_ood_vectors(
    bundle: PairedGradedOODBundle,
) -> None:
    """Every severe graded view must preserve its original OOD direction."""

    original = {
        event.event_id: event
        for event in bundle.original_dataset.events
        if event.split is SplitLabel.OOD_TEST
    }

    severe_views = bundle.graded_dataset.events_for_tier(
        OODTier.SEVERE
    )

    assert len(severe_views) == len(original)

    for view in severe_views:
        original_id = view.latent_event_id.removeprefix(
            "latent-"
        )
        source = original[original_id]

        assert view.text_vector == pytest.approx(
            source.text_vector,
            abs=1e-15,
        )
        assert view.image_vector == pytest.approx(
            source.image_vector,
            abs=1e-15,
        )
        assert view.audio_vector == pytest.approx(
            source.audio_vector,
            abs=1e-15,
        )


def test_every_generated_vector_is_unit_normalized(
    bundle: PairedGradedOODBundle,
) -> None:
    """All 360 modality vectors must have unit L2 norm."""

    for event in bundle.graded_dataset.events:
        for vector in (
            event.text_vector,
            event.image_vector,
            event.audio_vector,
        ):
            assert np.linalg.norm(vector) == pytest.approx(
                1.0,
                abs=1e-10,
            )


def test_ground_truth_is_paired_across_tiers(
    bundle: PairedGradedOODBundle,
) -> None:
    """Each latent triplet must preserve its complete ground truth."""

    for latent_id in bundle.graded_dataset.latent_event_ids:
        views = bundle.graded_dataset.events_for_latent_id(
            latent_id
        )

        assert len({
            view.target_item_id
            for view in views
        }) == 1
        assert len({
            view.target_family_id
            for view in views
        }) == 1
        assert len({
            view.template_id
            for view in views
        }) == 1


def test_observed_identifiers_are_unique(
    bundle: PairedGradedOODBundle,
) -> None:
    """Every tier observation must have one unique auditable ID."""

    identifiers = [
        event.observed_event_id
        for event in bundle.graded_dataset.events
    ]

    assert len(identifiers) == 120
    assert len(set(identifiers)) == 120


def test_generation_is_deterministic(
    bundle: PairedGradedOODBundle,
) -> None:
    """Repeating the complete replay must produce an equal bundle."""

    repeated = generate_paired_graded_ood_bundle(
        SYNTHETIC_PATH,
        PROTOCOL_PATH,
        AMENDMENT_PATH,
        DEFINITION_PATH,
        event_count=200,
    )

    assert repeated == bundle


def test_bundle_is_immutable(
    bundle: PairedGradedOODBundle,
) -> None:
    """Provenance cannot be changed after bundle construction."""

    with pytest.raises(FrozenInstanceError):
        bundle.ood_seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "invalid_event_count",
    (
        0,
        9,
        -1,
        True,
        200.0,
    ),
)
def test_invalid_event_count_is_rejected(
    invalid_event_count,
) -> None:
    """Generation requires an integer event count of at least ten."""

    with pytest.raises(
        PairedOODSourceGenerationError,
        match="event_count",
    ):
        generate_paired_graded_ood_bundle(
            SYNTHETIC_PATH,
            PROTOCOL_PATH,
            AMENDMENT_PATH,
            DEFINITION_PATH,
            event_count=invalid_event_count,
        )


def test_tampered_generation_definition_is_rejected(
    tmp_path: Path,
) -> None:
    """Any post-lock definition change must fail SHA verification."""

    tampered = tmp_path / "graded_ood_generation.yaml"
    tampered.write_text(
        DEFINITION_PATH.read_text(encoding="utf-8")
        + "\n# unauthorized change\n",
        encoding="utf-8",
    )

    with pytest.raises(
        PairedOODSourceGenerationError,
        match="SHA-256",
    ):
        generate_paired_graded_ood_bundle(
            SYNTHETIC_PATH,
            PROTOCOL_PATH,
            AMENDMENT_PATH,
            tampered,
            event_count=200,
        )


def test_tampered_amendment_is_rejected(
    tmp_path: Path,
) -> None:
    """Any post-lock amendment change must fail SHA verification."""

    tampered = tmp_path / "amendment.yaml"
    tampered.write_text(
        AMENDMENT_PATH.read_text(encoding="utf-8")
        + "\n# unauthorized change\n",
        encoding="utf-8",
    )

    with pytest.raises(
        PairedOODSourceGenerationError,
        match="SHA-256",
    ):
        generate_paired_graded_ood_bundle(
            SYNTHETIC_PATH,
            PROTOCOL_PATH,
            tampered,
            DEFINITION_PATH,
            event_count=200,
        )


def test_identity_and_severe_directions_are_not_identical(
    bundle: PairedGradedOODBundle,
) -> None:
    """The independent transformation must produce a real shift."""

    mild = bundle.graded_dataset.events_for_tier(
        OODTier.MILD
    )
    severe = bundle.graded_dataset.events_for_tier(
        OODTier.SEVERE
    )

    assert any(
        not np.allclose(
            mild_event.text_vector,
            severe_event.text_vector,
        )
        for mild_event, severe_event in zip(
            mild,
            severe,
            strict=True,
        )
    )


def test_target_identifiers_do_not_enter_vectors(
    bundle: PairedGradedOODBundle,
) -> None:
    """Feature vectors must remain strictly numeric."""

    for event in bundle.graded_dataset.events:
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
