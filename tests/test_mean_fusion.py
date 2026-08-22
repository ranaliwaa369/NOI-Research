"""Tests for the transparent mean-fusion baseline."""

from datetime import datetime, timezone

import numpy as np
import pytest

from src.encoders.mean_fusion import FusionError, mean_fuse_context
from src.models import MultimodalContext


def make_context(**changes) -> MultimodalContext:
    values = {
        "event_id": "event-001",
        "timestamp_utc": datetime.now(timezone.utc),
        "text_vector": (1.0, 0.0),
        "image_vector": (0.0, 1.0),
        "audio_vector": None,
        "temporal_vector": None,
    }
    values.update(changes)
    return MultimodalContext(**values)


def test_mean_fusion_returns_normalized_vector() -> None:
    fused = mean_fuse_context(make_context())

    np.testing.assert_allclose(
        fused,
        np.array([0.70710678, 0.70710678]),
        rtol=1e-7,
    )
    assert np.linalg.norm(fused) == pytest.approx(1.0)


def test_single_available_modality_is_normalized() -> None:
    context = make_context(image_vector=None)

    fused = mean_fuse_context(context)

    np.testing.assert_allclose(fused, np.array([1.0, 0.0]))


def test_unavailable_selected_modality_is_rejected() -> None:
    context = make_context(audio_vector=None)

    with pytest.raises(
        FusionError,
        match="None of the selected modalities",
    ):
        mean_fuse_context(context, modalities=("audio",))


def test_duplicate_modalities_are_rejected() -> None:
    with pytest.raises(
        FusionError,
        match="Duplicate modalities",
    ):
        mean_fuse_context(
            make_context(),
            modalities=("text", "text"),
        )


def test_unknown_modality_is_rejected() -> None:
    with pytest.raises(
        FusionError,
        match="Unsupported modalities",
    ):
        mean_fuse_context(
            make_context(),
            modalities=("text", "smell"),
        )


def test_mismatched_dimensions_are_rejected() -> None:
    context = make_context(
        text_vector=(1.0, 0.0),
        image_vector=(0.0, 1.0, 0.0),
    )

    with pytest.raises(
        FusionError,
        match="identical dimensions",
    ):
        mean_fuse_context(context)


def test_zero_fused_vector_is_rejected() -> None:
    context = make_context(
        text_vector=(1.0, 0.0),
        image_vector=(-1.0, 0.0),
    )

    with pytest.raises(
        FusionError,
        match="norm is zero",
    ):
        mean_fuse_context(context)