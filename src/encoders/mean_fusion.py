"""Transparent mean-fusion baseline for multimodal context vectors."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from src.models import MultimodalContext


FloatArray = NDArray[np.float64]

SUPPORTED_MODALITIES = {
    "text": "text_vector",
    "image": "image_vector",
    "audio": "audio_vector",
    "temporal": "temporal_vector",
}


class FusionError(ValueError):
    """Raised when contextual vectors cannot be fused fairly."""


def mean_fuse_context(
    context: MultimodalContext,
    modalities: Iterable[str] = ("text", "image", "audio"),
) -> FloatArray:
    """Average selected available modalities and L2-normalize the result.

    This function is a transparent baseline. It is not presented as a
    learned multimodal encoder or as evidence of perceptual validity.
    """

    selected_modalities = tuple(modalities)

    if not selected_modalities:
        raise FusionError("At least one modality must be selected.")

    if len(selected_modalities) != len(set(selected_modalities)):
        raise FusionError(
            "Duplicate modalities are not allowed because they would "
            "change the fusion weighting."
        )

    unknown = set(selected_modalities) - set(SUPPORTED_MODALITIES)
    if unknown:
        raise FusionError(
            f"Unsupported modalities: {sorted(unknown)}"
        )

    vectors: list[FloatArray] = []

    for modality in selected_modalities:
        attribute_name = SUPPORTED_MODALITIES[modality]
        value = getattr(context, attribute_name)

        if value is not None:
            vector = np.asarray(value, dtype=np.float64)

            if vector.ndim != 1:
                raise FusionError(
                    "Each modality must be represented by a "
                    "one-dimensional vector."
                )

            if not np.all(np.isfinite(vector)):
                raise FusionError(
                    f"The {modality} vector contains nonfinite values."
                )

            vectors.append(vector)

    if not vectors:
        raise FusionError(
            "None of the selected modalities is available."
        )

    dimensions = {vector.shape[0] for vector in vectors}

    if len(dimensions) != 1:
        raise FusionError(
            "All fused modality vectors must have identical dimensions."
        )

    fused = np.mean(
        np.stack(vectors, axis=0),
        axis=0,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(fused)):
        raise FusionError(
            "The fused vector must contain only finite values."
        )

    norm = float(np.linalg.norm(fused))

    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise FusionError(
            "The fused vector cannot be normalized because its norm is zero."
        )

    normalized = fused / norm

    return normalized.astype(np.float64, copy=False)