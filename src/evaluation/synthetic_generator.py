"""Independent synthetic pilot generator for NOI implementation testing."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.evaluation.split_plan import SplitPlan, create_split_plan
from src.evaluation.synthetic_config import (
    load_synthetic_configuration,
)
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
    SyntheticEvent,
    SyntheticOdorTarget,
)


FloatArray = NDArray[np.float64]
MODALITIES = ("text", "image", "audio")


class SyntheticGenerationError(ValueError):
    """Raised when a synthetic pilot cannot be generated safely."""


def generate_synthetic_pilot(
    configuration_path: str | Path,
    protocol_path: str | Path,
    *,
    event_count: int = 200,
) -> SyntheticDataset:
    """Generate a small deterministic pilot before full-data generation."""

    configuration = load_synthetic_configuration(
        configuration_path,
        protocol_path,
    )

    maximum_events = configuration["dataset"]["total_events"]

    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 10
        or event_count > maximum_events
    ):
        raise SyntheticGenerationError(
            "event_count must be an integer between 10 and "
            f"{maximum_events}."
        )

    return _generate_dataset(
        configuration=configuration,
        event_count=event_count,
    )


def _generate_dataset(
    *,
    configuration: dict[str, Any],
    event_count: int,
) -> SyntheticDataset:
    dataset_config = configuration["dataset"]
    split_config = configuration["splits"]
    randomness = configuration["randomness"]

    latent_dimension = dataset_config["latent_dimension"]
    modality_dimension = dataset_config["modality_dimension"]

    if latent_dimension != modality_dimension:
        raise SyntheticGenerationError(
            "The current orthogonal pilot requires equal latent "
            "and modality dimensions."
        )

    generator_seed = randomness["generator_seed"]
    ood_seed = randomness["independent_ood_seed"]

    id_rng = np.random.default_rng(generator_seed)
    ood_rng = np.random.default_rng(ood_seed)

    split_plan = create_split_plan(
        odor_families=dataset_config["odor_families"],
        context_templates=dataset_config["context_templates"],
        odor_family_held_out_fraction=split_config[
            "odor_family_held_out_fraction"
        ],
        context_template_held_out_fraction=split_config[
            "context_template_held_out_fraction"
        ],
        validation_fraction=split_config["validation_fraction"],
        seed=generator_seed,
    )

    odor_targets = _generate_odor_targets(
        family_count=dataset_config["odor_families"],
        target_count=dataset_config["odor_targets"],
        dimension=latent_dimension,
        split_plan=split_plan,
        id_rng=id_rng,
        ood_rng=ood_rng,
    )

    template_offsets = _generate_template_offsets(
        template_count=dataset_config["context_templates"],
        dimension=latent_dimension,
        split_plan=split_plan,
        id_rng=id_rng,
        ood_rng=ood_rng,
    )

    id_transforms = {
        modality: _orthogonal_matrix(
            latent_dimension,
            id_rng,
        )
        for modality in MODALITIES
    }

    ood_transforms = {
        modality: _orthogonal_matrix(
            latent_dimension,
            ood_rng,
        )
        for modality in MODALITIES
    }

    counts = _split_event_counts(
        event_count=event_count,
        validation_fraction=split_config["validation_fraction"],
        final_test_fraction=split_config["final_test_fraction"],
    )

    events = []

    events.extend(
        _generate_events_for_split(
            split=SplitLabel.TRAIN,
            count=counts[SplitLabel.TRAIN],
            allowed_families=split_plan.training_families,
            allowed_templates=split_plan.training_templates,
            odor_targets=odor_targets,
            template_offsets=template_offsets,
            transformations=id_transforms,
            rng=id_rng,
            base_noise=0.05,
        )
    )

    events.extend(
        _generate_events_for_split(
            split=SplitLabel.VALIDATION,
            count=counts[SplitLabel.VALIDATION],
            allowed_families=split_plan.validation_families,
            allowed_templates=split_plan.validation_templates,
            odor_targets=odor_targets,
            template_offsets=template_offsets,
            transformations=id_transforms,
            rng=id_rng,
            base_noise=0.05,
        )
    )

    events.extend(
        _generate_events_for_split(
            split=SplitLabel.OOD_TEST,
            count=counts[SplitLabel.OOD_TEST],
            allowed_families=split_plan.ood_test_families,
            allowed_templates=split_plan.ood_test_templates,
            odor_targets=odor_targets,
            template_offsets=template_offsets,
            transformations=ood_transforms,
            rng=ood_rng,
            base_noise=0.10,
        )
    )

    return SyntheticDataset(
        odor_targets=tuple(odor_targets),
        events=tuple(events),
        generator_version=configuration["generator"]["version"],
        generator_seed=generator_seed,
        ood_seed=ood_seed,
    )


def _generate_odor_targets(
    *,
    family_count: int,
    target_count: int,
    dimension: int,
    split_plan: SplitPlan,
    id_rng: np.random.Generator,
    ood_rng: np.random.Generator,
) -> list[SyntheticOdorTarget]:
    targets_per_family = target_count // family_count
    ood_families = set(split_plan.ood_test_families)
    targets = []
    item_index = 0

    for family_id in range(family_count):
        rng = ood_rng if family_id in ood_families else id_rng
        centroid = _normalize(rng.normal(size=dimension))

        for _ in range(targets_per_family):
            residual = rng.normal(
                loc=0.0,
                scale=0.25,
                size=dimension,
            )
            odor_vector = _normalize(centroid + residual)

            targets.append(
                SyntheticOdorTarget(
                    item_id=f"odor-{item_index:04d}",
                    family_id=family_id,
                    odor_vector=tuple(float(x) for x in odor_vector),
                )
            )
            item_index += 1

    return targets


def _generate_template_offsets(
    *,
    template_count: int,
    dimension: int,
    split_plan: SplitPlan,
    id_rng: np.random.Generator,
    ood_rng: np.random.Generator,
) -> dict[int, FloatArray]:
    ood_templates = set(split_plan.ood_test_templates)
    offsets = {}

    for template_id in range(template_count):
        rng = ood_rng if template_id in ood_templates else id_rng
        offsets[template_id] = _normalize(
            rng.normal(size=dimension)
        )

    return offsets


def _generate_events_for_split(
    *,
    split: SplitLabel,
    count: int,
    allowed_families: Sequence[int],
    allowed_templates: Sequence[int],
    odor_targets: Sequence[SyntheticOdorTarget],
    template_offsets: dict[int, FloatArray],
    transformations: dict[str, FloatArray],
    rng: np.random.Generator,
    base_noise: float,
) -> list[SyntheticEvent]:
    family_set = set(allowed_families)
    allowed_targets = [
        target
        for target in odor_targets
        if target.family_id in family_set
    ]

    events = []

    for index in range(count):
        target = allowed_targets[
            int(rng.integers(0, len(allowed_targets)))
        ]
        template_id = int(
            allowed_templates[
                int(rng.integers(0, len(allowed_templates)))
            ]
        )

        target_vector = np.asarray(
            target.odor_vector,
            dtype=np.float64,
        )

        latent_event = _normalize(
            target_vector
            + 0.25 * template_offsets[template_id]
            + rng.normal(
                loc=0.0,
                scale=0.05,
                size=target_vector.shape[0],
            )
        )

        modality_vectors = {}

        for modality in MODALITIES:
            projected = (
                transformations[modality] @ latent_event
                + rng.normal(
                    loc=0.0,
                    scale=base_noise,
                    size=latent_event.shape[0],
                )
            )
            modality_vectors[modality] = _normalize(projected)

        events.append(
            SyntheticEvent(
                event_id=f"{split.value}-{index:06d}",
                split=split,
                template_id=template_id,
                target_item_id=target.item_id,
                target_family_id=target.family_id,
                text_vector=tuple(
                    float(x)
                    for x in modality_vectors["text"]
                ),
                image_vector=tuple(
                    float(x)
                    for x in modality_vectors["image"]
                ),
                audio_vector=tuple(
                    float(x)
                    for x in modality_vectors["audio"]
                ),
            )
        )

    return events


def _split_event_counts(
    *,
    event_count: int,
    validation_fraction: float,
    final_test_fraction: float,
) -> dict[SplitLabel, int]:
    validation_count = int(round(
        event_count * validation_fraction
    ))
    test_count = int(round(
        event_count * final_test_fraction
    ))
    training_count = (
        event_count
        - validation_count
        - test_count
    )

    if min(training_count, validation_count, test_count) < 1:
        raise SyntheticGenerationError(
            "Every pilot split must contain at least one event."
        )

    return {
        SplitLabel.TRAIN: training_count,
        SplitLabel.VALIDATION: validation_count,
        SplitLabel.OOD_TEST: test_count,
    }


def _orthogonal_matrix(
    dimension: int,
    rng: np.random.Generator,
) -> FloatArray:
    matrix = rng.normal(size=(dimension, dimension))
    q_matrix, r_matrix = np.linalg.qr(matrix)

    signs = np.sign(np.diag(r_matrix))
    signs[signs == 0.0] = 1.0

    return q_matrix * signs


def _normalize(vector: FloatArray) -> FloatArray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise SyntheticGenerationError(
            "A generated vector has a zero or nonfinite norm."
        )

    normalized = vector / norm

    if not np.all(np.isfinite(normalized)):
        raise SyntheticGenerationError(
            "A generated vector contains nonfinite values."
        )

    return normalized