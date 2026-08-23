"""Replay-safe source generation for paired graded-OOD evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.evaluation.amendment_config import (
    file_sha256,
    load_amendment_configuration,
)
from src.evaluation.graded_ood_generator import (
    GradedOODDataset,
    PairedOODSource,
    generate_graded_ood_dataset,
)
from src.evaluation.split_plan import create_split_plan
from src.evaluation.synthetic_config import (
    load_synthetic_configuration,
)
from src.evaluation.synthetic_generator import (
    MODALITIES,
    _generate_odor_targets,
    _generate_template_offsets,
    _normalize,
    _orthogonal_matrix,
    _split_event_counts,
    generate_synthetic_pilot,
)
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
    SyntheticEvent,
    SyntheticOdorTarget,
)


EXPECTED_GENERATION_DEFINITION_SHA256 = (
    "45d247dfb18152d64701b3f088be950aeece07ff3592eba82e702021b2ea3cb3"
)
EXPECTED_AMENDMENT_SHA256 = (
    "5292e0208576a79342923fdf72855de39bffe63331721fff01b94ae35aa39f7b"
)


class PairedOODSourceGenerationError(ValueError):
    """Raised when replay-safe paired sources cannot be generated."""


@dataclass(frozen=True)
class PairedGradedOODBundle:
    """Original synthetic library plus replay-verified graded OOD data."""

    odor_targets: tuple[SyntheticOdorTarget, ...]
    graded_dataset: GradedOODDataset
    original_dataset: SyntheticDataset
    source_count: int
    generator_version: str
    generator_seed: int
    ood_seed: int
    severe_reference_verified: bool
    generation_definition_sha256: str
    amendment_sha256: str

    def __post_init__(self) -> None:
        if not self.odor_targets:
            raise PairedOODSourceGenerationError(
                "odor_targets must not be empty."
            )

        if not isinstance(self.graded_dataset, GradedOODDataset):
            raise PairedOODSourceGenerationError(
                "graded_dataset must be a GradedOODDataset."
            )

        if not isinstance(self.original_dataset, SyntheticDataset):
            raise PairedOODSourceGenerationError(
                "original_dataset must be a SyntheticDataset."
            )

        if (
            isinstance(self.source_count, bool)
            or not isinstance(self.source_count, int)
            or self.source_count < 1
        ):
            raise PairedOODSourceGenerationError(
                "source_count must be a positive integer."
            )

        if (
            self.source_count
            != self.graded_dataset.latent_event_count
        ):
            raise PairedOODSourceGenerationError(
                "source_count must equal the graded latent-event count."
            )

        if (
            self.graded_dataset.observed_event_count
            != self.source_count * 3
        ):
            raise PairedOODSourceGenerationError(
                "Every source must produce exactly three observed views."
            )

        if self.severe_reference_verified is not True:
            raise PairedOODSourceGenerationError(
                "Severe replay verification must pass."
            )

        if (
            self.generation_definition_sha256
            != EXPECTED_GENERATION_DEFINITION_SHA256
        ):
            raise PairedOODSourceGenerationError(
                "Unexpected generation-definition SHA-256."
            )

        if self.amendment_sha256 != EXPECTED_AMENDMENT_SHA256:
            raise PairedOODSourceGenerationError(
                "Unexpected amendment SHA-256."
            )

        if self.odor_targets != self.original_dataset.odor_targets:
            raise PairedOODSourceGenerationError(
                "The bundle odor library must match the original dataset."
            )


def generate_paired_graded_ood_bundle(
    synthetic_configuration_path: str | Path,
    protocol_path: str | Path,
    amendment_path: str | Path,
    generation_definition_path: str | Path,
    *,
    event_count: int = 200,
) -> PairedGradedOODBundle:
    """Replay original severe OOD events and create paired graded views."""

    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 10
    ):
        raise PairedOODSourceGenerationError(
            "event_count must be an integer of at least 10."
        )

    synthetic_path = Path(synthetic_configuration_path)
    protocol_file = Path(protocol_path)
    amendment_file = Path(amendment_path)
    definition_file = Path(generation_definition_path)

    definition_hash = file_sha256(definition_file)

    if definition_hash != EXPECTED_GENERATION_DEFINITION_SHA256:
        raise PairedOODSourceGenerationError(
            "The graded OOD generation definition failed SHA-256 "
            "verification."
        )

    amendment_hash = file_sha256(amendment_file)

    if amendment_hash != EXPECTED_AMENDMENT_SHA256:
        raise PairedOODSourceGenerationError(
            "The protocol amendment failed SHA-256 verification."
        )

    amendment = load_amendment_configuration(
        amendment_file,
        protocol_file,
    )

    definition = _load_generation_definition(
        definition_file
    )
    _validate_generation_definition(
        definition,
        amendment_hash=amendment_hash,
    )

    configuration = load_synthetic_configuration(
        synthetic_path,
        protocol_file,
    )

    original_dataset = generate_synthetic_pilot(
        synthetic_path,
        protocol_file,
        event_count=event_count,
    )

    sources, replayed_targets = _replay_paired_sources(
        configuration=configuration,
        event_count=event_count,
    )

    if tuple(replayed_targets) != original_dataset.odor_targets:
        raise PairedOODSourceGenerationError(
            "Replayed odor targets do not match the original dataset."
        )

    original_ood_events = tuple(
        event
        for event in original_dataset.events
        if event.split is SplitLabel.OOD_TEST
    )

    replayed_severe_events = tuple(
        source.severe_event
        for source in sources
    )

    if replayed_severe_events != original_ood_events:
        mismatch = _first_mismatch(
            replayed_severe_events,
            original_ood_events,
        )
        raise PairedOODSourceGenerationError(
            "Severe OOD replay does not match the original generator"
            f"{mismatch}."
        )

    graded_dataset = generate_graded_ood_dataset(
        sources,
        amendment,
    )

    return PairedGradedOODBundle(
        odor_targets=tuple(replayed_targets),
        graded_dataset=graded_dataset,
        original_dataset=original_dataset,
        source_count=len(sources),
        generator_version=original_dataset.generator_version,
        generator_seed=original_dataset.generator_seed,
        ood_seed=original_dataset.ood_seed,
        severe_reference_verified=True,
        generation_definition_sha256=definition_hash,
        amendment_sha256=amendment_hash,
    )


def _load_generation_definition(
    path: Path,
) -> dict[str, Any]:
    """Load the locked paired-generation definition."""

    if not path.is_file():
        raise PairedOODSourceGenerationError(
            f"Generation definition does not exist: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            definition = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise PairedOODSourceGenerationError(
            "The generation definition is not valid YAML."
        ) from error

    if not isinstance(definition, dict):
        raise PairedOODSourceGenerationError(
            "The generation definition must be a mapping."
        )

    return definition


def _validate_generation_definition(
    definition: dict[str, Any],
    *,
    amendment_hash: str,
) -> None:
    """Validate safeguards required by the locked definition."""

    try:
        schema = definition["schema"]
        governance = definition["governance"]
        source_policy = definition["paired_source_generation"]
        identity_policy = source_policy["identity_reference"]
        severe_policy = source_policy["severe_reference"]
        latent_policy = source_policy["latent_event"]
        leakage = definition["leakage_controls"]
        verification = definition["verification"]
    except (KeyError, TypeError) as error:
        raise PairedOODSourceGenerationError(
            "The generation definition is missing required sections."
        ) from error

    if schema.get("version") != "1.0.0":
        raise PairedOODSourceGenerationError(
            "Generation schema version must be 1.0.0."
        )

    if schema.get("status") != "preimplementation":
        raise PairedOODSourceGenerationError(
            "Generation definition status must be preimplementation."
        )

    if schema.get("owner") != "GUARDIANX LLC":
        raise PairedOODSourceGenerationError(
            "Generation definition owner must be GUARDIANX LLC."
        )

    if governance.get("amendment_sha256") != amendment_hash:
        raise PairedOODSourceGenerationError(
            "Generation definition references the wrong amendment hash."
        )

    if governance.get("preserve_original_pilot_results") is not True:
        raise PairedOODSourceGenerationError(
            "Original pilot results must be preserved."
        )

    if governance.get("modify_original_pilot_generator") is not False:
        raise PairedOODSourceGenerationError(
            "The original pilot generator must not be modified."
        )

    required_source_truth = (
        "reuse_same_latent_event",
        "preserve_target_item",
        "preserve_target_family",
        "preserve_context_template",
    )

    for key in required_source_truth:
        if source_policy.get(key) is not True:
            raise PairedOODSourceGenerationError(
                f"paired_source_generation.{key} must be true."
            )

    if source_policy.get("analysis_unit") != "latent_event_id":
        raise PairedOODSourceGenerationError(
            "The analysis unit must be latent_event_id."
        )

    if latent_policy.get("sample_once_per_latent_event") is not True:
        raise PairedOODSourceGenerationError(
            "Each latent event must be sampled exactly once."
        )

    if latent_policy.get("template_offset_weight") != 0.25:
        raise PairedOODSourceGenerationError(
            "template_offset_weight must be 0.25."
        )

    if latent_policy.get("latent_noise_standard_deviation") != 0.05:
        raise PairedOODSourceGenerationError(
            "latent_noise_standard_deviation must be 0.05."
        )

    if identity_policy.get("added_observation_noise") is not False:
        raise PairedOODSourceGenerationError(
            "Identity reference observation noise must be disabled."
        )

    if (
        identity_policy.get("observation_noise_standard_deviation")
        != 0.0
    ):
        raise PairedOODSourceGenerationError(
            "Identity observation noise must be 0.0."
        )

    if severe_policy.get("added_observation_noise") is not True:
        raise PairedOODSourceGenerationError(
            "Severe observation noise must remain enabled."
        )

    if (
        severe_policy.get("observation_noise_standard_deviation")
        != 0.10
    ):
        raise PairedOODSourceGenerationError(
            "Severe observation noise must be 0.10."
        )

    if severe_policy.get("require_original_severe_direction") is not True:
        raise PairedOODSourceGenerationError(
            "Original severe replay verification must be required."
        )

    required_false = (
        "expose_target_identifier_to_features",
        "expose_family_identifier_to_features",
        "derive_identity_reference_from_target_label",
        "pair_by_target_identifier_only",
    )

    for key in required_false:
        if leakage.get(key) is not False:
            raise PairedOODSourceGenerationError(
                f"leakage_controls.{key} must be false."
            )

    required_true = (
        "reuse_same_sampled_latent_event",
        "duplicate_observed_identifiers_prohibited",
        "cross_latent_feature_duplicates_prohibited",
    )

    for key in required_true:
        if leakage.get(key) is not True:
            raise PairedOODSourceGenerationError(
                f"leakage_controls.{key} must be true."
            )

    if verification.get("original_pilot_hashes_must_remain_unchanged") is not True:
        raise PairedOODSourceGenerationError(
            "Original pilot hashes must remain unchanged."
        )

    if verification.get("severe_view_must_equal_normalized_severe_reference") is not True:
        raise PairedOODSourceGenerationError(
            "Severe-view equality verification must remain enabled."
        )


def _replay_paired_sources(
    *,
    configuration: dict[str, Any],
    event_count: int,
) -> tuple[
    tuple[PairedOODSource, ...],
    tuple[SyntheticOdorTarget, ...],
]:
    """Replay the original RNG stream and retain identity references."""

    dataset_config = configuration["dataset"]
    split_config = configuration["splits"]
    randomness = configuration["randomness"]

    dimension = dataset_config["latent_dimension"]

    if dimension != dataset_config["modality_dimension"]:
        raise PairedOODSourceGenerationError(
            "Latent and modality dimensions must match."
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

    targets = _generate_odor_targets(
        family_count=dataset_config["odor_families"],
        target_count=dataset_config["odor_targets"],
        dimension=dimension,
        split_plan=split_plan,
        id_rng=id_rng,
        ood_rng=ood_rng,
    )

    template_offsets = _generate_template_offsets(
        template_count=dataset_config["context_templates"],
        dimension=dimension,
        split_plan=split_plan,
        id_rng=id_rng,
        ood_rng=ood_rng,
    )

    id_transforms = {
        modality: _orthogonal_matrix(
            dimension,
            id_rng,
        )
        for modality in MODALITIES
    }

    ood_transforms = {
        modality: _orthogonal_matrix(
            dimension,
            ood_rng,
        )
        for modality in MODALITIES
    }

    counts = _split_event_counts(
        event_count=event_count,
        validation_fraction=split_config["validation_fraction"],
        final_test_fraction=split_config["final_test_fraction"],
    )

    allowed_family_set = set(
        split_plan.ood_test_families
    )

    allowed_targets = [
        target
        for target in targets
        if target.family_id in allowed_family_set
    ]

    allowed_templates = split_plan.ood_test_templates
    sources: list[PairedOODSource] = []

    for index in range(counts[SplitLabel.OOD_TEST]):
        target = allowed_targets[
            int(ood_rng.integers(0, len(allowed_targets)))
        ]

        template_id = int(
            allowed_templates[
                int(
                    ood_rng.integers(
                        0,
                        len(allowed_templates),
                    )
                )
            ]
        )

        target_vector = np.asarray(
            target.odor_vector,
            dtype=np.float64,
        )

        latent_event = _normalize(
            target_vector
            + 0.25 * template_offsets[template_id]
            + ood_rng.normal(
                loc=0.0,
                scale=0.05,
                size=target_vector.shape[0],
            )
        )

        identity_vectors = {
            modality: _normalize(
                id_transforms[modality] @ latent_event
            )
            for modality in MODALITIES
        }

        severe_vectors = {}

        for modality in MODALITIES:
            severe_vectors[modality] = _normalize(
                ood_transforms[modality] @ latent_event
                + ood_rng.normal(
                    loc=0.0,
                    scale=0.10,
                    size=latent_event.shape[0],
                )
            )

        original_event_id = (
            f"{SplitLabel.OOD_TEST.value}-{index:06d}"
        )
        latent_event_id = f"latent-{original_event_id}"

        identity_event = SyntheticEvent(
            event_id=f"{original_event_id}::identity",
            split=SplitLabel.OOD_TEST,
            template_id=template_id,
            target_item_id=target.item_id,
            target_family_id=target.family_id,
            text_vector=tuple(
                float(value)
                for value in identity_vectors["text"]
            ),
            image_vector=tuple(
                float(value)
                for value in identity_vectors["image"]
            ),
            audio_vector=tuple(
                float(value)
                for value in identity_vectors["audio"]
            ),
        )

        severe_event = SyntheticEvent(
            event_id=original_event_id,
            split=SplitLabel.OOD_TEST,
            template_id=template_id,
            target_item_id=target.item_id,
            target_family_id=target.family_id,
            text_vector=tuple(
                float(value)
                for value in severe_vectors["text"]
            ),
            image_vector=tuple(
                float(value)
                for value in severe_vectors["image"]
            ),
            audio_vector=tuple(
                float(value)
                for value in severe_vectors["audio"]
            ),
        )

        sources.append(
            PairedOODSource(
                latent_event_id=latent_event_id,
                identity_event=identity_event,
                severe_event=severe_event,
            )
        )

    return tuple(sources), tuple(targets)


def _first_mismatch(
    replayed: tuple[SyntheticEvent, ...],
    original: tuple[SyntheticEvent, ...],
) -> str:
    """Describe the first replay mismatch without exposing feature data."""

    if len(replayed) != len(original):
        return (
            f": replayed {len(replayed)} events but "
            f"original contains {len(original)}"
        )

    for index, (left, right) in enumerate(
        zip(replayed, original, strict=True)
    ):
        if left != right:
            return (
                f" at index {index} "
                f"({left.event_id} versus {right.event_id})"
            )

    return ""
