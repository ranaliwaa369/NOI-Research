"""Validation for the independent NOI synthetic-data specification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class SyntheticConfigurationError(ValueError):
    """Raised when synthetic-data safeguards are invalid."""


def load_synthetic_configuration(
    configuration_path: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Load and cross-check synthetic configuration and research protocol."""

    configuration = _load_yaml(
        configuration_path,
        label="synthetic-data configuration",
    )
    protocol = _load_yaml(
        protocol_path,
        label="research protocol",
    )

    validate_synthetic_configuration(configuration, protocol)
    return configuration


def _load_yaml(
    path: str | Path,
    *,
    label: str,
) -> dict[str, Any]:
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Missing {label}: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise SyntheticConfigurationError(
            f"The {label} must be a YAML mapping."
        )

    return content


def validate_synthetic_configuration(
    configuration: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    """Enforce preregistered dataset and leakage-control requirements."""

    required_sections = {
        "generator",
        "limitations",
        "dataset",
        "randomness",
        "ground_truth",
        "generation_process",
        "splits",
        "evaluation_conditions",
        "ood_generation",
        "leakage_checks",
        "outputs",
    }

    missing = required_sections - configuration.keys()
    if missing:
        raise SyntheticConfigurationError(
            f"Missing synthetic configuration sections: {sorted(missing)}"
        )

    generator = configuration["generator"]

    if generator.get("model_independent") is not True:
        raise SyntheticConfigurationError(
            "The generator must be declared model-independent."
        )

    if generator.get("purpose") != "Implementation validation only":
        raise SyntheticConfigurationError(
            "Synthetic data must be limited to implementation validation."
        )

    dataset = configuration["dataset"]
    protocol_dataset = protocol["dataset"]

    if dataset.get("total_events") != protocol_dataset.get(
        "planned_events"
    ):
        raise SyntheticConfigurationError(
            "total_events must match the preregistered protocol."
        )

    if dataset.get("odor_targets", 0) < protocol_dataset.get(
        "minimum_odor_targets",
        0,
    ):
        raise SyntheticConfigurationError(
            "odor_targets is below the preregistered minimum."
        )

    odor_families = dataset.get("odor_families")
    odor_targets = dataset.get("odor_targets")

    if (
        not isinstance(odor_families, int)
        or odor_families < 2
        or odor_targets % odor_families != 0
    ):
        raise SyntheticConfigurationError(
            "odor_targets must divide evenly across at least two families."
        )

    for dimension_name in (
        "latent_dimension",
        "modality_dimension",
    ):
        dimension = dataset.get(dimension_name)

        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 2
        ):
            raise SyntheticConfigurationError(
                f"{dimension_name} must be an integer of at least 2."
            )

    configured_seeds = configuration["randomness"].get(
        "training_seeds"
    )
    protocol_seeds = protocol_dataset.get(
        "independent_training_seeds"
    )

    if configured_seeds != protocol_seeds:
        raise SyntheticConfigurationError(
            "Training seeds must exactly match the research protocol."
        )

    generator_seed = configuration["randomness"].get(
        "generator_seed"
    )
    ood_seed = configuration["randomness"].get(
        "independent_ood_seed"
    )

    if generator_seed == ood_seed:
        raise SyntheticConfigurationError(
            "ID and OOD generators must use different seeds."
        )

    ground_truth = configuration["ground_truth"]

    if ground_truth.get(
        "expose_target_identifier_to_features"
    ) is not False:
        raise SyntheticConfigurationError(
            "Target identifiers must not be exposed to features."
        )

    if ground_truth.get(
        "include_item_name_in_text_features"
    ) is not False:
        raise SyntheticConfigurationError(
            "Item names must not leak into text features."
        )

    if ground_truth.get(
        "include_family_identifier_in_metadata"
    ) is not False:
        raise SyntheticConfigurationError(
            "Odor-family identifiers must not leak into metadata."
        )

    splits = configuration["splits"]
    protocol_splits = protocol["splits"]

    split_pairs = {
        "odor_family_held_out_fraction":
            "odor_family_held_out_fraction",
        "context_template_held_out_fraction":
            "context_template_held_out_fraction",
    }

    for configured_name, protocol_name in split_pairs.items():
        if splits.get(configured_name) != protocol_splits.get(
            protocol_name
        ):
            raise SyntheticConfigurationError(
                f"{configured_name} must match the research protocol."
            )

    if splits.get("random_row_only_split_prohibited") is not True:
        raise SyntheticConfigurationError(
            "Random-row-only splitting must remain prohibited."
        )

    if splits.get("odor_family_leakage_prohibited") is not True:
        raise SyntheticConfigurationError(
            "Odor-family leakage must remain prohibited."
        )

    if splits.get("context_template_leakage_prohibited") is not True:
        raise SyntheticConfigurationError(
            "Context-template leakage must remain prohibited."
        )

    ood = configuration["ood_generation"]

    required_ood_guarantees = (
        "use_independent_seed",
        "resample_modality_transformations",
        "hold_out_odor_families",
        "hold_out_context_templates",
    )

    for guarantee in required_ood_guarantees:
        if ood.get(guarantee) is not True:
            raise SyntheticConfigurationError(
                f"OOD guarantee must remain enabled: {guarantee}"
            )

    leakage_checks = configuration["leakage_checks"]

    if not all(leakage_checks.values()):
        raise SyntheticConfigurationError(
            "Every leakage check must remain enabled."
        )