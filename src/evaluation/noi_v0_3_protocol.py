"""Validation for the NOI v0.3 preimplementation protocol."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


EXPECTED_PROTOCOL_ID = "NOI-PROTOCOL-0.3"
EXPECTED_PARENT_COMMIT = (
    "942c0274a1513f781cd75327f493249fd6aa74af"
)
EXPECTED_HYPOTHESES = ("H6", "H7", "H8")
EXPECTED_SEEDS = tuple(range(1301, 1311))
EXPECTED_CONDITIONS = (
    "clean",
    "degraded_odor",
    "degraded_touch",
    "missing_touch",
    "missing_odor",
    "contradictory_modalities",
    "temporal_misalignment",
)
EXPECTED_SUPPORT_ALLOCATION = {
    "seen_item_events": 800,
    "known_family_unseen_item_events": 600,
    "unseen_family_events": 600,
}

REQUIRED_MAPPING_SECTIONS = (
    "protocol",
    "parent_release",
    "scope",
    "support_regimes",
    "hypotheses",
    "synthetic_dataset",
    "modalities",
    "paired_conditions",
    "systems",
    "support_gate",
    "touch_request_policy",
    "fusion_policy",
    "metrics",
    "statistical_controls",
    "development_stages",
    "integrity_controls",
    "planned_artifacts",
)


class NOIProtocolConfigurationError(ValueError):
    """Raised when the v0.3 protocol is missing or inconsistent."""


def load_noi_v0_3_protocol(
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Load and validate the v0.3 preimplementation protocol."""

    path = Path(protocol_path)

    if not path.is_file():
        raise NOIProtocolConfigurationError(
            f"Protocol file does not exist: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            configuration = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise NOIProtocolConfigurationError(
            "The v0.3 protocol file is not valid YAML."
        ) from error

    if not isinstance(configuration, dict):
        raise NOIProtocolConfigurationError(
            "The v0.3 protocol must be a mapping."
        )

    _validate_required_sections(configuration)
    _validate_identity(configuration)
    _validate_parent_release(configuration)
    _validate_scope(configuration)
    _validate_hypotheses(configuration)
    _validate_dataset(configuration)
    _validate_modalities(configuration)
    _validate_conditions(configuration)
    _validate_threshold_state(configuration)
    _validate_statistics(configuration)
    _validate_integrity(configuration)
    _validate_artifact_paths(configuration)

    return configuration


def _validate_required_sections(
    configuration: Mapping[str, Any],
) -> None:
    """Require every prespecified mapping section."""

    missing = [
        section
        for section in REQUIRED_MAPPING_SECTIONS
        if section not in configuration
    ]

    if missing:
        raise NOIProtocolConfigurationError(
            f"Missing required v0.3 sections: {missing}"
        )

    for section in REQUIRED_MAPPING_SECTIONS:
        if not isinstance(configuration[section], Mapping):
            raise NOIProtocolConfigurationError(
                f"Section {section!r} must be a mapping."
            )

    question = configuration.get("primary_question")

    if not isinstance(question, str) or not question.strip():
        raise NOIProtocolConfigurationError(
            "primary_question must be a nonempty string."
        )


def _validate_identity(
    configuration: Mapping[str, Any],
) -> None:
    """Validate protocol identity, status, project, and ownership."""

    protocol = _mapping(configuration, "protocol")

    _require_equal(
        protocol,
        "id",
        EXPECTED_PROTOCOL_ID,
        context="protocol",
    )
    _require_equal(
        protocol,
        "version",
        "0.3.0-preimplementation",
        context="protocol",
    )
    _require_equal(
        protocol,
        "status",
        "preimplementation",
        context="protocol",
    )
    _require_equal(
        protocol,
        "project",
        "Neuro-Olfactive Intelligence",
        context="protocol",
    )
    _require_equal(
        protocol,
        "owner",
        "GuardianX LLC",
        context="protocol",
    )

    authors = protocol.get("authors")

    if authors != ["Rana Al-Dahlake", "Jalal Alazirji"]:
        raise NOIProtocolConfigurationError(
            "Protocol authors do not match the prespecified authors."
        )


def _validate_parent_release(
    configuration: Mapping[str, Any],
) -> None:
    """Require the immutable v0.2.0 parent release."""

    parent = _mapping(configuration, "parent_release")

    _require_equal(
        parent,
        "version",
        "0.2.0",
        context="parent_release",
    )
    _require_equal(
        parent,
        "tag",
        "v0.2.0",
        context="parent_release",
    )
    _require_equal(
        parent,
        "commit",
        EXPECTED_PARENT_COMMIT,
        context="parent_release",
    )
    _require_equal(
        parent,
        "final_results_document",
        "docs/noi_v0.2_final_results.md",
        context="parent_release",
    )


def _validate_scope(
    configuration: Mapping[str, Any],
) -> None:
    """Prevent unsupported physical, clinical, or biological claims."""

    scope = _mapping(configuration, "scope")

    _require_equal(
        scope,
        "evidence_type",
        "deterministic synthetic computational simulation",
        context="scope",
    )

    false_fields = (
        "physical_tactile_sensor_validated",
        "physical_olfactory_sensor_validated",
        "clinical_claims_allowed",
        "biological_equivalence_claims_allowed",
        "deployment_claims_allowed",
    )

    for field in false_fields:
        _require_equal(
            scope,
            field,
            False,
            context="scope",
        )


def _validate_hypotheses(
    configuration: Mapping[str, Any],
) -> None:
    """Lock hypothesis numbering and success criteria."""

    hypotheses = _mapping(configuration, "hypotheses")

    if tuple(hypotheses) != EXPECTED_HYPOTHESES:
        raise NOIProtocolConfigurationError(
            "Hypotheses must remain H6, H7, and H8 in order."
        )

    h6 = _mapping(hypotheses, "H6")
    h7 = _mapping(hypotheses, "H7")
    h8 = _mapping(hypotheses, "H8")

    _require_equal(h6, "role", "primary", context="H6")
    _require_equal(
        _mapping(h6, "success"),
        "minimum_absolute_false_known_reduction",
        0.05,
        context="H6.success",
    )
    _require_equal(
        _mapping(h6, "success"),
        "maximum_seen_item_mrr_loss",
        0.02,
        context="H6.success",
    )

    _require_equal(h7, "role", "secondary", context="H7")
    _require_equal(
        _mapping(h7, "success"),
        "minimum_absolute_mrr_improvement",
        0.05,
        context="H7.success",
    )
    _require_equal(
        _mapping(h7, "success"),
        "minimum_relative_mrr_improvement",
        0.10,
        context="H7.success",
    )

    _require_equal(h8, "role", "secondary", context="H8")
    _require_equal(
        _mapping(h8, "success"),
        "minimum_absolute_false_confident_reduction",
        0.05,
        context="H8.success",
    )
    _require_equal(
        _mapping(h8, "success"),
        "maximum_clean_mrr_loss",
        0.02,
        context="H8.success",
    )


def _validate_dataset(
    configuration: Mapping[str, Any],
) -> None:
    """Validate seeds, event counts, allocations, and split controls."""

    dataset = _mapping(configuration, "synthetic_dataset")
    seeds = dataset.get("independent_seeds")

    if not isinstance(seeds, list):
        raise NOIProtocolConfigurationError(
            "independent_seeds must be a list."
        )

    if any(type(seed) is not int for seed in seeds):
        raise NOIProtocolConfigurationError(
            "Every independent seed must be an integer."
        )

    if tuple(seeds) != EXPECTED_SEEDS:
        raise NOIProtocolConfigurationError(
            "Independent seeds must remain 1301 through 1310."
        )

    events = _mapping(dataset, "events_per_seed")
    expected_events = {
        "total_base_events": 10000,
        "training_events": 7000,
        "validation_events": 1000,
        "final_test_latent_events": 2000,
    }

    if dict(events) != expected_events:
        raise NOIProtocolConfigurationError(
            "Events per seed do not match the prespecified allocation."
        )

    allocation = _mapping(
        dataset,
        "final_test_support_allocation",
    )

    if dict(allocation) != EXPECTED_SUPPORT_ALLOCATION:
        raise NOIProtocolConfigurationError(
            "Final-test support allocation has changed."
        )

    if sum(allocation.values()) != events["final_test_latent_events"]:
        raise NOIProtocolConfigurationError(
            "Final-test support counts do not sum to 2000."
        )

    controls = _mapping(dataset, "split_controls")

    false_controls = (
        "exact_event_overlap_allowed",
        "latent_event_overlap_allowed",
        "template_leakage_allowed",
        "unseen_family_test_leakage_allowed",
    )

    for field in false_controls:
        _require_equal(
            controls,
            field,
            False,
            context="split_controls",
        )

    _require_equal(
        controls,
        "validation_unknown_families_distinct_from_test",
        True,
        context="split_controls",
    )


def _validate_modalities(
    configuration: Mapping[str, Any],
) -> None:
    """Validate olfactory and simulated tactile dimensions."""

    modalities = _mapping(configuration, "modalities")
    olfactory = _mapping(modalities, "olfactory")
    tactile = _mapping(modalities, "tactile")

    _require_equal(
        olfactory,
        "representation_dimension",
        16,
        context="olfactory",
    )
    _require_equal(
        tactile,
        "representation_dimension",
        8,
        context="tactile",
    )
    _require_equal(
        tactile,
        "physical_sensor_claim",
        False,
        context="tactile",
    )

    components = _mapping(tactile, "components")

    if sum(components.values()) != 8:
        raise NOIProtocolConfigurationError(
            "Tactile component dimensions must sum to 8."
        )

    prohibited = tactile.get("prohibited_inputs")

    if not isinstance(prohibited, list):
        raise NOIProtocolConfigurationError(
            "tactile.prohibited_inputs must be a list."
        )

    required_prohibitions = {
        "target_label",
        "split_membership",
        "evaluation_outcome",
        "support_regime_label",
    }

    if set(prohibited) != required_prohibitions:
        raise NOIProtocolConfigurationError(
            "Tactile prohibited inputs have changed."
        )


def _validate_conditions(
    configuration: Mapping[str, Any],
) -> None:
    """Validate all seven paired stress conditions."""

    paired = _mapping(configuration, "paired_conditions")
    conditions = _mapping(paired, "conditions")

    _require_equal(
        paired,
        "views_per_applicable_latent_event",
        7,
        context="paired_conditions",
    )

    if tuple(conditions) != EXPECTED_CONDITIONS:
        raise NOIProtocolConfigurationError(
            "The seven paired conditions have changed."
        )

    _require_equal(
        paired,
        "statistical_unit",
        "latent_event_id",
        context="paired_conditions",
    )
    _require_equal(
        paired,
        "condition_views_are_independent_samples",
        False,
        context="paired_conditions",
    )

    conflict = _mapping(
        conditions,
        "contradictory_modalities",
    )
    _require_equal(
        conflict,
        "mismatch_rule",
        "different_target_and_different_family",
        context="contradictory_modalities",
    )


def _validate_threshold_state(
    configuration: Mapping[str, Any],
) -> None:
    """Require validation-derived thresholds to remain unlocked."""

    support_gate = _mapping(configuration, "support_gate")
    support_threshold = _mapping(
        support_gate,
        "threshold",
    )
    uncertainty_band = _mapping(
        support_gate,
        "uncertainty_band",
    )
    fusion = _mapping(configuration, "fusion_policy")
    reliability = _mapping(
        fusion,
        "reliability_threshold",
    )
    conflict = _mapping(
        fusion,
        "conflict_threshold",
    )

    threshold_fields = (
        ("support threshold", support_threshold, ("value",)),
        (
            "uncertainty band",
            uncertainty_band,
            ("lower", "upper"),
        ),
        ("reliability threshold", reliability, ("value",)),
        ("conflict threshold", conflict, ("value",)),
    )

    for name, threshold, value_fields in threshold_fields:
        for field in value_fields:
            if threshold.get(field) is not None:
                raise NOIProtocolConfigurationError(
                    f"{name} must remain null before protocol lock."
                )

        _require_equal(
            threshold,
            "status",
            "to_be_derived_and_locked",
            context=name,
        )
        _require_equal(
            threshold,
            "source",
            "validation_only",
            context=name,
        )


def _validate_statistics(
    configuration: Mapping[str, Any],
) -> None:
    """Validate paired statistical analysis settings."""

    controls = _mapping(
        configuration,
        "statistical_controls",
    )

    _require_equal(
        controls,
        "confidence_level",
        0.95,
        context="statistical_controls",
    )
    _require_equal(
        controls,
        "bootstrap_seed",
        4242,
        context="statistical_controls",
    )
    _require_equal(
        controls,
        "bootstrap_resamples",
        10000,
        context="statistical_controls",
    )

    paired = _mapping(controls, "paired_bootstrap")
    _require_equal(
        paired,
        "enabled",
        True,
        context="paired_bootstrap",
    )
    _require_equal(
        paired,
        "resampling_unit",
        "latent_event_id",
        context="paired_bootstrap",
    )

    correction = _mapping(
        controls,
        "multiple_comparison_correction",
    )
    _require_equal(
        correction,
        "method",
        "Holm",
        context="multiple_comparison_correction",
    )


def _validate_integrity(
    configuration: Mapping[str, Any],
) -> None:
    """Validate non-negotiable integrity controls."""

    controls = _mapping(
        configuration,
        "integrity_controls",
    )

    required_true = (
        "hash_configs",
        "hash_manifests",
        "hash_results",
        "hash_aggregates",
    )

    required_false = (
        "test_labels_for_training_allowed",
        "test_labels_for_calibration_allowed",
        "silent_seed_removal_allowed",
        "silent_condition_removal_allowed",
        "pooled_unstratified_support_reporting_allowed",
    )

    for field in required_true:
        _require_equal(
            controls,
            field,
            True,
            context="integrity_controls",
        )

    for field in required_false:
        _require_equal(
            controls,
            field,
            False,
            context="integrity_controls",
        )


def _validate_artifact_paths(
    configuration: Mapping[str, Any],
) -> None:
    """Validate planned versioned artifact locations."""

    artifacts = _mapping(
        configuration,
        "planned_artifacts",
    )

    expected = {
        "protocol_document": "docs/noi_v0.3_research_protocol.md",
        "protocol_config": "configs/noi_v0.3_protocol.yaml",
        "implementation_plan": "docs/noi_v0.3_implementation_plan.md",
        "final_results": "docs/noi_v0.3_final_results.md",
    }

    if dict(artifacts) != expected:
        raise NOIProtocolConfigurationError(
            "Planned v0.3 artifact paths have changed."
        )


def _mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Return one nested mapping or raise a configuration error."""

    value = mapping.get(key)

    if not isinstance(value, Mapping):
        raise NOIProtocolConfigurationError(
            f"{key!r} must be a mapping."
        )

    return value


def _require_equal(
    mapping: Mapping[str, Any],
    key: str,
    expected: Any,
    *,
    context: str,
) -> None:
    """Require one exact prespecified value."""

    actual = mapping.get(key)

    if actual != expected or type(actual) is not type(expected):
        raise NOIProtocolConfigurationError(
            f"{context}.{key} must be {expected!r}; "
            f"received {actual!r}."
        )
