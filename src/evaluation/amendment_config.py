"""Validation for the preregistered NOI protocol amendment v0.2."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


EXPECTED_AMENDMENT_ID = "NOI-PROTOCOL-AMENDMENT-0.2"
EXPECTED_PARENT_PROTOCOL_SHA256 = (
    "e885f537b3209d7052d1517efc5be75324df39689cb4e54b38a241c95f22e512"
)

EXPECTED_TIERS = {
    "mild": {
        "shift_strength": 0.25,
        "seed": 7001,
    },
    "moderate": {
        "shift_strength": 0.50,
        "seed": 8001,
    },
    "severe": {
        "shift_strength": 1.00,
        "seed": 9001,
    },
}

REQUIRED_TOP_LEVEL_SECTIONS = (
    "amendment",
    "parent_protocol",
    "reason_for_amendment",
    "evidence_reviewed",
    "unchanged_commitments",
    "graded_ood_design",
    "confirmatory_evaluation",
    "statistical_analysis",
    "implementation_lock",
    "publication_language",
)


class AmendmentConfigurationError(ValueError):
    """Raised when the protocol amendment is missing or inconsistent."""


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""

    file_path = Path(path)

    if not file_path.is_file():
        raise AmendmentConfigurationError(
            f"File does not exist: {file_path}"
        )

    digest = sha256()

    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(65_536), b""):
            digest.update(block)

    return digest.hexdigest()


def load_amendment_configuration(
    amendment_path: str | Path,
    parent_protocol_path: str | Path,
) -> dict[str, Any]:
    """Load and validate the preregistered protocol amendment."""

    amendment_file = Path(amendment_path)
    parent_file = Path(parent_protocol_path)

    if not amendment_file.is_file():
        raise AmendmentConfigurationError(
            f"Amendment file does not exist: {amendment_file}"
        )

    try:
        with amendment_file.open("r", encoding="utf-8") as handle:
            configuration = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise AmendmentConfigurationError(
            "The amendment file is not valid YAML."
        ) from error

    if not isinstance(configuration, dict):
        raise AmendmentConfigurationError(
            "The amendment configuration must be a mapping."
        )

    _validate_required_sections(configuration)
    _validate_amendment_identity(configuration)
    _validate_parent_protocol(
        configuration,
        parent_protocol_path=parent_file,
    )
    _validate_graded_ood_design(configuration)
    _validate_confirmatory_evaluation(configuration)
    _validate_statistical_analysis(configuration)
    _validate_implementation_lock(configuration)
    _validate_unchanged_commitments(configuration)

    return configuration


def _validate_required_sections(
    configuration: Mapping[str, Any],
) -> None:
    """Require every locked amendment section."""

    missing = [
        section
        for section in REQUIRED_TOP_LEVEL_SECTIONS
        if section not in configuration
    ]

    if missing:
        raise AmendmentConfigurationError(
            f"Missing required amendment sections: {missing}"
        )

    for section in REQUIRED_TOP_LEVEL_SECTIONS:
        if not isinstance(configuration[section], Mapping):
            raise AmendmentConfigurationError(
                f"Section {section!r} must be a mapping."
            )


def _validate_amendment_identity(
    configuration: Mapping[str, Any],
) -> None:
    """Validate amendment identity, status, project, and ownership."""

    amendment = _mapping(
        configuration,
        "amendment",
    )

    if amendment.get("id") != EXPECTED_AMENDMENT_ID:
        raise AmendmentConfigurationError(
            f"Amendment id must be {EXPECTED_AMENDMENT_ID!r}."
        )

    if amendment.get("version") != "0.2.0":
        raise AmendmentConfigurationError(
            "Amendment version must be '0.2.0'."
        )

    if amendment.get("status") != "preimplementation":
        raise AmendmentConfigurationError(
            "Amendment status must remain 'preimplementation'."
        )

    if amendment.get("owner") != "GUARDIANX LLC":
        raise AmendmentConfigurationError(
            "Amendment owner must be GUARDIANX LLC."
        )

    if amendment.get("project") != "Neuro-Olfactive Interface":
        raise AmendmentConfigurationError(
            "Unexpected project name in the amendment."
        )


def _validate_parent_protocol(
    configuration: Mapping[str, Any],
    *,
    parent_protocol_path: Path,
) -> None:
    """Verify that the amendment references the locked parent protocol."""

    parent = _mapping(
        configuration,
        "parent_protocol",
    )

    declared_hash = parent.get("sha256")

    if declared_hash != EXPECTED_PARENT_PROTOCOL_SHA256:
        raise AmendmentConfigurationError(
            "The declared parent-protocol SHA-256 is not the locked value."
        )

    if parent.get("version_tag") != "protocol-v0.1-preimplementation":
        raise AmendmentConfigurationError(
            "Unexpected parent-protocol version tag."
        )

    actual_hash = file_sha256(parent_protocol_path)

    if actual_hash != declared_hash:
        raise AmendmentConfigurationError(
            "The parent protocol does not match its declared SHA-256."
        )


def _validate_graded_ood_design(
    configuration: Mapping[str, Any],
) -> None:
    """Validate the locked mild, moderate, and severe OOD tiers."""

    design = _mapping(
        configuration,
        "graded_ood_design",
    )

    _require_true(
        design,
        "enabled",
        context="graded_ood_design",
    )
    _require_true(
        design,
        "latent_event_policy",
        nested_key="reuse_same_latent_events_across_tiers",
    )
    _require_true(
        design,
        "latent_event_policy",
        nested_key="preserve_target_item",
    )
    _require_true(
        design,
        "latent_event_policy",
        nested_key="preserve_target_family",
    )
    _require_true(
        design,
        "latent_event_policy",
        nested_key="preserve_context_template",
    )
    _require_true(
        design,
        "latent_event_policy",
        nested_key="vary_only_shift_transformation_strength",
    )
    _require_true(
        design,
        "latent_event_policy",
        nested_key="observed_tier_rows_are_not_independent_units",
    )

    tiers = _mapping(
        design,
        "tiers",
    )

    if tuple(tiers) != tuple(EXPECTED_TIERS):
        raise AmendmentConfigurationError(
            "OOD tiers must appear exactly as mild, moderate, and severe."
        )

    observed_seeds: list[int] = []

    for tier_name, expected in EXPECTED_TIERS.items():
        tier = _mapping(
            tiers,
            tier_name,
        )

        strength = tier.get("shift_strength")
        seed = tier.get("seed")

        _require_finite_number(
            strength,
            label=f"{tier_name}.shift_strength",
        )

        if float(strength) != expected["shift_strength"]:
            raise AmendmentConfigurationError(
                f"{tier_name} shift_strength must be "
                f"{expected['shift_strength']}."
            )

        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise AmendmentConfigurationError(
                f"{tier_name} seed must be a nonnegative integer."
            )

        if seed != expected["seed"]:
            raise AmendmentConfigurationError(
                f"{tier_name} seed must be {expected['seed']}."
            )

        observed_seeds.append(seed)

    if len(set(observed_seeds)) != len(observed_seeds):
        raise AmendmentConfigurationError(
            "Every OOD tier must use a distinct seed."
        )

    transformation = _mapping(
        design,
        "transformation_rule",
    )

    _require_true(
        transformation,
        "deterministic",
        context="transformation_rule",
    )

    if transformation.get("clipping_or_hidden_adjustment_allowed") is not False:
        raise AmendmentConfigurationError(
            "Hidden adjustment or clipping must remain prohibited."
        )

    if transformation.get("target_identifiers_exposed_to_features") is not False:
        raise AmendmentConfigurationError(
            "Target identifiers must not be exposed to features."
        )


def _validate_confirmatory_evaluation(
    configuration: Mapping[str, Any],
) -> None:
    """Validate locked confirmatory counts and paired OOD analysis."""

    evaluation = _mapping(
        configuration,
        "confirmatory_evaluation",
    )

    expected_values = {
        "dataset_event_count": 10_000,
        "training_fraction": 0.70,
        "validation_fraction": 0.10,
        "base_ood_fraction": 0.20,
    }

    for key, expected in expected_values.items():
        value = evaluation.get(key)

        if value != expected:
            raise AmendmentConfigurationError(
                f"confirmatory_evaluation.{key} must be {expected}."
            )

    total_fraction = (
        float(evaluation["training_fraction"])
        + float(evaluation["validation_fraction"])
        + float(evaluation["base_ood_fraction"])
    )

    if abs(total_fraction - 1.0) > 1e-12:
        raise AmendmentConfigurationError(
            "Confirmatory split fractions must sum to 1.0."
        )

    ood = _mapping(
        evaluation,
        "ood_evaluation",
    )

    if ood.get("base_latent_ood_events") != 2_000:
        raise AmendmentConfigurationError(
            "base_latent_ood_events must be 2000."
        )

    if ood.get("severity_views_per_latent_event") != 3:
        raise AmendmentConfigurationError(
            "severity_views_per_latent_event must be 3."
        )

    if ood.get("expected_observed_ood_rows") != 6_000:
        raise AmendmentConfigurationError(
            "expected_observed_ood_rows must be 6000."
        )

    if ood.get("analysis_unit") != "latent_event_id":
        raise AmendmentConfigurationError(
            "The OOD analysis unit must be latent_event_id."
        )

    if ood.get("paired_across_severity_tiers") is not True:
        raise AmendmentConfigurationError(
            "OOD severity tiers must use paired evaluation."
        )


def _validate_statistical_analysis(
    configuration: Mapping[str, Any],
) -> None:
    """Validate bootstrap and oracle restrictions."""

    analysis = _mapping(
        configuration,
        "statistical_analysis",
    )

    if analysis.get("comparison_structure") != "paired by latent_event_id":
        raise AmendmentConfigurationError(
            "Statistical comparisons must be paired by latent_event_id."
        )

    if analysis.get("confidence_level") != 0.95:
        raise AmendmentConfigurationError(
            "confidence_level must be 0.95."
        )

    if analysis.get("bootstrap_unit") != "latent_event_id":
        raise AmendmentConfigurationError(
            "bootstrap_unit must be latent_event_id."
        )

    if analysis.get("bootstrap_resamples") != 10_000:
        raise AmendmentConfigurationError(
            "bootstrap_resamples must be 10000."
        )

    if analysis.get("bootstrap_seed") != 4242:
        raise AmendmentConfigurationError(
            "bootstrap_seed must be 4242."
        )

    oracle = _mapping(
        analysis,
        "oracle_policy",
    )

    required_true = (
        "allowed_for_diagnostics_only",
        "excluded_from_baseline_superiority_tests",
        "excluded_from_deployment_claims",
        "must_be_labeled_as_ood_calibrated",
    )

    for key in required_true:
        if oracle.get(key) is not True:
            raise AmendmentConfigurationError(
                f"oracle_policy.{key} must be true."
            )


def _validate_implementation_lock(
    configuration: Mapping[str, Any],
) -> None:
    """Require pre-confirmatory implementation protections."""

    lock = _mapping(
        configuration,
        "implementation_lock",
    )

    required_true = (
        "implementation_must_follow_amendment",
        "tests_required_before_confirmatory_run",
        "leakage_audit_required",
        "deterministic_export_required",
        "hashes_required",
        "confirmatory_results_must_not_be_inspected_before_lock",
    )

    for key in required_true:
        if lock.get(key) is not True:
            raise AmendmentConfigurationError(
                f"implementation_lock.{key} must be true."
            )


def _validate_unchanged_commitments(
    configuration: Mapping[str, Any],
) -> None:
    """Ensure that the amendment does not hide or weaken prior commitments."""

    commitments = _mapping(
        configuration,
        "unchanged_commitments",
    )

    required_true = (
        "original_negative_result_retained",
        "severe_ood_stress_test_retained",
        "original_hypotheses_unchanged",
        "original_success_thresholds_unchanged",
        "locked_metrics_unchanged",
        "safety_policy_unchanged",
        "leakage_controls_unchanged",
        "synthetic_scope_unchanged",
    )

    for key in required_true:
        if commitments.get(key) is not True:
            raise AmendmentConfigurationError(
                f"unchanged_commitments.{key} must be true."
            )

    prohibited_claims = commitments.get("prohibited_claims")

    if (
        not isinstance(prohibited_claims, list)
        or not prohibited_claims
        or any(
            not isinstance(claim, str) or not claim.strip()
            for claim in prohibited_claims
        )
    ):
        raise AmendmentConfigurationError(
            "prohibited_claims must be a nonempty list of strings."
        )


def _mapping(
    parent: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Return a required nested mapping."""

    value = parent.get(key)

    if not isinstance(value, Mapping):
        raise AmendmentConfigurationError(
            f"{key!r} must be a mapping."
        )

    return value


def _require_true(
    parent: Mapping[str, Any],
    key: str,
    *,
    context: str | None = None,
    nested_key: str | None = None,
) -> None:
    """Require a locked boolean value to be exactly true."""

    if nested_key is None:
        value = parent.get(key)
        label = f"{context}.{key}" if context else key
    else:
        nested = _mapping(parent, key)
        value = nested.get(nested_key)
        label = f"{key}.{nested_key}"

    if value is not True:
        raise AmendmentConfigurationError(
            f"{label} must be true."
        )


def _require_finite_number(
    value: Any,
    *,
    label: str,
) -> None:
    """Require a real finite numeric value."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise AmendmentConfigurationError(
            f"{label} must be a finite number."
        )