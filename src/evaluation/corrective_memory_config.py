"""Validation for the preregistered corrective-memory evaluation."""

from __future__ import annotations

from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


class CorrectiveMemoryConfigurationError(ValueError):
    """Raised when the corrective-memory definition is invalid."""


def load_corrective_memory_configuration(
    configuration_path: str | Path,
    checksum_path: str | Path,
) -> dict[str, Any]:
    """Load and strictly validate the locked H2 evaluation definition."""

    path = Path(configuration_path)
    digest_path = Path(checksum_path)

    if not path.is_file():
        raise CorrectiveMemoryConfigurationError(
            f"Configuration file not found: {path}"
        )

    if not digest_path.is_file():
        raise CorrectiveMemoryConfigurationError(
            f"Checksum file not found: {digest_path}"
        )

    _verify_checksum(
        configuration_path=path,
        checksum_path=digest_path,
    )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            configuration = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise CorrectiveMemoryConfigurationError(
            "Could not load corrective-memory configuration."
        ) from error

    if not isinstance(configuration, dict):
        raise CorrectiveMemoryConfigurationError(
            "Configuration must be a mapping."
        )

    _validate_configuration(configuration)

    return configuration


def _verify_checksum(
    *,
    configuration_path: Path,
    checksum_path: Path,
) -> None:
    """Require the configuration to match its locked SHA-256 file."""

    checksum_text = checksum_path.read_text(
        encoding="utf-8"
    ).strip()

    parts = checksum_text.split()

    if len(parts) < 1:
        raise CorrectiveMemoryConfigurationError(
            "Checksum file is empty."
        )

    expected = parts[0].lower()

    if (
        len(expected) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected
        )
    ):
        raise CorrectiveMemoryConfigurationError(
            "Checksum file must contain a valid SHA-256 digest."
        )

    actual = sha256(
        configuration_path.read_bytes()
    ).hexdigest()

    if actual != expected:
        raise CorrectiveMemoryConfigurationError(
            "Configuration SHA-256 does not match the locked checksum."
        )


def _validate_configuration(
    configuration: dict[str, Any],
) -> None:
    """Validate every result-affecting locked field."""

    expected_top_level = {
        "schema",
        "governance",
        "source_dataset",
        "eligibility",
        "analysis_unit",
        "corruption",
        "arms",
        "retrieval",
        "primary_outcome",
        "secondary_outcomes",
        "old_memory_degradation",
        "statistics",
        "safeguards",
        "interpretation_limit",
    }

    if set(configuration) != expected_top_level:
        raise CorrectiveMemoryConfigurationError(
            "Top-level configuration fields differ from the locked schema."
        )

    schema = _mapping(
        configuration,
        "schema",
    )

    _require_equal(
        schema,
        "name",
        "NOI Corrective Memory Evaluation",
    )
    _require_equal(
        schema,
        "version",
        "0.1.0",
    )
    _require_equal(
        schema,
        "status",
        "preimplementation",
    )
    _require_equal(
        schema,
        "owner",
        "GUARDIANX LLC",
    )

    governance = _mapping(
        configuration,
        "governance",
    )

    _require_equal(
        governance,
        "hypothesis",
        "H2_corrective_updating",
    )
    _require_true(
        governance,
        "exploratory_status",
    )
    _require_true(
        governance,
        "negative_results_reported",
    )
    _require_true(
        governance,
        "post_result_setting_changes_prohibited",
    )

    source = _mapping(
        configuration,
        "source_dataset",
    )

    _require_equal(
        source,
        "event_count",
        200,
    )
    _require_equal(
        source,
        "generator_seed",
        1001,
    )
    _require_equal(
        source,
        "independent_ood_seed",
        9001,
    )

    eligibility = _mapping(
        configuration,
        "eligibility",
    )

    _require_equal(
        eligibility,
        "query_split",
        "validation",
    )
    _require_equal(
        eligibility,
        "memory_source_split",
        "train",
    )
    _require_true(
        eligibility,
        "require_query_target_represented_in_training_memory",
    )
    _require_true(
        eligibility,
        "unknown_validation_targets_excluded",
    )
    _require_true(
        eligibility,
        "use_all_eligible_targets",
    )
    _require_equal(
        eligibility,
        "expected_eligible_validation_events",
        15,
    )
    _require_equal(
        eligibility,
        "expected_eligible_targets",
        14,
    )
    _require_equal(
        eligibility,
        "expected_ineligible_validation_targets",
        5,
    )
    _require_true(
        eligibility,
        "target_selection_after_metric_inspection_prohibited",
    )

    analysis_unit = _mapping(
        configuration,
        "analysis_unit",
    )

    _require_equal(
        analysis_unit,
        "primary",
        "target_item_id",
    )
    _require_true(
        analysis_unit,
        "query_level_results_retained",
    )
    _require_true(
        analysis_unit,
        "repeated_queries_per_target_aggregated_by_mean",
    )
    _require_true(
        analysis_unit,
        "paired_arms",
    )

    corruption = _mapping(
        configuration,
        "corruption",
    )

    _require_true(
        corruption,
        "apply_to_all_training_memory_records_for_selected_target",
    )
    _require_equal(
        corruption,
        "decoy_candidate_pool",
        "lexicographically sorted training-represented odor target ids",
    )
    _require_equal(
        corruption,
        "decoy_selection_rule",
        "next target id cyclically after the true target id",
    )
    _require_true(
        corruption,
        "prohibit_true_target_as_decoy",
    )
    _require_true(
        corruption,
        "same_corruption_in_both_arms",
    )
    _require_equal(
        corruption,
        "corruption_time_offset_days",
        0,
    )

    arms = _mapping(
        configuration,
        "arms",
    )

    if set(arms) != {"no_update", "corrected"}:
        raise CorrectiveMemoryConfigurationError(
            "Exactly no_update and corrected arms are required."
        )

    no_update = _mapping(
        arms,
        "no_update",
    )
    corrected = _mapping(
        arms,
        "corrected",
    )

    _require_equal(
        no_update,
        "action_after_corruption",
        "none",
    )
    _require_equal(
        corrected,
        "correction_time_offset_days",
        1,
    )
    _require_true(
        corrected,
        "unique_correction_ids_required",
    )
    _require_true(
        corrected,
        "audit_record_required",
    )
    _require_true(
        corrected,
        "protocol_hash_required",
    )

    retrieval = _mapping(
        configuration,
        "retrieval",
    )

    _require_equal(
        retrieval,
        "primary_system",
        "memory_only",
    )
    _require_equal(
        retrieval,
        "alpha",
        0.0,
    )
    _require_equal(
        retrieval,
        "apply_temporal_decay",
        False,
    )
    _require_equal(
        retrieval,
        "top_k",
        10,
    )
    _require_equal(
        retrieval,
        "ood_oracle_used",
        False,
    )
    _require_equal(
        retrieval,
        "ood_tuning_used",
        False,
    )

    primary = _mapping(
        configuration,
        "primary_outcome",
    )

    _require_equal(
        primary,
        "metric",
        "mean reciprocal rank",
    )
    _require_equal(
        primary,
        "contrast",
        "corrected minus no_update",
    )

    success_rule = _mapping(
        primary,
        "success_rule",
    )

    _require_equal(
        success_rule,
        "minimum_absolute_mrr_improvement",
        0.05,
    )
    _require_true(
        success_rule,
        "paired_bootstrap_confidence_interval_must_exclude_zero",
    )

    degradation = _mapping(
        configuration,
        "old_memory_degradation",
    )

    _require_equal(
        degradation,
        "maximum_allowed_mean_degradation",
        0.02,
    )
    _require_true(
        degradation,
        "selected_target_queries_excluded_from_old_memory_set",
    )
    _require_true(
        degradation,
        "temporal_decay_disabled",
    )

    statistics = _mapping(
        configuration,
        "statistics",
    )

    _require_equal(
        statistics,
        "paired_unit",
        "target_item_id",
    )
    _require_equal(
        statistics,
        "bootstrap_seed",
        4242,
    )
    _require_equal(
        statistics,
        "bootstrap_resamples",
        10000,
    )
    _require_equal(
        statistics,
        "confidence_level",
        0.95,
    )
    _require_equal(
        statistics,
        "bootstrap_resample_unit",
        "target_item_id",
    )
    _require_true(
        statistics,
        "two_sided_interval",
    )
    _require_true(
        statistics,
        "report_all_target_level_differences",
    )

    safeguards = _mapping(
        configuration,
        "safeguards",
    )

    for key in (
        "train_validation_overlap_prohibited",
        "target_truth_hidden_from_retrieval_features",
        "correction_truth_used_only_for_controlled_intervention",
        "no_physical_emission",
        "policy_gate_not_used_to_change_rankings",
        "deterministic_replay_required",
        "immutable_result_records_required",
        "full_test_suite_required_before_release",
    ):
        _require_true(
            safeguards,
            key,
        )

    interpretation = configuration.get(
        "interpretation_limit"
    )

    if (
        not isinstance(interpretation, str)
        or not interpretation.strip()
    ):
        raise CorrectiveMemoryConfigurationError(
            "interpretation_limit must not be empty."
        )


def _mapping(
    parent: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = parent.get(key)

    if not isinstance(value, dict):
        raise CorrectiveMemoryConfigurationError(
            f"{key} must be a mapping."
        )

    return value


def _require_equal(
    mapping: dict[str, Any],
    key: str,
    expected: Any,
) -> None:
    value = mapping.get(key)

    if (
        isinstance(expected, float)
        and not isinstance(expected, bool)
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) != expected
        ):
            raise CorrectiveMemoryConfigurationError(
                f"{key} must equal {expected!r}."
            )
        return

    if value != expected or (
        isinstance(expected, bool)
        and not isinstance(value, bool)
    ):
        raise CorrectiveMemoryConfigurationError(
            f"{key} must equal {expected!r}."
        )


def _require_true(
    mapping: dict[str, Any],
    key: str,
) -> None:
    if mapping.get(key) is not True:
        raise CorrectiveMemoryConfigurationError(
            f"{key} must be true."
        )
