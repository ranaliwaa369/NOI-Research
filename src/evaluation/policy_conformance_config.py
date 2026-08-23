"""Validation for the locked policy-conformance evaluation."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


ALLOWED_OUTCOMES = {
    "ALLOW",
    "BLOCK",
    "REQUIRE_MISSING_INFORMATION",
}

EXPECTED_OUTCOME_COUNTS = {
    "ALLOW": 4,
    "BLOCK": 15,
    "REQUIRE_MISSING_INFORMATION": 7,
}

REQUEST_FIELDS = (
    "request_id",
    "description",
    "item_id",
    "concentration_ppm",
    "duration_seconds",
    "environment_volume_m3",
    "ventilation_ach",
    "user_consent",
    "expected_outcome",
)


class PolicyConformanceConfigurationError(ValueError):
    """Raised when the policy-conformance definition is invalid."""


def load_policy_conformance_configuration(
    configuration_path: str | Path,
    checksum_path: str | Path,
) -> dict[str, Any]:
    """Load and strictly validate the preregistered H3 suite."""

    path = Path(configuration_path)
    digest_path = Path(checksum_path)

    if not path.is_file():
        raise PolicyConformanceConfigurationError(
            f"Configuration file not found: {path}"
        )

    if not digest_path.is_file():
        raise PolicyConformanceConfigurationError(
            f"Checksum file not found: {digest_path}"
        )

    _verify_checksum(path, digest_path)

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            configuration = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise PolicyConformanceConfigurationError(
            "Could not load policy-conformance configuration."
        ) from error

    if not isinstance(configuration, dict):
        raise PolicyConformanceConfigurationError(
            "Configuration must be a mapping."
        )

    _validate_configuration(configuration)

    return configuration


def _verify_checksum(
    configuration_path: Path,
    checksum_path: Path,
) -> None:
    checksum_text = checksum_path.read_text(
        encoding="utf-8"
    ).strip()
    parts = checksum_text.split()

    if not parts:
        raise PolicyConformanceConfigurationError(
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
        raise PolicyConformanceConfigurationError(
            "Checksum file must contain a valid SHA-256 digest."
        )

    actual = sha256(
        configuration_path.read_bytes()
    ).hexdigest()

    if actual != expected:
        raise PolicyConformanceConfigurationError(
            "Configuration SHA-256 does not match the locked checksum."
        )


def _validate_configuration(
    configuration: dict[str, Any],
) -> None:
    expected_top_level = {
        "schema",
        "governance",
        "policy",
        "metrics",
        "case_generation",
        "cases",
        "safeguards",
        "interpretation_limit",
    }

    if set(configuration) != expected_top_level:
        raise PolicyConformanceConfigurationError(
            "Top-level fields differ from the locked schema."
        )

    schema = _mapping(configuration, "schema")
    _equal(
        schema,
        "name",
        "NOI Deterministic Policy Conformance Evaluation",
    )
    _equal(schema, "version", "0.1.0")
    _equal(schema, "status", "preimplementation")
    _equal(schema, "owner", "GUARDIANX LLC")

    governance = _mapping(
        configuration,
        "governance",
    )
    _equal(
        governance,
        "hypothesis",
        "H3_policy_conformance",
    )
    _true(governance, "exploratory_status")
    _true(governance, "negative_results_reported")
    _true(
        governance,
        "post_result_case_changes_prohibited",
    )

    policy = _mapping(configuration, "policy")
    _equal(
        policy,
        "configuration_path",
        "configs/policy_rules.yaml",
    )
    _true(policy, "simulation_only_required")
    _equal(
        policy,
        "default_action_required",
        "BLOCK",
    )

    allowed_outcomes = policy.get(
        "allowed_outcomes"
    )

    if (
        not isinstance(allowed_outcomes, list)
        or set(allowed_outcomes) != ALLOWED_OUTCOMES
        or len(allowed_outcomes) != 3
    ):
        raise PolicyConformanceConfigurationError(
            "allowed_outcomes differ from the locked set."
        )

    _equal(
        policy,
        "physical_emission_performed",
        False,
    )

    metrics = _mapping(configuration, "metrics")
    _equal(
        metrics,
        "violation_false_allow_target",
        0,
    )
    _equal(
        metrics,
        "maximum_false_block_rate",
        0.01,
    )
    _equal(
        metrics,
        "exact_conformance_target",
        1.0,
    )
    _equal(
        metrics,
        "policy_coverage_target",
        1.0,
    )

    generation = _mapping(
        configuration,
        "case_generation",
    )
    _true(generation, "deterministic_order")
    _true(generation, "unique_request_ids_required")
    _equal(generation, "expected_case_count", 26)
    _equal(generation, "boundary_epsilon", 0.000001)
    _true(
        generation,
        "values_are_simulated_not_exposure_limits",
    )

    cases = configuration.get("cases")

    if not isinstance(cases, list):
        raise PolicyConformanceConfigurationError(
            "cases must be a list."
        )

    if len(cases) != 26:
        raise PolicyConformanceConfigurationError(
            "cases must contain exactly 26 requests."
        )

    request_ids = []

    for index, case in enumerate(cases):
        _validate_case(case, index=index)
        request_ids.append(case["request_id"])

    if len(set(request_ids)) != len(request_ids):
        raise PolicyConformanceConfigurationError(
            "request_id values must be unique."
        )

    outcome_counts = Counter(
        case["expected_outcome"]
        for case in cases
    )

    if dict(outcome_counts) != EXPECTED_OUTCOME_COUNTS:
        raise PolicyConformanceConfigurationError(
            "Expected outcome counts differ from the locked suite."
        )

    safeguards = _mapping(
        configuration,
        "safeguards",
    )

    for key in (
        "policy_configuration_hash_required",
        "expected_outcome_locked_before_execution",
        "request_order_locked",
        "no_case_removal_after_results",
        "no_physical_emission",
        "no_chemical_safety_claim",
        "no_clinical_safety_claim",
        "exact_decision_audit_retained",
        "deterministic_replay_required",
        "full_test_suite_required_before_release",
    ):
        _true(safeguards, key)

    interpretation = configuration.get(
        "interpretation_limit"
    )

    if (
        not isinstance(interpretation, str)
        or not interpretation.strip()
    ):
        raise PolicyConformanceConfigurationError(
            "interpretation_limit must not be empty."
        )


def _validate_case(
    case: Any,
    *,
    index: int,
) -> None:
    if not isinstance(case, dict):
        raise PolicyConformanceConfigurationError(
            f"cases[{index}] must be a mapping."
        )

    if set(case) != set(REQUEST_FIELDS):
        raise PolicyConformanceConfigurationError(
            f"cases[{index}] fields differ from the locked schema."
        )

    for key in (
        "request_id",
        "description",
        "item_id",
    ):
        value = case.get(key)

        if not isinstance(value, str) or not value.strip():
            raise PolicyConformanceConfigurationError(
                f"cases[{index}].{key} must not be empty."
            )

    for key in (
        "concentration_ppm",
        "duration_seconds",
        "environment_volume_m3",
        "ventilation_ach",
    ):
        value = case.get(key)

        if value is None:
            continue

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) < 0.0
        ):
            raise PolicyConformanceConfigurationError(
                f"cases[{index}].{key} must be null or finite and nonnegative."
            )

    consent = case.get("user_consent")

    if consent is not None and not isinstance(
        consent,
        bool,
    ):
        raise PolicyConformanceConfigurationError(
            f"cases[{index}].user_consent must be boolean or null."
        )

    outcome = case.get("expected_outcome")

    if outcome not in ALLOWED_OUTCOMES:
        raise PolicyConformanceConfigurationError(
            f"cases[{index}].expected_outcome is invalid."
        )


def _mapping(
    parent: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = parent.get(key)

    if not isinstance(value, dict):
        raise PolicyConformanceConfigurationError(
            f"{key} must be a mapping."
        )

    return value


def _equal(
    mapping: dict[str, Any],
    key: str,
    expected: Any,
) -> None:
    value = mapping.get(key)

    if isinstance(expected, float):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or float(value) != expected
        ):
            raise PolicyConformanceConfigurationError(
                f"{key} must equal {expected!r}."
            )
        return

    if value != expected or (
        isinstance(expected, bool)
        and not isinstance(value, bool)
    ):
        raise PolicyConformanceConfigurationError(
            f"{key} must equal {expected!r}."
        )


def _true(
    mapping: dict[str, Any],
    key: str,
) -> None:
    if mapping.get(key) is not True:
        raise PolicyConformanceConfigurationError(
            f"{key} must be true."
        )
