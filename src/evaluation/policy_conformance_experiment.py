"""Deterministic policy-conformance evaluation for synthetic NOI."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from src.models import (
    OutputRequest,
    PolicyOutcome,
)
from src.safety.policy_gate import (
    DeterministicPolicyGate,
)


class PolicyConformanceExperimentError(ValueError):
    """Raised when policy conformance cannot be evaluated."""


@dataclass(frozen=True)
class PolicyCaseResult:
    """Auditable decision for one preregistered request."""

    request_id: str
    description: str
    expected_outcome: PolicyOutcome
    predicted_outcome: PolicyOutcome
    rule_ids: tuple[str, ...]
    explanation: str
    protocol_hash: str
    exact_match: bool
    false_allow: bool
    false_block: bool

    def __post_init__(self) -> None:
        if not self.request_id:
            raise PolicyConformanceExperimentError(
                "request_id must not be empty."
            )

        if not self.description:
            raise PolicyConformanceExperimentError(
                "description must not be empty."
            )

        if not isinstance(
            self.expected_outcome,
            PolicyOutcome,
        ):
            raise PolicyConformanceExperimentError(
                "expected_outcome must be a PolicyOutcome."
            )

        if not isinstance(
            self.predicted_outcome,
            PolicyOutcome,
        ):
            raise PolicyConformanceExperimentError(
                "predicted_outcome must be a PolicyOutcome."
            )

        if not self.rule_ids:
            raise PolicyConformanceExperimentError(
                "At least one rule ID is required."
            )

        if not self.protocol_hash:
            raise PolicyConformanceExperimentError(
                "protocol_hash must not be empty."
            )

        expected_match = (
            self.expected_outcome
            is self.predicted_outcome
        )

        if self.exact_match is not expected_match:
            raise PolicyConformanceExperimentError(
                "exact_match is inconsistent."
            )


@dataclass(frozen=True)
class PolicyConformanceExperiment:
    """Complete locked policy-conformance evaluation."""

    case_results: tuple[PolicyCaseResult, ...]
    total_case_count: int
    expected_allow_count: int
    expected_block_count: int
    expected_missing_information_count: int
    exact_match_count: int
    false_allow_count: int
    false_allow_rate: float
    false_block_count: int
    false_block_rate: float
    exact_conformance_rate: float
    policy_coverage: float
    false_allow_target_passed: bool
    false_block_target_passed: bool
    exact_conformance_target_passed: bool
    policy_coverage_target_passed: bool
    h3_success_rule_passed: bool
    physical_emission_performed: bool
    protocol_hash: str

    def __post_init__(self) -> None:
        if self.total_case_count != 26:
            raise PolicyConformanceExperimentError(
                "The locked suite must contain 26 cases."
            )

        if len(self.case_results) != 26:
            raise PolicyConformanceExperimentError(
                "case_results must contain 26 decisions."
            )

        if (
            self.expected_allow_count != 4
            or self.expected_block_count != 15
            or self.expected_missing_information_count != 7
        ):
            raise PolicyConformanceExperimentError(
                "Expected outcome counts changed."
            )

        for label, value in (
            ("false_allow_rate", self.false_allow_rate),
            ("false_block_rate", self.false_block_rate),
            (
                "exact_conformance_rate",
                self.exact_conformance_rate,
            ),
            ("policy_coverage", self.policy_coverage),
        ):
            if (
                not isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise PolicyConformanceExperimentError(
                    f"{label} must be in [0, 1]."
                )

        if self.physical_emission_performed is not False:
            raise PolicyConformanceExperimentError(
                "Physical emission is prohibited."
            )


def run_policy_conformance_experiment(
    evaluation_configuration: Mapping[str, Any],
    *,
    policy_configuration: Mapping[str, Any],
    protocol_hash: str,
) -> PolicyConformanceExperiment:
    """Evaluate all preregistered requests in their locked order."""

    if not isinstance(
        evaluation_configuration,
        Mapping,
    ):
        raise PolicyConformanceExperimentError(
            "evaluation_configuration must be a mapping."
        )

    if not isinstance(policy_configuration, Mapping):
        raise PolicyConformanceExperimentError(
            "policy_configuration must be a mapping."
        )

    if not protocol_hash:
        raise PolicyConformanceExperimentError(
            "protocol_hash must not be empty."
        )

    cases = evaluation_configuration.get("cases")
    metrics = evaluation_configuration.get("metrics")
    policy_definition = evaluation_configuration.get(
        "policy"
    )

    if not isinstance(cases, list) or len(cases) != 26:
        raise PolicyConformanceExperimentError(
            "Exactly 26 locked cases are required."
        )

    if not isinstance(metrics, Mapping):
        raise PolicyConformanceExperimentError(
            "metrics configuration is required."
        )

    if not isinstance(policy_definition, Mapping):
        raise PolicyConformanceExperimentError(
            "policy configuration is required."
        )

    if (
        policy_definition.get(
            "physical_emission_performed"
        )
        is not False
    ):
        raise PolicyConformanceExperimentError(
            "Physical emission must remain false."
        )

    gate = DeterministicPolicyGate(
        dict(policy_configuration),
        protocol_hash,
    )

    results = []

    for case in cases:
        request = OutputRequest(
            request_id=case["request_id"],
            item_id=case["item_id"],
            concentration_ppm=case[
                "concentration_ppm"
            ],
            duration_seconds=case[
                "duration_seconds"
            ],
            environment_volume_m3=case[
                "environment_volume_m3"
            ],
            ventilation_ach=case[
                "ventilation_ach"
            ],
            user_consent=case["user_consent"],
        )

        expected = PolicyOutcome(
            case["expected_outcome"]
        )
        decision = gate.evaluate(request)

        exact_match = (
            decision.outcome is expected
        )
        false_allow = (
            decision.outcome is PolicyOutcome.ALLOW
            and expected is not PolicyOutcome.ALLOW
        )
        false_block = (
            decision.outcome is PolicyOutcome.BLOCK
            and expected is PolicyOutcome.ALLOW
        )

        results.append(
            PolicyCaseResult(
                request_id=case["request_id"],
                description=case["description"],
                expected_outcome=expected,
                predicted_outcome=decision.outcome,
                rule_ids=decision.rule_ids,
                explanation=decision.explanation,
                protocol_hash=decision.protocol_hash,
                exact_match=exact_match,
                false_allow=false_allow,
                false_block=false_block,
            )
        )

    result_tuple = tuple(results)
    total = len(result_tuple)

    false_allow_count = sum(
        result.false_allow
        for result in result_tuple
    )
    false_block_count = sum(
        result.false_block
        for result in result_tuple
    )
    exact_match_count = sum(
        result.exact_match
        for result in result_tuple
    )

    valid_outcomes = {
        PolicyOutcome.ALLOW,
        PolicyOutcome.BLOCK,
        PolicyOutcome.REQUIRE_MISSING_INFORMATION,
    }
    covered_count = sum(
        result.predicted_outcome in valid_outcomes
        for result in result_tuple
    )

    false_allow_rate = false_allow_count / total
    expected_allow_count = 4
    false_block_rate = (
        false_block_count / expected_allow_count
    )
    exact_rate = exact_match_count / total
    coverage = covered_count / total

    false_allow_passed = (
        false_allow_count
        == metrics["violation_false_allow_target"]
    )
    false_block_passed = (
        false_block_rate
        <= metrics["maximum_false_block_rate"]
    )
    exact_passed = (
        exact_rate
        >= metrics["exact_conformance_target"]
    )
    coverage_passed = (
        coverage
        >= metrics["policy_coverage_target"]
    )

    return PolicyConformanceExperiment(
        case_results=result_tuple,
        total_case_count=total,
        expected_allow_count=4,
        expected_block_count=15,
        expected_missing_information_count=7,
        exact_match_count=exact_match_count,
        false_allow_count=false_allow_count,
        false_allow_rate=false_allow_rate,
        false_block_count=false_block_count,
        false_block_rate=false_block_rate,
        exact_conformance_rate=exact_rate,
        policy_coverage=coverage,
        false_allow_target_passed=(
            false_allow_passed
        ),
        false_block_target_passed=(
            false_block_passed
        ),
        exact_conformance_target_passed=(
            exact_passed
        ),
        policy_coverage_target_passed=(
            coverage_passed
        ),
        h3_success_rule_passed=(
            false_allow_passed
            and false_block_passed
            and exact_passed
            and coverage_passed
        ),
        physical_emission_performed=False,
        protocol_hash=protocol_hash,
    )
