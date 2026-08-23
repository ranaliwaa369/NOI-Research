"""Tests for deterministic policy-conformance evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.evaluation.policy_conformance_config import (
    load_policy_conformance_configuration,
)
from src.evaluation.policy_conformance_experiment import (
    PolicyConformanceExperimentError,
    run_policy_conformance_experiment,
)
from src.models import PolicyOutcome
from src.safety.policy_gate import load_policy_rules


PROTOCOL_HASH = (
    "e885f537b3209d7052d1517efc5be753"
    "24df39689cb4e54b38a241c95f22e512"
)


@pytest.fixture(scope="module")
def configuration():
    return load_policy_conformance_configuration(
        "configs/policy_conformance_evaluation_v0.1.yaml",
        "configs/policy_conformance_evaluation_v0.1.sha256",
    )


@pytest.fixture(scope="module")
def policy_configuration():
    return load_policy_rules(
        "configs/policy_rules.yaml"
    )


@pytest.fixture(scope="module")
def experiment(
    configuration,
    policy_configuration,
):
    return run_policy_conformance_experiment(
        configuration,
        policy_configuration=policy_configuration,
        protocol_hash=PROTOCOL_HASH,
    )


def test_locked_case_count(experiment) -> None:
    assert experiment.total_case_count == 26
    assert len(experiment.case_results) == 26


def test_expected_outcome_counts(experiment) -> None:
    assert experiment.expected_allow_count == 4
    assert experiment.expected_block_count == 15
    assert (
        experiment.expected_missing_information_count
        == 7
    )


def test_all_cases_match(experiment) -> None:
    assert experiment.exact_match_count == 26
    assert all(
        result.exact_match
        for result in experiment.case_results
    )


def test_no_false_allows(experiment) -> None:
    assert experiment.false_allow_count == 0
    assert experiment.false_allow_rate == 0.0
    assert experiment.false_allow_target_passed is True


def test_no_false_blocks(experiment) -> None:
    assert experiment.false_block_count == 0
    assert experiment.false_block_rate == 0.0
    assert experiment.false_block_target_passed is True


def test_exact_conformance_and_coverage(experiment) -> None:
    assert experiment.exact_conformance_rate == 1.0
    assert experiment.policy_coverage == 1.0
    assert (
        experiment.exact_conformance_target_passed
        is True
    )
    assert (
        experiment.policy_coverage_target_passed
        is True
    )


def test_h3_success_rule_passes(experiment) -> None:
    assert experiment.h3_success_rule_passed is True


def test_no_physical_emission(experiment) -> None:
    assert experiment.physical_emission_performed is False


def test_request_ids_are_unique_and_ordered(
    experiment,
    configuration,
) -> None:
    observed = tuple(
        result.request_id
        for result in experiment.case_results
    )
    expected = tuple(
        case["request_id"]
        for case in configuration["cases"]
    )

    assert observed == expected
    assert len(set(observed)) == 26


def test_decisions_retain_audit_information(
    experiment,
) -> None:
    for result in experiment.case_results:
        assert result.rule_ids
        assert result.explanation
        assert result.protocol_hash == PROTOCOL_HASH


@pytest.mark.parametrize(
    "prefix",
    (
        "ALLOW-",
        "BLOCK-",
        "MISSING-",
        "PRECEDENCE-",
    ),
)
def test_registered_case_category_is_present(
    experiment,
    prefix: str,
) -> None:
    assert any(
        result.request_id.startswith(prefix)
        for result in experiment.case_results
    )


def test_allow_cases_are_allowed(experiment) -> None:
    allow_cases = [
        result
        for result in experiment.case_results
        if result.request_id.startswith("ALLOW-")
    ]

    assert len(allow_cases) == 4
    assert all(
        result.expected_outcome is PolicyOutcome.ALLOW
        and result.predicted_outcome is PolicyOutcome.ALLOW
        for result in allow_cases
    )


def test_missing_cases_request_information(
    experiment,
) -> None:
    missing_cases = [
        result
        for result in experiment.case_results
        if result.request_id.startswith("MISSING-")
    ]

    assert len(missing_cases) == 6
    assert all(
        result.predicted_outcome
        is PolicyOutcome.REQUIRE_MISSING_INFORMATION
        for result in missing_cases
    )


def test_experiment_is_immutable(experiment) -> None:
    with pytest.raises(FrozenInstanceError):
        experiment.false_allow_count = 1  # type: ignore[misc]


def test_case_result_is_immutable(experiment) -> None:
    with pytest.raises(FrozenInstanceError):
        experiment.case_results[0].exact_match = False  # type: ignore[misc]


def test_invalid_evaluation_type_is_rejected(
    policy_configuration,
) -> None:
    with pytest.raises(
        PolicyConformanceExperimentError,
        match="mapping",
    ):
        run_policy_conformance_experiment(
            "invalid",  # type: ignore[arg-type]
            policy_configuration=policy_configuration,
            protocol_hash=PROTOCOL_HASH,
        )


def test_invalid_policy_type_is_rejected(
    configuration,
) -> None:
    with pytest.raises(
        PolicyConformanceExperimentError,
        match="mapping",
    ):
        run_policy_conformance_experiment(
            configuration,
            policy_configuration="invalid",  # type: ignore[arg-type]
            protocol_hash=PROTOCOL_HASH,
        )


def test_empty_protocol_hash_is_rejected(
    configuration,
    policy_configuration,
) -> None:
    with pytest.raises(
        PolicyConformanceExperimentError,
        match="protocol_hash",
    ):
        run_policy_conformance_experiment(
            configuration,
            policy_configuration=policy_configuration,
            protocol_hash="",
        )


def test_experiment_is_deterministic(
    configuration,
    policy_configuration,
    experiment,
) -> None:
    repeated = run_policy_conformance_experiment(
        configuration,
        policy_configuration=policy_configuration,
        protocol_hash=PROTOCOL_HASH,
    )

    assert repeated == experiment
