"""Tests for the deterministic NOI policy gate."""

from copy import deepcopy

import pytest

from src.models import OutputRequest, PolicyOutcome
from src.safety.policy_gate import (
    DeterministicPolicyGate,
    PolicyConfigurationError,
    load_policy_rules,
    validate_policy_rules,
)


POLICY_PATH = "configs/policy_rules.yaml"
PROTOCOL_HASH = "b" * 64


@pytest.fixture
def configuration() -> dict:
    return deepcopy(load_policy_rules(POLICY_PATH))


@pytest.fixture
def gate(configuration: dict) -> DeterministicPolicyGate:
    return DeterministicPolicyGate(
        configuration=configuration,
        protocol_hash=PROTOCOL_HASH,
    )


def make_request(**changes) -> OutputRequest:
    values = {
        "request_id": "request-001",
        "item_id": "SIM-CARTRIDGE-001",
        "concentration_ppm": 0.5,
        "duration_seconds": 15.0,
        "environment_volume_m3": 30.0,
        "ventilation_ach": 2.0,
        "user_consent": True,
    }
    values.update(changes)
    return OutputRequest(**values)


def test_compliant_request_is_allowed(
    gate: DeterministicPolicyGate,
) -> None:
    decision = gate.evaluate(make_request())

    assert decision.outcome is PolicyOutcome.ALLOW
    assert decision.rule_ids == ("RULE-ALLOW-001",)


def test_missing_information_requires_information(
    gate: DeterministicPolicyGate,
) -> None:
    decision = gate.evaluate(
        make_request(ventilation_ach=None)
    )

    assert (
        decision.outcome
        is PolicyOutcome.REQUIRE_MISSING_INFORMATION
    )
    assert decision.rule_ids == ("RULE-MISSING-001",)


def test_missing_consent_is_not_treated_as_permission(
    gate: DeterministicPolicyGate,
) -> None:
    decision = gate.evaluate(
        make_request(user_consent=None)
    )

    assert (
        decision.outcome
        is PolicyOutcome.REQUIRE_MISSING_INFORMATION
    )


def test_explicitly_denied_consent_blocks_request(
    gate: DeterministicPolicyGate,
) -> None:
    decision = gate.evaluate(
        make_request(user_consent=False)
    )

    assert decision.outcome is PolicyOutcome.BLOCK
    assert decision.rule_ids == ("RULE-CONSENT-001",)


def test_unknown_item_is_blocked(
    gate: DeterministicPolicyGate,
) -> None:
    decision = gate.evaluate(
        make_request(item_id="UNKNOWN-ITEM")
    )

    assert decision.outcome is PolicyOutcome.BLOCK
    assert decision.rule_ids == ("RULE-ITEM-001",)


def test_disabled_item_is_blocked(
    gate: DeterministicPolicyGate,
) -> None:
    decision = gate.evaluate(
        make_request(item_id="SIM-CARTRIDGE-DISABLED")
    )

    assert decision.outcome is PolicyOutcome.BLOCK
    assert decision.rule_ids == ("RULE-ITEM-002",)


def test_excess_concentration_is_blocked(
    gate: DeterministicPolicyGate,
) -> None:
    decision = gate.evaluate(
        make_request(concentration_ppm=1.1)
    )

    assert decision.outcome is PolicyOutcome.BLOCK
    assert decision.rule_ids == ("RULE-LIMIT-001",)


def test_excess_duration_is_blocked(
    gate: DeterministicPolicyGate,
) -> None:
    decision = gate.evaluate(
        make_request(duration_seconds=31.0)
    )

    assert decision.outcome is PolicyOutcome.BLOCK
    assert decision.rule_ids == ("RULE-LIMIT-002",)


def test_inadequate_environment_is_blocked(
    gate: DeterministicPolicyGate,
) -> None:
    decision = gate.evaluate(
        make_request(environment_volume_m3=19.0)
    )

    assert decision.outcome is PolicyOutcome.BLOCK
    assert decision.rule_ids == ("RULE-ENV-001",)


def test_inadequate_ventilation_is_blocked(
    gate: DeterministicPolicyGate,
) -> None:
    decision = gate.evaluate(
        make_request(ventilation_ach=0.5)
    )

    assert decision.outcome is PolicyOutcome.BLOCK
    assert decision.rule_ids == ("RULE-ENV-002",)


def test_non_simulation_policy_is_rejected(
    configuration: dict,
) -> None:
    configuration["policy"]["simulation_only"] = False

    with pytest.raises(
        PolicyConfigurationError,
        match="explicitly simulation-only",
    ):
        validate_policy_rules(configuration)