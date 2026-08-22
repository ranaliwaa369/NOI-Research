"""Deterministic, fail-closed policy gate for simulated NOI outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.models import OutputRequest, PolicyDecision, PolicyOutcome


class PolicyConfigurationError(ValueError):
    """Raised when the simulated policy configuration is invalid."""


def load_policy_rules(path: str | Path) -> dict[str, Any]:
    """Load and validate the simulated policy configuration."""

    policy_path = Path(path)

    if not policy_path.is_file():
        raise FileNotFoundError(
            f"Policy configuration was not found: {policy_path}"
        )

    with policy_path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    if not isinstance(configuration, dict):
        raise PolicyConfigurationError(
            "Policy configuration must be a YAML mapping."
        )

    validate_policy_rules(configuration)
    return configuration


def validate_policy_rules(configuration: dict[str, Any]) -> None:
    """Ensure the policy is explicitly simulation-only and fail-closed."""

    required_sections = {
        "policy",
        "disclaimer",
        "required_request_fields",
        "rules",
        "simulated_items",
    }

    missing = required_sections - configuration.keys()
    if missing:
        raise PolicyConfigurationError(
            f"Missing policy sections: {sorted(missing)}"
        )

    policy = configuration["policy"]

    if policy.get("simulation_only") is not True:
        raise PolicyConfigurationError(
            "The current policy must be explicitly simulation-only."
        )

    if policy.get("default_action") != "BLOCK":
        raise PolicyConfigurationError(
            "The default policy action must be BLOCK."
        )

    required_rule_names = {
        "missing_information",
        "consent_required",
        "unknown_item",
        "disabled_item",
        "concentration_limit",
        "duration_limit",
        "minimum_environment_volume",
        "minimum_ventilation",
        "allow",
    }

    missing_rules = required_rule_names - configuration["rules"].keys()
    if missing_rules:
        raise PolicyConfigurationError(
            f"Missing policy rules: {sorted(missing_rules)}"
        )

    if not configuration["simulated_items"]:
        raise PolicyConfigurationError(
            "At least one simulated item must be configured."
        )


class DeterministicPolicyGate:
    """Evaluate simulated output requests using prespecified rules."""

    def __init__(
        self,
        configuration: dict[str, Any],
        protocol_hash: str,
    ) -> None:
        validate_policy_rules(configuration)

        if len(protocol_hash) != 64:
            raise ValueError(
                "protocol_hash must be a 64-character SHA-256 value."
            )

        self.configuration = configuration
        self.protocol_hash = protocol_hash

    def evaluate(self, request: OutputRequest) -> PolicyDecision:
        """Return ALLOW, BLOCK, or REQUIRE_MISSING_INFORMATION."""

        missing_fields = self._missing_fields(request)

        if missing_fields:
            return self._decision(
                request=request,
                outcome=PolicyOutcome.REQUIRE_MISSING_INFORMATION,
                rule_name="missing_information",
                explanation=(
                    "Required simulated request information is missing: "
                    + ", ".join(missing_fields)
                ),
            )

        if request.user_consent is not True:
            return self._decision(
                request=request,
                outcome=PolicyOutcome.BLOCK,
                rule_name="consent_required",
                explanation=(
                    "The simulated request was blocked because current "
                    "affirmative user consent was not recorded."
                ),
            )

        item = self.configuration["simulated_items"].get(request.item_id)

        if item is None:
            return self._decision(
                request=request,
                outcome=PolicyOutcome.BLOCK,
                rule_name="unknown_item",
                explanation=(
                    "The requested item is not present in the locked "
                    "simulated item inventory."
                ),
            )

        if item.get("enabled") is not True:
            return self._decision(
                request=request,
                outcome=PolicyOutcome.BLOCK,
                rule_name="disabled_item",
                explanation="The simulated item is disabled by policy.",
            )

        if (
            request.concentration_ppm
            > item["maximum_concentration_ppm"]
        ):
            return self._decision(
                request=request,
                outcome=PolicyOutcome.BLOCK,
                rule_name="concentration_limit",
                explanation=(
                    "The request exceeds the prespecified simulated "
                    "concentration threshold."
                ),
            )

        if request.duration_seconds > item["maximum_duration_seconds"]:
            return self._decision(
                request=request,
                outcome=PolicyOutcome.BLOCK,
                rule_name="duration_limit",
                explanation=(
                    "The request exceeds the prespecified simulated "
                    "duration threshold."
                ),
            )

        if (
            request.environment_volume_m3
            < item["minimum_environment_volume_m3"]
        ):
            return self._decision(
                request=request,
                outcome=PolicyOutcome.BLOCK,
                rule_name="minimum_environment_volume",
                explanation=(
                    "The simulated environment volume is below the "
                    "prespecified policy threshold."
                ),
            )

        if request.ventilation_ach < item["minimum_ventilation_ach"]:
            return self._decision(
                request=request,
                outcome=PolicyOutcome.BLOCK,
                rule_name="minimum_ventilation",
                explanation=(
                    "The simulated ventilation value is below the "
                    "prespecified policy threshold."
                ),
            )

        return self._decision(
            request=request,
            outcome=PolicyOutcome.ALLOW,
            rule_name="allow",
            explanation=(
                "The request conforms to all prespecified computational "
                "policy rules for the simulated test environment."
            ),
        )

    def _missing_fields(self, request: OutputRequest) -> list[str]:
        """Return required request fields whose value is unknown."""

        return [
            field_name
            for field_name in self.configuration[
                "required_request_fields"
            ]
            if getattr(request, field_name, None) is None
        ]

    def _decision(
        self,
        *,
        request: OutputRequest,
        outcome: PolicyOutcome,
        rule_name: str,
        explanation: str,
    ) -> PolicyDecision:
        """Construct an auditable policy decision."""

        rule_id = self.configuration["rules"][rule_name]["rule_id"]

        return PolicyDecision(
            request_id=request.request_id,
            outcome=outcome,
            rule_ids=(rule_id,),
            explanation=explanation,
            protocol_hash=self.protocol_hash,
        )