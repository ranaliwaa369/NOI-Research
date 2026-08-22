"""Load, validate, and fingerprint the prespecified NOI research protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


class ProtocolValidationError(ValueError):
    """Raised when the research protocol is missing required information."""


REQUIRED_TOP_LEVEL_SECTIONS = {
    "project",
    "scope",
    "representation",
    "hypotheses",
    "dataset",
    "splits",
    "baselines",
    "evaluation",
    "statistics",
    "reproducibility",
}

REQUIRED_HYPOTHESES = {
    "H1_retrieval",
    "H2_corrective_updating",
    "H3_policy_conformance",
    "H4_robustness",
}


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML research protocol."""

    protocol_path = Path(path)

    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Research protocol was not found: {protocol_path}"
        )

    with protocol_path.open("r", encoding="utf-8") as file:
        protocol = yaml.safe_load(file)

    if not isinstance(protocol, dict):
        raise ProtocolValidationError(
            "The protocol must contain a top-level YAML mapping."
        )

    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Verify the required scope, hypotheses, thresholds, and safeguards."""

    missing_sections = REQUIRED_TOP_LEVEL_SECTIONS - protocol.keys()
    if missing_sections:
        raise ProtocolValidationError(
            f"Missing protocol sections: {sorted(missing_sections)}"
        )

    hypotheses = protocol["hypotheses"]
    if not isinstance(hypotheses, dict):
        raise ProtocolValidationError(
            "'hypotheses' must be a YAML mapping."
        )

    missing_hypotheses = REQUIRED_HYPOTHESES - hypotheses.keys()
    if missing_hypotheses:
        raise ProtocolValidationError(
            f"Missing hypotheses: {sorted(missing_hypotheses)}"
        )

    h1 = hypotheses["H1_retrieval"]
    h2 = hypotheses["H2_corrective_updating"]
    h3 = hypotheses["H3_policy_conformance"]

    delta = h1.get("minimum_absolute_mrr_improvement")
    epsilon = h2.get("maximum_old_memory_degradation")
    alpha = h3.get("maximum_false_block_rate")
    false_allow_target = h3.get("violation_false_allow_target")

    _validate_probability("delta", delta, allow_zero=False)
    _validate_probability("epsilon", epsilon, allow_zero=True)
    _validate_probability("alpha", alpha, allow_zero=True)

    if false_allow_target != 0:
        raise ProtocolValidationError(
            "The prespecified violation false-allow target must equal zero."
        )

    seeds = protocol["dataset"].get("independent_training_seeds")
    if not isinstance(seeds, list) or len(set(seeds)) < 5:
        raise ProtocolValidationError(
            "At least five unique independent training seeds are required."
        )

    baselines = protocol["baselines"]
    if not isinstance(baselines, list) or "full_NOI" not in baselines:
        raise ProtocolValidationError(
            "The baseline list must include 'full_NOI'."
        )

    if not protocol["reproducibility"].get(
        "lock_test_set_before_final_evaluation"
    ):
        raise ProtocolValidationError(
            "The final test set must be locked before evaluation."
        )


def _validate_probability(
    name: str,
    value: Any,
    *,
    allow_zero: bool,
) -> None:
    """Validate a prespecified threshold constrained to the unit interval."""

    if not isinstance(value, (int, float)):
        raise ProtocolValidationError(
            f"{name} must be numeric."
        )

    lower_bound = 0.0 if allow_zero else 0.0
    valid_lower = value >= lower_bound if allow_zero else value > lower_bound

    if not valid_lower or value > 1.0:
        raise ProtocolValidationError(
            f"{name} must be within the prespecified unit interval."
        )


def protocol_sha256(path: str | Path) -> str:
    """Return a SHA-256 fingerprint for protocol version tracking."""

    protocol_path = Path(path)
    return hashlib.sha256(protocol_path.read_bytes()).hexdigest()