"""Tests for the NOI prespecified research protocol."""

from copy import deepcopy
from pathlib import Path

import pytest

from src.protocol import (
    ProtocolValidationError,
    load_protocol,
    protocol_sha256,
    validate_protocol,
)


PROTOCOL_PATH = Path("configs/research_protocol.yaml")


@pytest.fixture
def valid_protocol() -> dict:
    """Return a fresh copy of the valid research protocol."""

    return deepcopy(load_protocol(PROTOCOL_PATH))


def test_valid_protocol_loads() -> None:
    """The prespecified protocol should load successfully."""

    protocol = load_protocol(PROTOCOL_PATH)

    assert protocol["project"]["name"] == "Neuro-Olfactive Interface"
    assert "H1_retrieval" in protocol["hypotheses"]
    assert "full_NOI" in protocol["baselines"]


def test_protocol_hash_is_deterministic() -> None:
    """The same protocol file should always produce the same fingerprint."""

    first_hash = protocol_sha256(PROTOCOL_PATH)
    second_hash = protocol_sha256(PROTOCOL_PATH)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_missing_section_is_rejected(valid_protocol: dict) -> None:
    """A protocol missing a required section must be rejected."""

    del valid_protocol["statistics"]

    with pytest.raises(
        ProtocolValidationError,
        match="Missing protocol sections",
    ):
        validate_protocol(valid_protocol)


def test_nonzero_false_allow_target_is_rejected(
    valid_protocol: dict,
) -> None:
    """The locked policy suite must require zero false allows."""

    valid_protocol["hypotheses"]["H3_policy_conformance"][
        "violation_false_allow_target"
    ] = 1

    with pytest.raises(
        ProtocolValidationError,
        match="false-allow target must equal zero",
    ):
        validate_protocol(valid_protocol)


def test_insufficient_independent_seeds_are_rejected(
    valid_protocol: dict,
) -> None:
    """At least five unique independent seeds must be declared."""

    valid_protocol["dataset"]["independent_training_seeds"] = [
        11,
        22,
        33,
    ]

    with pytest.raises(
        ProtocolValidationError,
        match="At least five unique independent training seeds",
    ):
        validate_protocol(valid_protocol)


def test_unlocked_final_test_set_is_rejected(
    valid_protocol: dict,
) -> None:
    """Final evaluation cannot proceed with an unlocked test set."""

    valid_protocol["reproducibility"][
        "lock_test_set_before_final_evaluation"
    ] = False

    with pytest.raises(
        ProtocolValidationError,
        match="final test set must be locked",
    ):
        validate_protocol(valid_protocol)