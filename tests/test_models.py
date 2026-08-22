"""Tests for the core NOI data models."""

from datetime import datetime, timezone
from math import nan

import pytest

from src.models import (
    MultimodalContext,
    OdorLibraryItem,
    OutputRequest,
    PolicyDecision,
    PolicyOutcome,
    RetrievalCandidate,
)


PROTOCOL_HASH = "a" * 64


def test_valid_multimodal_context() -> None:
    context = MultimodalContext(
        event_id="event-001",
        timestamp_utc=datetime.now(timezone.utc),
        text_vector=(0.1, 0.2, 0.3),
        image_vector=(0.4, 0.5, 0.6),
    )

    assert context.event_id == "event-001"


def test_context_requires_at_least_one_modality() -> None:
    with pytest.raises(
        ValueError,
        match="At least one contextual modality",
    ):
        MultimodalContext(
            event_id="event-002",
            timestamp_utc=datetime.now(timezone.utc),
        )


def test_context_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        MultimodalContext(
            event_id="event-003",
            timestamp_utc=datetime.now(),
            text_vector=(0.1, 0.2),
        )


def test_odor_vector_rejects_nonfinite_values() -> None:
    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        OdorLibraryItem(
            item_id="odor-001",
            odor_vector=(0.1, nan),
            descriptors=("floral",),
        )


def test_retrieval_rank_must_start_at_one() -> None:
    with pytest.raises(
        ValueError,
        match="rank must be at least 1",
    ):
        RetrievalCandidate(
            item_id="odor-002",
            score=0.90,
            rank=0,
        )


def test_output_request_rejects_negative_exposure() -> None:
    with pytest.raises(
        ValueError,
        match="duration_seconds must be finite and nonnegative",
    ):
        OutputRequest(
            request_id="request-001",
            item_id="odor-003",
            concentration_ppm=1.0,
            duration_seconds=-5.0,
            environment_volume_m3=30.0,
            ventilation_ach=2.0,
            user_consent=True,
        )


def test_policy_decision_requires_rule_identifier() -> None:
    with pytest.raises(
        ValueError,
        match="at least one rule",
    ):
        PolicyDecision(
            request_id="request-002",
            outcome=PolicyOutcome.BLOCK,
            rule_ids=(),
            explanation="Blocked by the policy gate.",
            protocol_hash=PROTOCOL_HASH,
        )


def test_valid_policy_decision_is_auditable() -> None:
    decision = PolicyDecision(
        request_id="request-003",
        outcome=PolicyOutcome.ALLOW,
        rule_ids=("RULE-CONSENT-001",),
        explanation="All prespecified simulated requirements are present.",
        protocol_hash=PROTOCOL_HASH,
    )

    assert decision.outcome is PolicyOutcome.ALLOW
    assert decision.protocol_hash == PROTOCOL_HASH