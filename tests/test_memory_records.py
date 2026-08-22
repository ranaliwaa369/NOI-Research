"""Tests for validated NOI associative-memory records."""

from datetime import datetime, timedelta, timezone

import pytest

from src.memory.records import (
    AssociativeMemoryRecord,
    MemoryRetrievalCandidate,
)


NOW = datetime.now(timezone.utc)


def make_record(**changes) -> AssociativeMemoryRecord:
    values = {
        "memory_id": "memory-001",
        "context_vector": (1.0, 0.0, 0.5),
        "odor_item_id": "odor-001",
        "created_at_utc": NOW,
        "updated_at_utc": NOW,
        "strength": 1.0,
        "correction_count": 0,
        "active": True,
    }
    values.update(changes)
    return AssociativeMemoryRecord(**values)


def test_valid_memory_record() -> None:
    record = make_record()

    assert record.memory_id == "memory-001"
    assert record.active is True
    assert record.correction_count == 0


def test_memory_id_is_required() -> None:
    with pytest.raises(
        ValueError,
        match="memory_id must not be empty",
    ):
        make_record(memory_id=" ")


def test_context_vector_must_be_finite() -> None:
    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        make_record(context_vector=(1.0, float("nan")))


def test_timestamps_must_be_timezone_aware() -> None:
    naive_time = datetime.now()

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        make_record(
            created_at_utc=naive_time,
            updated_at_utc=naive_time,
        )


def test_update_cannot_precede_creation() -> None:
    with pytest.raises(
        ValueError,
        match="cannot precede",
    ):
        make_record(
            updated_at_utc=NOW - timedelta(seconds=1),
        )


def test_strength_must_be_within_unit_interval() -> None:
    with pytest.raises(
        ValueError,
        match=r"within \(0, 1\]",
    ):
        make_record(strength=1.1)


def test_correction_count_must_be_nonnegative_integer() -> None:
    with pytest.raises(
        ValueError,
        match="nonnegative integer",
    ):
        make_record(correction_count=-1)


def test_valid_memory_candidate_is_auditable() -> None:
    candidate = MemoryRetrievalCandidate(
        memory_id="memory-001",
        odor_item_id="odor-001",
        score=0.72,
        contextual_similarity=0.90,
        temporal_weight=0.80,
        memory_strength=1.0,
        rank=1,
    )

    assert candidate.score == pytest.approx(0.72)
    assert candidate.rank == 1


def test_candidate_rejects_invalid_temporal_weight() -> None:
    with pytest.raises(
        ValueError,
        match="temporal_weight",
    ):
        MemoryRetrievalCandidate(
            memory_id="memory-001",
            odor_item_id="odor-001",
            score=0.0,
            contextual_similarity=0.0,
            temporal_weight=0.0,
            memory_strength=1.0,
            rank=1,
        )