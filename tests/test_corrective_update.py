"""Tests for auditable corrective associative-memory updating."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.correction.corrective_update import (
    CorrectiveMemoryUpdater,
    CorrectiveUpdateError,
)
from src.memory.records import AssociativeMemoryRecord
from src.memory.temporal_memory import (
    AssociativeMemoryError,
    TemporalAssociativeMemory,
)


NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
PROTOCOL_HASH = "c" * 64


@pytest.fixture
def memory() -> TemporalAssociativeMemory:
    store = TemporalAssociativeMemory(
        dimension=2,
        decay_rate_per_day=0.1,
    )
    store.add(
        AssociativeMemoryRecord(
            memory_id="memory-001",
            context_vector=(1.0, 0.0),
            odor_item_id="odor-wrong",
            created_at_utc=NOW,
            updated_at_utc=NOW,
        )
    )
    return store


@pytest.fixture
def updater(
    memory: TemporalAssociativeMemory,
) -> CorrectiveMemoryUpdater:
    return CorrectiveMemoryUpdater(
        memory=memory,
        protocol_hash=PROTOCOL_HASH,
    )


def test_correction_updates_target_and_count(
    memory: TemporalAssociativeMemory,
    updater: CorrectiveMemoryUpdater,
) -> None:
    corrected_at = NOW + timedelta(hours=1)

    audit = updater.apply(
        correction_id="correction-001",
        memory_id="memory-001",
        corrected_at_utc=corrected_at,
        reason="The stored target was incorrect.",
        corrected_odor_item_id="odor-correct",
    )

    updated = memory.get("memory-001")

    assert updated.odor_item_id == "odor-correct"
    assert updated.correction_count == 1
    assert updated.created_at_utc == NOW
    assert updated.updated_at_utc == corrected_at
    assert audit.previous_odor_item_id == "odor-wrong"
    assert audit.corrected_odor_item_id == "odor-correct"


def test_context_correction_changes_vector_hash(
    updater: CorrectiveMemoryUpdater,
) -> None:
    audit = updater.apply(
        correction_id="correction-002",
        memory_id="memory-001",
        corrected_at_utc=NOW + timedelta(hours=1),
        reason="The contextual representation was corrected.",
        corrected_context_vector=(0.0, 1.0),
    )

    assert (
        audit.previous_context_hash
        != audit.corrected_context_hash
    )


def test_duplicate_correction_id_is_rejected(
    updater: CorrectiveMemoryUpdater,
) -> None:
    updater.apply(
        correction_id="correction-003",
        memory_id="memory-001",
        corrected_at_utc=NOW + timedelta(hours=1),
        reason="First correction.",
    )

    with pytest.raises(
        CorrectiveUpdateError,
        match="Duplicate correction_id",
    ):
        updater.apply(
            correction_id="correction-003",
            memory_id="memory-001",
            corrected_at_utc=NOW + timedelta(hours=2),
            reason="Duplicate correction.",
        )


def test_correction_cannot_move_time_backward(
    updater: CorrectiveMemoryUpdater,
) -> None:
    with pytest.raises(
        CorrectiveUpdateError,
        match="cannot precede",
    ):
        updater.apply(
            correction_id="correction-004",
            memory_id="memory-001",
            corrected_at_utc=NOW - timedelta(seconds=1),
            reason="Invalid earlier correction.",
        )


def test_unknown_memory_is_rejected(
    updater: CorrectiveMemoryUpdater,
) -> None:
    with pytest.raises(
        AssociativeMemoryError,
        match="Unknown memory_id",
    ):
        updater.apply(
            correction_id="correction-005",
            memory_id="unknown-memory",
            corrected_at_utc=NOW + timedelta(hours=1),
            reason="Unknown record.",
        )


def test_failed_correction_does_not_enter_audit_log(
    updater: CorrectiveMemoryUpdater,
) -> None:
    with pytest.raises(
        ValueError,
        match="strength",
    ):
        updater.apply(
            correction_id="correction-006",
            memory_id="memory-001",
            corrected_at_utc=NOW + timedelta(hours=1),
            reason="Invalid strength.",
            corrected_strength=1.5,
        )

    assert updater.audit_log == ()


def test_replace_cannot_change_creation_time(
    memory: TemporalAssociativeMemory,
) -> None:
    original = memory.get("memory-001")
    invalid = replace(
        original,
        created_at_utc=NOW + timedelta(seconds=1),
        updated_at_utc=NOW + timedelta(hours=1),
    )

    with pytest.raises(
        AssociativeMemoryError,
        match="cannot change created_at_utc",
    ):
        memory.replace(invalid)


def test_replace_cannot_reduce_correction_count(
    memory: TemporalAssociativeMemory,
) -> None:
    original = memory.get("memory-001")
    corrected = replace(
        original,
        updated_at_utc=NOW + timedelta(hours=1),
        correction_count=1,
    )
    memory.replace(corrected)

    invalid = replace(
        corrected,
        updated_at_utc=NOW + timedelta(hours=2),
        correction_count=0,
    )

    with pytest.raises(
        AssociativeMemoryError,
        match="cannot reduce correction_count",
    ):
        memory.replace(invalid)


def test_corrected_association_is_retrievable(
    memory: TemporalAssociativeMemory,
    updater: CorrectiveMemoryUpdater,
) -> None:
    updater.apply(
        correction_id="correction-007",
        memory_id="memory-001",
        corrected_at_utc=NOW + timedelta(hours=1),
        reason="Correct the target association.",
        corrected_odor_item_id="odor-correct",
    )

    results = memory.retrieve(
        (1.0, 0.0),
        as_of_utc=NOW + timedelta(hours=2),
        top_k=1,
    )

    assert results[0].odor_item_id == "odor-correct"


def test_audit_log_is_exposed_as_immutable_tuple(
    updater: CorrectiveMemoryUpdater,
) -> None:
    updater.apply(
        correction_id="correction-008",
        memory_id="memory-001",
        corrected_at_utc=NOW + timedelta(hours=1),
        reason="Audited correction.",
    )

    assert isinstance(updater.audit_log, tuple)
    assert len(updater.audit_log) == 1
    assert updater.audit_log[0].protocol_hash == PROTOCOL_HASH