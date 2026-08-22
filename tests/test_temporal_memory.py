"""Tests for the temporally structured NOI associative memory."""

from datetime import datetime, timedelta, timezone

import pytest

from src.memory.records import AssociativeMemoryRecord
from src.memory.temporal_memory import (
    AssociativeMemoryError,
    TemporalAssociativeMemory,
)


NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def make_record(
    *,
    memory_id: str,
    odor_item_id: str,
    context_vector: tuple[float, ...] = (1.0, 0.0),
    days_old: float = 0.0,
    strength: float = 1.0,
    active: bool = True,
) -> AssociativeMemoryRecord:
    timestamp = NOW - timedelta(days=days_old)

    return AssociativeMemoryRecord(
        memory_id=memory_id,
        context_vector=context_vector,
        odor_item_id=odor_item_id,
        created_at_utc=timestamp,
        updated_at_utc=timestamp,
        strength=strength,
        active=active,
    )


def make_memory(
    *,
    decay_rate: float = 0.1,
) -> TemporalAssociativeMemory:
    return TemporalAssociativeMemory(
        dimension=2,
        decay_rate_per_day=decay_rate,
    )


def test_memory_reports_properties() -> None:
    memory = make_memory(decay_rate=0.2)
    memory.add(
        make_record(
            memory_id="memory-001",
            odor_item_id="odor-001",
        )
    )

    assert memory.dimension == 2
    assert memory.decay_rate_per_day == pytest.approx(0.2)
    assert memory.size == 1
    assert memory.active_size == 1


def test_duplicate_memory_id_is_rejected() -> None:
    memory = make_memory()
    record = make_record(
        memory_id="memory-001",
        odor_item_id="odor-001",
    )
    memory.add(record)

    with pytest.raises(
        AssociativeMemoryError,
        match="Duplicate memory_id",
    ):
        memory.add(record)


def test_mismatched_record_dimension_is_rejected() -> None:
    memory = make_memory()

    with pytest.raises(
        AssociativeMemoryError,
        match="record dimension",
    ):
        memory.add(
            make_record(
                memory_id="memory-001",
                odor_item_id="odor-001",
                context_vector=(1.0, 0.0, 0.0),
            )
        )


def test_most_contextually_similar_memory_ranks_first() -> None:
    memory = make_memory(decay_rate=0.0)
    memory.add(
        make_record(
            memory_id="memory-a",
            odor_item_id="odor-a",
            context_vector=(1.0, 0.0),
        )
    )
    memory.add(
        make_record(
            memory_id="memory-b",
            odor_item_id="odor-b",
            context_vector=(0.0, 1.0),
        )
    )

    results = memory.retrieve(
        (0.9, 0.1),
        as_of_utc=NOW,
        top_k=2,
    )

    assert results[0].odor_item_id == "odor-a"
    assert results[0].rank == 1


def test_temporal_decay_favors_newer_equal_memory() -> None:
    memory = make_memory(decay_rate=0.2)
    memory.add(
        make_record(
            memory_id="memory-old",
            odor_item_id="odor-a",
            days_old=10.0,
        )
    )
    memory.add(
        make_record(
            memory_id="memory-new",
            odor_item_id="odor-z",
            days_old=1.0,
        )
    )

    results = memory.retrieve(
        (1.0, 0.0),
        as_of_utc=NOW,
        top_k=2,
        apply_temporal_decay=True,
    )

    assert results[0].memory_id == "memory-new"
    assert (
        results[0].temporal_weight
        > results[1].temporal_weight
    )


def test_disabling_decay_removes_temporal_advantage() -> None:
    memory = make_memory(decay_rate=0.2)
    memory.add(
        make_record(
            memory_id="memory-old",
            odor_item_id="odor-a",
            days_old=10.0,
        )
    )
    memory.add(
        make_record(
            memory_id="memory-new",
            odor_item_id="odor-z",
            days_old=1.0,
        )
    )

    results = memory.retrieve(
        (1.0, 0.0),
        as_of_utc=NOW,
        top_k=2,
        apply_temporal_decay=False,
    )

    assert results[0].memory_id == "memory-old"
    assert results[0].temporal_weight == pytest.approx(1.0)
    assert results[1].temporal_weight == pytest.approx(1.0)


def test_inactive_memories_are_excluded() -> None:
    memory = make_memory()
    memory.add(
        make_record(
            memory_id="memory-active",
            odor_item_id="odor-active",
        )
    )
    memory.add(
        make_record(
            memory_id="memory-inactive",
            odor_item_id="odor-inactive",
            active=False,
        )
    )

    results = memory.retrieve(
        (1.0, 0.0),
        as_of_utc=NOW,
        top_k=10,
    )

    assert len(results) == 1
    assert results[0].memory_id == "memory-active"


def test_query_before_memory_update_is_rejected() -> None:
    memory = make_memory()
    memory.add(
        make_record(
            memory_id="memory-001",
            odor_item_id="odor-001",
        )
    )

    with pytest.raises(
        AssociativeMemoryError,
        match="cannot precede",
    ):
        memory.retrieve(
            (1.0, 0.0),
            as_of_utc=NOW - timedelta(seconds=1),
        )


def test_zero_query_is_rejected() -> None:
    memory = make_memory()
    memory.add(
        make_record(
            memory_id="memory-001",
            odor_item_id="odor-001",
        )
    )

    with pytest.raises(
        AssociativeMemoryError,
        match="nonzero finite norm",
    ):
        memory.retrieve(
            (0.0, 0.0),
            as_of_utc=NOW,
        )


def test_top_k_limits_returned_candidates() -> None:
    memory = make_memory(decay_rate=0.0)

    for index in range(3):
        memory.add(
            make_record(
                memory_id=f"memory-{index}",
                odor_item_id=f"odor-{index}",
            )
        )

    results = memory.retrieve(
        (1.0, 0.0),
        as_of_utc=NOW,
        top_k=2,
    )

    assert len(results) == 2
    assert [item.rank for item in results] == [1, 2]