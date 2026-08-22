"""Tests for the prespecified NOI retrieval metrics."""

import pytest

from src.evaluation.retrieval_metrics import (
    MetricInputError,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


RANKINGS = (
    ("odor-a", "odor-b", "odor-c"),
    ("odor-x", "odor-y", "odor-z"),
)

RELEVANT = (
    {"odor-a"},
    {"odor-z"},
)


def test_recall_at_one() -> None:
    result = recall_at_k(RANKINGS, RELEVANT, k=1)

    assert result == pytest.approx(0.5)


def test_recall_at_three() -> None:
    result = recall_at_k(RANKINGS, RELEVANT, k=3)

    assert result == pytest.approx(1.0)


def test_mean_reciprocal_rank() -> None:
    result = mean_reciprocal_rank(RANKINGS, RELEVANT)

    expected = (1.0 + (1.0 / 3.0)) / 2.0
    assert result == pytest.approx(expected)


def test_ndcg_at_three() -> None:
    result = ndcg_at_k(RANKINGS, RELEVANT, k=3)

    expected = (1.0 + 0.5) / 2.0
    assert result == pytest.approx(expected)


def test_multiple_relevant_items_use_fractional_recall() -> None:
    rankings = (("odor-a", "odor-b", "odor-c"),)
    relevant = ({"odor-a", "odor-c"},)

    result = recall_at_k(rankings, relevant, k=2)

    assert result == pytest.approx(0.5)


def test_empty_rankings_are_rejected() -> None:
    with pytest.raises(
        MetricInputError,
        match="At least one ranking",
    ):
        recall_at_k((), (), k=1)


def test_unequal_batch_lengths_are_rejected() -> None:
    with pytest.raises(
        MetricInputError,
        match="equal lengths",
    ):
        mean_reciprocal_rank(
            (("odor-a",), ("odor-b",)),
            ({"odor-a"},),
        )


def test_invalid_k_is_rejected() -> None:
    with pytest.raises(
        MetricInputError,
        match="at least 1",
    ):
        recall_at_k(RANKINGS, RELEVANT, k=0)


def test_empty_relevance_set_is_rejected() -> None:
    with pytest.raises(
        MetricInputError,
        match="must not be empty",
    ):
        ndcg_at_k(
            (("odor-a",),),
            (set(),),
            k=1,
        )


def test_duplicate_ranked_items_are_rejected() -> None:
    with pytest.raises(
        MetricInputError,
        match="duplicate item identifiers",
    ):
        mean_reciprocal_rank(
            (("odor-a", "odor-a"),),
            ({"odor-a"},),
        )