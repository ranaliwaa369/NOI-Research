"""Prespecified retrieval metrics for NOI baseline comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from math import log2


Ranking = Sequence[str]
RelevantItems = Sequence[str] | set[str] | frozenset[str]


class MetricInputError(ValueError):
    """Raised when retrieval-metric inputs are invalid."""


def recall_at_k(
    rankings: Sequence[Ranking],
    relevant_items: Sequence[RelevantItems],
    *,
    k: int,
) -> float:
    """Return macro-averaged Recall@k across independent queries."""

    validated = _validate_inputs(rankings, relevant_items, k=k)

    recalls = []

    for ranking, relevant in validated:
        retrieved = set(ranking[:k])
        recalls.append(len(retrieved & relevant) / len(relevant))

    return sum(recalls) / len(recalls)


def mean_reciprocal_rank(
    rankings: Sequence[Ranking],
    relevant_items: Sequence[RelevantItems],
) -> float:
    """Return mean reciprocal rank of the first relevant item."""

    validated = _validate_inputs(
        rankings,
        relevant_items,
        k=None,
    )

    reciprocal_ranks = []

    for ranking, relevant in validated:
        reciprocal_rank = 0.0

        for rank, item_id in enumerate(ranking, start=1):
            if item_id in relevant:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg_at_k(
    rankings: Sequence[Ranking],
    relevant_items: Sequence[RelevantItems],
    *,
    k: int,
) -> float:
    """Return macro-averaged binary-relevance nDCG@k."""

    validated = _validate_inputs(rankings, relevant_items, k=k)

    values = []

    for ranking, relevant in validated:
        dcg = sum(
            1.0 / log2(rank + 1)
            for rank, item_id in enumerate(ranking[:k], start=1)
            if item_id in relevant
        )

        ideal_relevant_count = min(k, len(relevant))

        idcg = sum(
            1.0 / log2(rank + 1)
            for rank in range(1, ideal_relevant_count + 1)
        )

        values.append(dcg / idcg)

    return sum(values) / len(values)


def _validate_inputs(
    rankings: Sequence[Ranking],
    relevant_items: Sequence[RelevantItems],
    *,
    k: int | None,
) -> list[tuple[tuple[str, ...], frozenset[str]]]:
    """Validate and normalize paired ranking and relevance inputs."""

    if not rankings:
        raise MetricInputError(
            "At least one ranking is required."
        )

    if len(rankings) != len(relevant_items):
        raise MetricInputError(
            "rankings and relevant_items must have equal lengths."
        )

    if k is not None:
        if isinstance(k, bool) or not isinstance(k, int):
            raise MetricInputError("k must be an integer.")

        if k < 1:
            raise MetricInputError("k must be at least 1.")

    validated = []

    for index, (ranking, relevant) in enumerate(
        zip(rankings, relevant_items, strict=True)
    ):
        normalized_ranking = tuple(ranking)
        normalized_relevant = frozenset(relevant)

        if not normalized_ranking:
            raise MetricInputError(
                f"Ranking {index} must not be empty."
            )

        if not normalized_relevant:
            raise MetricInputError(
                f"Relevant-item set {index} must not be empty."
            )

        if any(
            not isinstance(item_id, str) or not item_id.strip()
            for item_id in normalized_ranking
        ):
            raise MetricInputError(
                f"Ranking {index} contains an invalid item identifier."
            )

        if any(
            not isinstance(item_id, str) or not item_id.strip()
            for item_id in normalized_relevant
        ):
            raise MetricInputError(
                f"Relevant-item set {index} contains an invalid identifier."
            )

        if len(normalized_ranking) != len(set(normalized_ranking)):
            raise MetricInputError(
                f"Ranking {index} contains duplicate item identifiers."
            )

        validated.append(
            (normalized_ranking, normalized_relevant)
        )

    return validated