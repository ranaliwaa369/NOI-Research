"""Tests for the transparent cosine odor retriever."""

import pytest

from src.models import OdorLibraryItem
from src.retrieval.cosine_retriever import (
    CosineOdorRetriever,
    RetrievalError,
)


def make_library() -> tuple[OdorLibraryItem, ...]:
    return (
        OdorLibraryItem(
            item_id="odor-a",
            odor_vector=(1.0, 0.0),
            descriptors=("descriptor-a",),
        ),
        OdorLibraryItem(
            item_id="odor-b",
            odor_vector=(0.0, 1.0),
            descriptors=("descriptor-b",),
        ),
        OdorLibraryItem(
            item_id="odor-c",
            odor_vector=(-1.0, 0.0),
            descriptors=("descriptor-c",),
        ),
    )


def test_retriever_reports_library_properties() -> None:
    retriever = CosineOdorRetriever(make_library())

    assert retriever.dimension == 2
    assert retriever.library_size == 3


def test_retrieval_ranks_most_similar_item_first() -> None:
    retriever = CosineOdorRetriever(make_library())

    results = retriever.retrieve((0.9, 0.1), top_k=3)

    assert [result.item_id for result in results] == [
        "odor-a",
        "odor-b",
        "odor-c",
    ]
    assert [result.rank for result in results] == [1, 2, 3]


def test_query_scaling_does_not_change_ranking() -> None:
    retriever = CosineOdorRetriever(make_library())

    first = retriever.retrieve((1.0, 0.2), top_k=3)
    second = retriever.retrieve((10.0, 2.0), top_k=3)

    assert [item.item_id for item in first] == [
        item.item_id for item in second
    ]


def test_top_k_is_capped_by_library_size() -> None:
    retriever = CosineOdorRetriever(make_library())

    results = retriever.retrieve((1.0, 0.0), top_k=100)

    assert len(results) == 3


def test_equal_scores_use_item_id_tie_breaking() -> None:
    library = (
        OdorLibraryItem(
            item_id="odor-z",
            odor_vector=(1.0, 0.0),
            descriptors=("z",),
        ),
        OdorLibraryItem(
            item_id="odor-a",
            odor_vector=(1.0, 0.0),
            descriptors=("a",),
        ),
    )
    retriever = CosineOdorRetriever(library)

    results = retriever.retrieve((1.0, 0.0), top_k=2)

    assert [result.item_id for result in results] == [
        "odor-a",
        "odor-z",
    ]


def test_empty_library_is_rejected() -> None:
    with pytest.raises(
        RetrievalError,
        match="at least one item",
    ):
        CosineOdorRetriever(())


def test_duplicate_item_ids_are_rejected() -> None:
    duplicated = (
        OdorLibraryItem(
            item_id="odor-a",
            odor_vector=(1.0, 0.0),
            descriptors=("first",),
        ),
        OdorLibraryItem(
            item_id="odor-a",
            odor_vector=(0.0, 1.0),
            descriptors=("second",),
        ),
    )

    with pytest.raises(
        RetrievalError,
        match="item_id must be unique",
    ):
        CosineOdorRetriever(duplicated)


def test_mismatched_library_dimensions_are_rejected() -> None:
    library = (
        OdorLibraryItem(
            item_id="odor-a",
            odor_vector=(1.0, 0.0),
            descriptors=("a",),
        ),
        OdorLibraryItem(
            item_id="odor-b",
            odor_vector=(1.0, 0.0, 0.0),
            descriptors=("b",),
        ),
    )

    with pytest.raises(
        RetrievalError,
        match="identical dimensions",
    ):
        CosineOdorRetriever(library)


def test_incorrect_query_dimension_is_rejected() -> None:
    retriever = CosineOdorRetriever(make_library())

    with pytest.raises(
        RetrievalError,
        match="query dimension",
    ):
        retriever.retrieve((1.0, 0.0, 0.0))


def test_zero_query_is_rejected() -> None:
    retriever = CosineOdorRetriever(make_library())

    with pytest.raises(
        RetrievalError,
        match="nonzero finite norm",
    ):
        retriever.retrieve((0.0, 0.0))


def test_invalid_top_k_is_rejected() -> None:
    retriever = CosineOdorRetriever(make_library())

    with pytest.raises(
        RetrievalError,
        match="at least 1",
    ):
        retriever.retrieve((1.0, 0.0), top_k=0)