"""Tests for prespecified NOI retrieval baselines."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.retrieval_baselines import (
    BaselineError,
    BaselineKind,
    RidgeFusionRetriever,
    build_odor_library,
    evaluate_baseline,
    mean_fuse_event,
)
from src.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.synthetic_generator import generate_synthetic_pilot
from src.evaluation.synthetic_records import SplitLabel


SYNTHETIC_CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"


@pytest.fixture(scope="module")
def pilot_dataset():
    """Create one deterministic dataset for baseline tests."""

    return generate_synthetic_pilot(
        SYNTHETIC_CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )


def test_odor_library_is_complete_and_sorted(
    pilot_dataset,
) -> None:
    """Every synthetic target must appear once in the fixed library."""

    library = build_odor_library(pilot_dataset)
    item_ids = tuple(item.item_id for item in library)

    assert len(library) == 200
    assert len(set(item_ids)) == 200
    assert item_ids == tuple(sorted(item_ids))


def test_mean_fusion_returns_unit_vector(
    pilot_dataset,
) -> None:
    """Mean-fused contextual vectors must be finite and normalized."""

    event = pilot_dataset.events[0]
    fused = mean_fuse_event(event)

    assert fused.ndim == 1
    assert np.all(np.isfinite(fused))
    assert np.linalg.norm(fused) == pytest.approx(1.0)


def test_random_baseline_is_deterministic(
    pilot_dataset,
) -> None:
    """The same seed and events must reproduce identical rankings."""

    first = evaluate_baseline(
        pilot_dataset,
        baseline=BaselineKind.RANDOM,
        split=SplitLabel.VALIDATION,
        top_k=10,
        random_seed=2026,
    )

    second = evaluate_baseline(
        pilot_dataset,
        baseline=BaselineKind.RANDOM,
        split=SplitLabel.VALIDATION,
        top_k=10,
        random_seed=2026,
    )

    assert first.event_ids == second.event_ids
    assert first.rankings == second.rankings
    assert first.relevant_items == second.relevant_items


def test_random_baseline_changes_with_seed(
    pilot_dataset,
) -> None:
    """Different random seeds must produce different rankings."""

    first = evaluate_baseline(
        pilot_dataset,
        baseline=BaselineKind.RANDOM,
        split=SplitLabel.VALIDATION,
        random_seed=2026,
    )

    second = evaluate_baseline(
        pilot_dataset,
        baseline=BaselineKind.RANDOM,
        split=SplitLabel.VALIDATION,
        random_seed=2027,
    )

    assert first.rankings != second.rankings


def test_text_only_baseline_returns_valid_rankings(
    pilot_dataset,
) -> None:
    """Text-only retrieval must rank unique library items."""

    evaluation = evaluate_baseline(
        pilot_dataset,
        baseline=BaselineKind.TEXT_ONLY_COSINE,
        split=SplitLabel.VALIDATION,
        top_k=10,
    )

    assert len(evaluation.event_ids) == 20
    assert len(evaluation.rankings) == 20
    assert evaluation.training_event_count == 140

    for ranking in evaluation.rankings:
        assert len(ranking) == 10
        assert len(set(ranking)) == 10


def test_mean_fusion_baseline_returns_valid_rankings(
    pilot_dataset,
) -> None:
    """Transparent multimodal fusion must return valid rankings."""

    evaluation = evaluate_baseline(
        pilot_dataset,
        baseline=BaselineKind.MEAN_FUSION_COSINE,
        split=SplitLabel.VALIDATION,
        top_k=10,
    )

    assert len(evaluation.rankings) == 20
    assert all(
        len(ranking) == 10
        for ranking in evaluation.rankings
    )


def test_ridge_baseline_fits_training_events_only(
    pilot_dataset,
) -> None:
    """The learned baseline must fit only the locked training split."""

    retriever = RidgeFusionRetriever(alpha=1.0)

    assert retriever.is_fitted is False

    retriever.fit(pilot_dataset)

    assert retriever.is_fitted is True
    assert retriever.training_event_count == 140

    validation_event = next(
        event
        for event in pilot_dataset.events
        if event.split is SplitLabel.VALIDATION
    )

    ranking = retriever.retrieve(
        validation_event,
        top_k=10,
    )

    assert len(ranking) == 10
    assert len(set(ranking)) == 10


def test_unfitted_ridge_retriever_is_rejected(
    pilot_dataset,
) -> None:
    """Retrieval before fitting must fail explicitly."""

    retriever = RidgeFusionRetriever(alpha=1.0)

    with pytest.raises(
        BaselineError,
        match="fitted before retrieval",
    ):
        retriever.retrieve(
            pilot_dataset.events[0],
            top_k=10,
        )


@pytest.mark.parametrize(
    "invalid_alpha",
    (
        -1.0,
        float("inf"),
        float("nan"),
        True,
    ),
)
def test_invalid_ridge_alpha_is_rejected(
    invalid_alpha,
) -> None:
    """Invalid regularization values must be rejected."""

    with pytest.raises(
        BaselineError,
        match="finite and nonnegative",
    ):
        RidgeFusionRetriever(alpha=invalid_alpha)


def test_top_k_larger_than_library_is_rejected(
    pilot_dataset,
) -> None:
    """A requested ranking cannot exceed the candidate library."""

    with pytest.raises(
        BaselineError,
        match="cannot exceed",
    ):
        evaluate_baseline(
            pilot_dataset,
            baseline=BaselineKind.RANDOM,
            split=SplitLabel.VALIDATION,
            top_k=201,
        )


@pytest.mark.parametrize(
    "baseline",
    tuple(BaselineKind),
)
def test_all_baselines_are_compatible_with_metrics(
    pilot_dataset,
    baseline: BaselineKind,
) -> None:
    """Every baseline must produce inputs accepted by locked metrics."""

    evaluation = evaluate_baseline(
        pilot_dataset,
        baseline=baseline,
        split=SplitLabel.VALIDATION,
        top_k=10,
        random_seed=2026,
        ridge_alpha=1.0,
    )

    recall_1 = recall_at_k(
        evaluation.rankings,
        evaluation.relevant_items,
        k=1,
    )

    recall_10 = recall_at_k(
        evaluation.rankings,
        evaluation.relevant_items,
        k=10,
    )

    mrr = mean_reciprocal_rank(
        evaluation.rankings,
        evaluation.relevant_items,
    )

    ndcg_10 = ndcg_at_k(
        evaluation.rankings,
        evaluation.relevant_items,
        k=10,
    )

    for metric in (
        recall_1,
        recall_10,
        mrr,
        ndcg_10,
    ):
        assert 0.0 <= metric <= 1.0

    assert recall_10 >= recall_1


def test_ood_split_contains_expected_events(
    pilot_dataset,
) -> None:
    """OOD evaluation must operate on the held-out 40-event split."""

    evaluation = evaluate_baseline(
        pilot_dataset,
        baseline=BaselineKind.RIDGE_FUSION,
        split=SplitLabel.OOD_TEST,
        top_k=10,
    )

    assert len(evaluation.event_ids) == 40
    assert len(evaluation.rankings) == 40
    assert len(evaluation.relevant_items) == 40
    assert evaluation.training_event_count == 140