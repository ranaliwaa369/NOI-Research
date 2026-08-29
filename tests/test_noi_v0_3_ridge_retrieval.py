"""Tests for training-only NOI v0.3 ridge retrieval."""

import math

import pytest

from src.evaluation.multisensory_records import (
    LatentMultisensoryEvent,
    MultisensorySplit,
    MultisensoryTarget,
    SupportRegime,
)
from src.evaluation.noi_v0_3_retrieval import (
    NOIV03Modality,
    NOIV03RetrievalError,
    NOIV03RidgeRetriever,
)


def odor(index: int) -> tuple[float, ...]:
    """Return one nonzero 16D vector."""

    values = [0.0] * 16
    values[index] = 1.0
    return tuple(values)


def touch(index: int) -> tuple[float, ...]:
    """Return one nonzero 8D vector."""

    values = [0.0] * 8
    values[index] = 1.0
    return tuple(values)


def target(
    item_id: str,
    family_id: int,
    index: int,
) -> MultisensoryTarget:
    """Create one paired deterministic target."""

    return MultisensoryTarget(
        item_id=item_id,
        family_id=family_id,
        olfactory_prototype=odor(index),
        tactile_prototype=touch(index),
    )


def event(
    event_id: str,
    item: MultisensoryTarget,
    *,
    split: MultisensorySplit = MultisensorySplit.TRAIN,
) -> LatentMultisensoryEvent:
    """Create one deterministic latent event."""

    regime = (
        SupportRegime.DEVELOPMENT
        if split is MultisensorySplit.TRAIN
        else SupportRegime.SEEN_ITEM
    )

    return LatentMultisensoryEvent(
        latent_event_id=event_id,
        split=split,
        template_id=0,
        target_item_id=item.item_id,
        target_family_id=item.family_id,
        support_regime=regime,
        olfactory_vector=item.olfactory_prototype,
        tactile_vector=item.tactile_prototype,
        generator_seed=1301,
    )


def records() -> tuple[
    tuple[LatentMultisensoryEvent, ...],
    tuple[MultisensoryTarget, ...],
]:
    """Return training events plus one unrepresented target."""

    item_a = target("item-a", 0, 0)
    item_b = target("item-b", 1, 1)
    withheld = target("item-withheld", 2, 2)

    training = (
        event("train-a-1", item_a),
        event("train-a-2", item_a),
        event("train-b-1", item_b),
        event("train-b-2", item_b),
    )

    return training, (
        item_a,
        item_b,
        withheld,
    )


@pytest.mark.parametrize(
    "modality",
    (
        NOIV03Modality.ODOR,
        NOIV03Modality.TOUCH,
    ),
)
def test_ridge_fit_uses_training_items_only(
    modality: NOIV03Modality,
) -> None:
    """Ridge memory cannot include an unrepresented target."""

    training, targets = records()

    retriever = NOIV03RidgeRetriever(
        modality=modality,
        alpha=1.0,
    ).fit(
        training_events=training,
        targets=targets,
    )

    assert retriever.is_fitted is True
    assert retriever.training_event_count == 4
    assert retriever.item_ids == (
        "item-a",
        "item-b",
    )
    assert "item-withheld" not in retriever.item_ids


def test_odor_ridge_ranks_matching_item_first() -> None:
    """Odor ridge maps 16D evidence to odor target space."""

    training, targets = records()

    retriever = NOIV03RidgeRetriever(
        modality=NOIV03Modality.ODOR,
        alpha=1.0,
    ).fit(
        training_events=training,
        targets=targets,
    )

    result = retriever.retrieve(
        event_id="odor-query",
        query_vector=odor(1),
        top_k=2,
    )

    assert result.ranking[0] == "item-b"
    assert result.abstained is False
    assert result.odor_weight == 1.0
    assert result.touch_weight == 0.0


def test_touch_ridge_ranks_matching_item_first() -> None:
    """Touch ridge maps 8D evidence to touch target space."""

    training, targets = records()

    retriever = NOIV03RidgeRetriever(
        modality=NOIV03Modality.TOUCH,
        alpha=1.0,
    ).fit(
        training_events=training,
        targets=targets,
    )

    result = retriever.retrieve(
        event_id="touch-query",
        query_vector=touch(0),
        top_k=2,
    )

    assert result.ranking[0] == "item-a"
    assert result.odor_weight == 0.0
    assert result.touch_weight == 1.0


def test_ridge_retrieval_is_deterministic() -> None:
    """Identical fitted evidence reproduces exactly."""

    training, targets = records()

    retriever = NOIV03RidgeRetriever(
        modality=NOIV03Modality.ODOR,
        alpha=1.0,
    ).fit(
        training_events=training,
        targets=targets,
    )

    first = retriever.retrieve(
        event_id="repeat",
        query_vector=odor(0),
        top_k=2,
    )
    second = retriever.retrieve(
        event_id="repeat",
        query_vector=odor(0),
        top_k=2,
    )

    assert first == second


def test_ridge_output_is_finite() -> None:
    """All emitted similarity scores must be finite."""

    training, targets = records()

    retriever = NOIV03RidgeRetriever(
        modality=NOIV03Modality.TOUCH,
        alpha=1.0,
    ).fit(
        training_events=training,
        targets=targets,
    )

    result = retriever.retrieve(
        event_id="finite",
        query_vector=touch(1),
        top_k=2,
    )

    assert all(math.isfinite(score) for score in result.scores)


def test_unfitted_ridge_is_rejected() -> None:
    """Inference cannot occur before training-only fitting."""

    retriever = NOIV03RidgeRetriever(
        modality=NOIV03Modality.ODOR,
        alpha=1.0,
    )

    with pytest.raises(
        NOIV03RetrievalError,
        match="fitted",
    ):
        retriever.retrieve(
            event_id="unfitted",
            query_vector=odor(0),
            top_k=1,
        )


def test_nontraining_records_are_rejected() -> None:
    """Validation or final events cannot fit ridge parameters."""

    training, targets = records()
    corrupted = training[0]

    object.__setattr__(
        corrupted,
        "split",
        MultisensorySplit.FINAL_TEST,
    )

    with pytest.raises(
        NOIV03RetrievalError,
        match="training",
    ):
        NOIV03RidgeRetriever(
            modality=NOIV03Modality.ODOR,
            alpha=1.0,
        ).fit(
            training_events=training,
            targets=targets,
        )


@pytest.mark.parametrize(
    "alpha",
    (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        True,
    ),
)
def test_invalid_ridge_alpha_is_rejected(
    alpha: float,
) -> None:
    """The registered ridge penalty must be finite and positive."""

    with pytest.raises(
        NOIV03RetrievalError,
        match="positive",
    ):
        NOIV03RidgeRetriever(
            modality=NOIV03Modality.ODOR,
            alpha=alpha,
        )


def test_unknown_modality_is_rejected() -> None:
    """Only the two registered modality spaces are allowed."""

    with pytest.raises(
        NOIV03RetrievalError,
        match="NOIV03Modality",
    ):
        NOIV03RidgeRetriever(
            modality="odor",  # type: ignore[arg-type]
            alpha=1.0,
        )


def test_query_dimension_matches_selected_modality() -> None:
    """Odor and touch ridge inputs cannot be interchanged."""

    training, targets = records()

    retriever = NOIV03RidgeRetriever(
        modality=NOIV03Modality.ODOR,
        alpha=1.0,
    ).fit(
        training_events=training,
        targets=targets,
    )

    with pytest.raises(
        NOIV03RetrievalError,
        match="dimension 16",
    ):
        retriever.retrieve(
            event_id="wrong-dimension",
            query_vector=touch(0),
            top_k=1,
        )


def test_ridge_result_contains_no_ground_truth() -> None:
    """Ridge inference output cannot expose target answers."""

    training, targets = records()

    result = NOIV03RidgeRetriever(
        modality=NOIV03Modality.ODOR,
        alpha=1.0,
    ).fit(
        training_events=training,
        targets=targets,
    ).retrieve(
        event_id="schema",
        query_vector=odor(0),
        top_k=2,
    )

    fields = set(result.__dataclass_fields__)

    assert "target_item_id" not in fields
    assert "target_family_id" not in fields
    assert "condition" not in fields
    assert "support_regime" not in fields
