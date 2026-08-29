"""Tests for metadata-blind NOI v0.3 retrieval mechanics."""

import math

import pytest

from src.evaluation.multisensory_records import (
    LatentMultisensoryEvent,
    MultisensorySplit,
    MultisensoryTarget,
    SupportRegime,
)
from src.evaluation.noi_v0_3_retrieval import (
    NOIV03RetrievalError,
    NOIV03RetrievalLibrary,
)


def odor(index: int) -> tuple[float, ...]:
    """Return a nonzero deterministic 16D odor prototype."""

    values = [0.0] * 16
    values[index] = 1.0
    return tuple(values)


def touch(index: int) -> tuple[float, ...]:
    """Return a nonzero deterministic 8D touch prototype."""

    values = [0.0] * 8
    values[index] = 1.0
    return tuple(values)


def target(
    item_id: str,
    family_id: int,
    odor_index: int,
    touch_index: int,
) -> MultisensoryTarget:
    """Create one deterministic target."""

    return MultisensoryTarget(
        item_id=item_id,
        family_id=family_id,
        olfactory_prototype=odor(odor_index),
        tactile_prototype=touch(touch_index),
    )


def training_event(
    event_id: str,
    item: MultisensoryTarget,
) -> LatentMultisensoryEvent:
    """Create one valid training event."""

    return LatentMultisensoryEvent(
        latent_event_id=event_id,
        split=MultisensorySplit.TRAIN,
        template_id=0,
        target_item_id=item.item_id,
        target_family_id=item.family_id,
        support_regime=SupportRegime.DEVELOPMENT,
        olfactory_vector=item.olfactory_prototype,
        tactile_vector=item.tactile_prototype,
        generator_seed=1301,
    )


def library() -> NOIV03RetrievalLibrary:
    """Build a two-item training-only retrieval library."""

    item_a = target("item-a", 0, 0, 0)
    item_b = target("item-b", 1, 1, 1)
    withheld = target("item-withheld", 2, 2, 2)

    return NOIV03RetrievalLibrary.from_training_records(
        training_events=(
            training_event("train-a", item_a),
            training_event("train-b", item_b),
        ),
        targets=(
            item_a,
            item_b,
            withheld,
        ),
    )


def test_library_contains_training_items_only() -> None:
    """Validation or final-only targets cannot enter retrieval memory."""

    retrieval = library()

    assert retrieval.item_ids == (
        "item-a",
        "item-b",
    )
    assert retrieval.library_size == 2
    assert "item-withheld" not in retrieval.item_ids


def test_odor_cosine_ranks_matching_training_item_first() -> None:
    """Odor-only cosine must use the registered 16D modality."""

    result = library().rank(
        event_id="query-odor",
        olfactory_vector=odor(1),
        tactile_vector=None,
        odor_weight=1.0,
        touch_weight=0.0,
        top_k=2,
    )

    assert result.ranking == (
        "item-b",
        "item-a",
    )
    assert result.scores[0] > result.scores[1]
    assert result.abstained is False


def test_touch_cosine_ranks_matching_training_item_first() -> None:
    """Touch-only cosine must use the registered 8D modality."""

    result = library().rank(
        event_id="query-touch",
        olfactory_vector=None,
        tactile_vector=touch(0),
        odor_weight=0.0,
        touch_weight=1.0,
        top_k=2,
    )

    assert result.ranking[0] == "item-a"
    assert result.abstained is False


def test_weighted_score_fusion_combines_modalities() -> None:
    """Dynamic fusion must combine independent modality similarities."""

    result = library().rank(
        event_id="query-fused",
        olfactory_vector=odor(0),
        tactile_vector=touch(1),
        odor_weight=0.75,
        touch_weight=0.25,
        top_k=2,
    )

    assert result.ranking[0] == "item-a"
    assert result.odor_weight == 0.75
    assert result.touch_weight == 0.25
    assert all(math.isfinite(score) for score in result.scores)


def test_equal_score_tie_break_uses_item_id() -> None:
    """Equal similarity must use ascending item ID deterministically."""

    mixed_odor = tuple(
        1.0 if index in (0, 1) else 0.0
        for index in range(16)
    )

    result = library().rank(
        event_id="query-tie",
        olfactory_vector=mixed_odor,
        tactile_vector=None,
        odor_weight=1.0,
        touch_weight=0.0,
        top_k=2,
    )

    assert result.ranking == (
        "item-a",
        "item-b",
    )


def test_ranking_is_deterministic() -> None:
    """Identical metadata-blind evidence must reproduce exactly."""

    arguments = {
        "event_id": "query-repeat",
        "olfactory_vector": odor(0),
        "tactile_vector": touch(1),
        "odor_weight": 0.5,
        "touch_weight": 0.5,
        "top_k": 2,
    }

    first = library().rank(**arguments)
    second = library().rank(**arguments)

    assert first == second


def test_explicit_abstention_has_no_identity_ranking() -> None:
    """Unsupported queries cannot receive forced nearest-item answers."""

    result = library().abstain(
        event_id="query-unsupported",
        reason="Support gate rejected the query.",
    )

    assert result.abstained is True
    assert result.ranking == ()
    assert result.scores == ()
    assert result.odor_weight == 0.0
    assert result.touch_weight == 0.0


def test_unavailable_modality_cannot_receive_weight() -> None:
    """Missing modality evidence must always have zero weight."""

    with pytest.raises(
        NOIV03RetrievalError,
        match="Unavailable odor",
    ):
        library().rank(
            event_id="bad-odor-weight",
            olfactory_vector=None,
            tactile_vector=touch(0),
            odor_weight=0.5,
            touch_weight=0.5,
            top_k=2,
        )

    with pytest.raises(
        NOIV03RetrievalError,
        match="Unavailable touch",
    ):
        library().rank(
            event_id="bad-touch-weight",
            olfactory_vector=odor(0),
            tactile_vector=None,
            odor_weight=0.5,
            touch_weight=0.5,
            top_k=2,
        )


def test_available_weights_must_sum_to_one() -> None:
    """Weighted score fusion must retain an auditable unit total."""

    with pytest.raises(
        NOIV03RetrievalError,
        match="sum to 1",
    ):
        library().rank(
            event_id="bad-weight-total",
            olfactory_vector=odor(0),
            tactile_vector=touch(0),
            odor_weight=0.4,
            touch_weight=0.4,
            top_k=2,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("odor_weight", -0.1),
        ("odor_weight", 1.1),
        ("touch_weight", -0.1),
        ("touch_weight", 1.1),
        ("odor_weight", float("nan")),
        ("touch_weight", float("nan")),
    ),
)
def test_invalid_weights_are_rejected(
    field: str,
    value: float,
) -> None:
    """Fusion weights must remain finite probabilities."""

    arguments = {
        "event_id": "bad-weight",
        "olfactory_vector": odor(0),
        "tactile_vector": touch(0),
        "odor_weight": 0.5,
        "touch_weight": 0.5,
        "top_k": 2,
    }
    arguments[field] = value

    with pytest.raises(
        NOIV03RetrievalError,
        match="between 0 and 1",
    ):
        library().rank(**arguments)


def test_top_k_cannot_exceed_training_library() -> None:
    """Ranking cannot silently invent unavailable candidates."""

    with pytest.raises(
        NOIV03RetrievalError,
        match="library size",
    ):
        library().rank(
            event_id="bad-top-k",
            olfactory_vector=odor(0),
            tactile_vector=None,
            odor_weight=1.0,
            touch_weight=0.0,
            top_k=3,
        )


def test_nontraining_event_is_rejected_from_library_fit() -> None:
    """Only training records may establish retrieval memory."""

    item = target("item-a", 0, 0, 0)
    event = training_event("train-a", item)

    object.__setattr__(
        event,
        "split",
        MultisensorySplit.FINAL_TEST,
    )

    with pytest.raises(
        NOIV03RetrievalError,
        match="training",
    ):
        NOIV03RetrievalLibrary.from_training_records(
            training_events=(event,),
            targets=(item,),
        )


def test_missing_training_target_is_rejected() -> None:
    """Every represented training item needs its fixed prototype."""

    item = target("item-a", 0, 0, 0)

    with pytest.raises(
        NOIV03RetrievalError,
        match="absent",
    ):
        NOIV03RetrievalLibrary.from_training_records(
            training_events=(
                training_event("train-a", item),
            ),
            targets=(),
        )


def test_result_schema_contains_no_ground_truth() -> None:
    """Inference output cannot expose target answers or condition labels."""

    result = library().rank(
        event_id="schema-check",
        olfactory_vector=odor(0),
        tactile_vector=None,
        odor_weight=1.0,
        touch_weight=0.0,
        top_k=2,
    )

    fields = set(result.__dataclass_fields__)

    assert "target_item_id" not in fields
    assert "target_family_id" not in fields
    assert "support_regime" not in fields
    assert "condition" not in fields
    assert "modality_conflict" not in fields
