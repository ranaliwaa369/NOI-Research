"""Tests for the nine-system NOI v0.3 execution policy."""

import inspect

import pytest

from src.evaluation.multisensory_records import (
    LatentMultisensoryEvent,
    MultisensorySplit,
    MultisensoryTarget,
    SupportRegime,
)
from src.evaluation.noi_v0_3_retrieval import (
    NOIV03RetrievalError,
    NOIV03System,
    NOIV03SystemPolicy,
)
from src.evaluation.reliability_fusion import (
    FusionAction,
    LockedFusionDecision,
    LockedFusionTrace,
)
from src.evaluation.support_gate import (
    SupportDecision,
    SupportMethod,
    UncertaintyStatus,
)


def odor(index: int) -> tuple[float, ...]:
    values = [0.0] * 16
    values[index] = 1.0
    return tuple(values)


def touch(index: int) -> tuple[float, ...]:
    values = [0.0] * 8
    values[index] = 1.0
    return tuple(values)


def target(
    item_id: str,
    family_id: int,
    index: int,
) -> MultisensoryTarget:
    return MultisensoryTarget(
        item_id=item_id,
        family_id=family_id,
        olfactory_prototype=odor(index),
        tactile_prototype=touch(index),
    )


def event(
    event_id: str,
    item: MultisensoryTarget,
) -> LatentMultisensoryEvent:
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


def policy() -> NOIV03SystemPolicy:
    item_a = target("item-a", 0, 0)
    item_b = target("item-b", 1, 1)

    return NOIV03SystemPolicy.fit(
        training_events=(
            event("train-a-1", item_a),
            event("train-a-2", item_a),
            event("train-b-1", item_b),
            event("train-b-2", item_b),
        ),
        targets=(
            item_a,
            item_b,
            target("item-withheld", 2, 2),
        ),
        ridge_alpha=1.0,
    )


def support_decision(
    *,
    status: UncertaintyStatus,
    supported: bool,
    request_touch: bool,
) -> SupportDecision:
    return SupportDecision(
        event_id="query",
        method=SupportMethod.MAHALANOBIS,
        support_score=0.5,
        threshold=0.5,
        is_supported=supported,
        uncertainty_status=status,
        request_touch=request_touch,
    )


def fusion_decision(
    *,
    action: FusionAction,
    odor_weight: float,
    touch_weight: float,
) -> LockedFusionDecision:
    abstained = action is FusionAction.ABSTAIN

    trace = LockedFusionTrace(
        event_id="query",
        odor_available=True,
        touch_available=True,
        odor_reliability=0.8,
        touch_reliability=0.6,
        reliability_threshold=0.2,
        conflict_available=True,
        conflict_score=0.1,
        conflict_threshold=0.5,
        conflict_detected=False,
        temporal_offset_steps=0,
        temporal_conflict_detected=False,
        selected_action=action,
        odor_weight=odor_weight,
        touch_weight=touch_weight,
        reason="Test evidence decision.",
        generator_version="test",
        condition_metadata_used=False,
        target_labels_used=False,
        final_test_labels_used=False,
    )

    fused_vector = (
        None
        if abstained
        else (
            tuple(value * odor_weight for value in odor(0))
            + tuple(value * touch_weight for value in touch(1))
        )
    )

    return LockedFusionDecision(
        event_id="query",
        action=action,
        odor_reliability=0.8,
        touch_reliability=0.6,
        odor_weight=odor_weight,
        touch_weight=touch_weight,
        conflict_score=0.1,
        conflict_detected=False,
        temporal_conflict_detected=False,
        abstained=abstained,
        fused_vector=fused_vector,
        trace=trace,
    )


def test_system_registry_matches_execution_specification() -> None:
    assert {system.value for system in NOIV03System} == {
        "odor_only_ridge",
        "odor_only_cosine",
        "touch_only_ridge",
        "touch_only_cosine",
        "naive_concatenation",
        "fixed_weight_fusion",
        "support_gate_odor_only",
        "reliability_gated_olfactory_tactile_fusion",
        "support_gate_reliability_fusion_with_abstention",
    }


@pytest.mark.parametrize(
    "system",
    (
        NOIV03System.ODOR_ONLY_RIDGE,
        NOIV03System.ODOR_ONLY_COSINE,
        NOIV03System.TOUCH_ONLY_RIDGE,
        NOIV03System.TOUCH_ONLY_COSINE,
        NOIV03System.NAIVE_CONCATENATION,
        NOIV03System.FIXED_WEIGHT_FUSION,
    ),
)
def test_baseline_systems_produce_training_only_rankings(
    system: NOIV03System,
) -> None:
    result = policy().evaluate(
        system=system,
        event_id="query",
        olfactory_vector=odor(0),
        tactile_vector=touch(0),
        top_k=2,
    )

    assert result.system is system
    assert result.retrieval.ranking[0] == "item-a"
    assert "item-withheld" not in result.retrieval.ranking
    assert result.retrieval.abstained is False


def test_support_gate_odor_only_abstains_when_unsupported() -> None:
    result = policy().evaluate(
        system=NOIV03System.SUPPORT_GATE_ODOR_ONLY,
        event_id="query",
        olfactory_vector=odor(0),
        tactile_vector=touch(0),
        support_decision=support_decision(
            status=UncertaintyStatus.CERTAIN_UNSUPPORTED,
            supported=False,
            request_touch=False,
        ),
        top_k=2,
    )

    assert result.retrieval.abstained is True
    assert result.retrieval.ranking == ()


def test_support_gate_odor_only_retrieves_when_supported() -> None:
    result = policy().evaluate(
        system=NOIV03System.SUPPORT_GATE_ODOR_ONLY,
        event_id="query",
        olfactory_vector=odor(1),
        tactile_vector=touch(0),
        support_decision=support_decision(
            status=UncertaintyStatus.CERTAIN_SUPPORTED,
            supported=True,
            request_touch=False,
        ),
        top_k=2,
    )

    assert result.retrieval.ranking[0] == "item-b"
    assert result.retrieval.odor_weight == 1.0
    assert result.retrieval.touch_weight == 0.0


@pytest.mark.parametrize(
    ("action", "odor_weight", "touch_weight"),
    (
        (FusionAction.ODOR_ONLY, 1.0, 0.0),
        (FusionAction.TOUCH_ONLY, 0.0, 1.0),
        (FusionAction.FUSED, 0.6, 0.4),
    ),
)
def test_reliability_system_applies_locked_fusion_action(
    action: FusionAction,
    odor_weight: float,
    touch_weight: float,
) -> None:
    result = policy().evaluate(
        system=(
            NOIV03System
            .RELIABILITY_GATED_OLFACTORY_TACTILE_FUSION
        ),
        event_id="query",
        olfactory_vector=odor(0),
        tactile_vector=touch(1),
        fusion_decision=fusion_decision(
            action=action,
            odor_weight=odor_weight,
            touch_weight=touch_weight,
        ),
        top_k=2,
    )

    assert result.retrieval.abstained is False
    assert result.retrieval.odor_weight == odor_weight
    assert result.retrieval.touch_weight == touch_weight


def test_reliability_system_preserves_locked_abstention() -> None:
    result = policy().evaluate(
        system=(
            NOIV03System
            .RELIABILITY_GATED_OLFACTORY_TACTILE_FUSION
        ),
        event_id="query",
        olfactory_vector=odor(0),
        tactile_vector=touch(0),
        fusion_decision=fusion_decision(
            action=FusionAction.ABSTAIN,
            odor_weight=0.0,
            touch_weight=0.0,
        ),
        top_k=2,
    )

    assert result.retrieval.abstained is True
    assert result.retrieval.ranking == ()


def test_combined_system_abstains_when_certainly_unsupported() -> None:
    result = policy().evaluate(
        system=(
            NOIV03System
            .SUPPORT_GATE_RELIABILITY_FUSION_WITH_ABSTENTION
        ),
        event_id="query",
        olfactory_vector=odor(0),
        tactile_vector=touch(0),
        support_decision=support_decision(
            status=UncertaintyStatus.CERTAIN_UNSUPPORTED,
            supported=False,
            request_touch=False,
        ),
        fusion_decision=fusion_decision(
            action=FusionAction.FUSED,
            odor_weight=0.5,
            touch_weight=0.5,
        ),
        top_k=2,
    )

    assert result.retrieval.abstained is True


def test_combined_system_uses_touch_when_requested() -> None:
    result = policy().evaluate(
        system=(
            NOIV03System
            .SUPPORT_GATE_RELIABILITY_FUSION_WITH_ABSTENTION
        ),
        event_id="query",
        olfactory_vector=odor(0),
        tactile_vector=touch(1),
        support_decision=support_decision(
            status=UncertaintyStatus.UNCERTAIN,
            supported=True,
            request_touch=True,
        ),
        fusion_decision=fusion_decision(
            action=FusionAction.FUSED,
            odor_weight=0.4,
            touch_weight=0.6,
        ),
        top_k=2,
    )

    assert result.retrieval.abstained is False
    assert result.retrieval.touch_weight == 0.6


def test_combined_system_uses_odor_when_touch_not_requested() -> None:
    result = policy().evaluate(
        system=(
            NOIV03System
            .SUPPORT_GATE_RELIABILITY_FUSION_WITH_ABSTENTION
        ),
        event_id="query",
        olfactory_vector=odor(1),
        tactile_vector=touch(0),
        support_decision=support_decision(
            status=UncertaintyStatus.CERTAIN_SUPPORTED,
            supported=True,
            request_touch=False,
        ),
        fusion_decision=fusion_decision(
            action=FusionAction.FUSED,
            odor_weight=0.5,
            touch_weight=0.5,
        ),
        top_k=2,
    )

    assert result.retrieval.ranking[0] == "item-b"
    assert result.retrieval.odor_weight == 1.0
    assert result.retrieval.touch_weight == 0.0


def test_required_locked_decision_cannot_be_omitted() -> None:
    with pytest.raises(
        NOIV03RetrievalError,
        match="fusion_decision",
    ):
        policy().evaluate(
            system=(
                NOIV03System
                .RELIABILITY_GATED_OLFACTORY_TACTILE_FUSION
            ),
            event_id="query",
            olfactory_vector=odor(0),
            tactile_vector=touch(0),
            top_k=2,
        )


def test_policy_inference_signature_contains_no_ground_truth() -> None:
    parameters = set(
        inspect.signature(
            NOIV03SystemPolicy.evaluate
        ).parameters
    )

    assert "condition" not in parameters
    assert "support_regime" not in parameters
    assert "target_item_id" not in parameters
    assert "target_family_id" not in parameters
    assert "modality_conflict" not in parameters
    assert "olfactory_quality" not in parameters
    assert "tactile_quality" not in parameters


def test_policy_is_deterministic() -> None:
    evaluator = policy()

    arguments = {
        "system": NOIV03System.FIXED_WEIGHT_FUSION,
        "event_id": "query",
        "olfactory_vector": odor(0),
        "tactile_vector": touch(1),
        "top_k": 2,
    }

    assert evaluator.evaluate(**arguments) == evaluator.evaluate(
        **arguments
    )
