"""Tests for reliability- and conflict-gated NOI v0.3 fusion."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from src.evaluation.multisensory_records import (
    ConditionLabel,
    MultisensoryConditionView,
)
from src.evaluation.reliability_fusion import (
    FusionAction,
    FusionConfig,
    FusionDecision,
    FusionError,
    FusionMethod,
    FusionTrace,
    fuse_multisensory_view,
)


ODOR = tuple(
    (index + 1) / 20.0
    for index in range(16)
)
TOUCH = tuple(
    (index + 1) / 10.0
    for index in range(8)
)


def make_view(
    *,
    condition: ConditionLabel = ConditionLabel.CLEAN,
    olfactory_vector=ODOR,
    tactile_vector=TOUCH,
    olfactory_quality: float = 1.0,
    tactile_quality: float = 1.0,
    modality_conflict: bool = False,
    temporal_offset_steps: int = 0,
) -> MultisensoryConditionView:
    """Return one valid paired condition view."""

    return MultisensoryConditionView(
        view_id=f"latent-001-{condition.value}",
        latent_event_id="latent-001",
        condition=condition,
        target_item_id="item-001",
        target_family_id=1,
        olfactory_vector=olfactory_vector,
        tactile_vector=tactile_vector,
        olfactory_quality=olfactory_quality,
        tactile_quality=tactile_quality,
        modality_conflict=modality_conflict,
        temporal_offset_steps=temporal_offset_steps,
    )


def make_config(
    **changes: object,
) -> FusionConfig:
    """Return one feasibility reliability policy."""

    values: dict[str, object] = {
        "minimum_reliability": 0.30,
        "generator_version": "0.3.0-feasibility",
    }
    values.update(changes)

    return FusionConfig(**values)


def test_methods_include_proposed_and_two_prespecified_baselines() -> None:
    """The implementation exposes exactly the registered fusion methods."""

    assert tuple(FusionMethod) == (
        FusionMethod.RELIABILITY_GATED,
        FusionMethod.NAIVE_CONCATENATION,
        FusionMethod.FIXED_EQUAL,
    )


def test_actions_are_explicit() -> None:
    """Every operational outcome has a stable label."""

    assert tuple(FusionAction) == (
        FusionAction.ODOR_ONLY,
        FusionAction.TOUCH_ONLY,
        FusionAction.FUSED,
        FusionAction.ABSTAIN,
    )


def test_clean_reliable_view_is_fused() -> None:
    """Two reliable compatible modalities receive fused treatment."""

    decision = fuse_multisensory_view(
        view=make_view(),
        config=make_config(),
        method=FusionMethod.RELIABILITY_GATED,
    )

    assert isinstance(decision, FusionDecision)
    assert decision.action is FusionAction.FUSED
    assert decision.abstained is False
    assert decision.fused_vector is not None
    assert len(decision.fused_vector) == 24
    assert decision.odor_weight == pytest.approx(0.5)
    assert decision.touch_weight == pytest.approx(0.5)


def test_reliability_weights_are_normalized_from_quality() -> None:
    """Compatible evidence weights reflect relative modality reliability."""

    decision = fuse_multisensory_view(
        view=make_view(
            olfactory_quality=0.75,
            tactile_quality=0.25,
        ),
        config=make_config(
            minimum_reliability=0.20,
        ),
        method=FusionMethod.RELIABILITY_GATED,
    )

    assert decision.action is FusionAction.FUSED
    assert decision.odor_reliability == 0.75
    assert decision.touch_reliability == 0.25
    assert decision.odor_weight == pytest.approx(0.75)
    assert decision.touch_weight == pytest.approx(0.25)
    assert decision.odor_weight + decision.touch_weight == pytest.approx(1.0)


def test_weak_touch_selects_odor_only() -> None:
    """A reliable odor channel is not diluted by unreliable touch."""

    decision = fuse_multisensory_view(
        view=make_view(
            olfactory_quality=0.90,
            tactile_quality=0.20,
        ),
        config=make_config(),
        method=FusionMethod.RELIABILITY_GATED,
    )

    assert decision.action is FusionAction.ODOR_ONLY
    assert decision.odor_weight == 1.0
    assert decision.touch_weight == 0.0
    assert decision.fused_vector is not None
    assert decision.fused_vector[16:] == (0.0,) * 8


def test_weak_odor_selects_touch_only() -> None:
    """Reliable touch can be used when odor evidence is weak."""

    decision = fuse_multisensory_view(
        view=make_view(
            olfactory_quality=0.20,
            tactile_quality=0.90,
        ),
        config=make_config(),
        method=FusionMethod.RELIABILITY_GATED,
    )

    assert decision.action is FusionAction.TOUCH_ONLY
    assert decision.odor_weight == 0.0
    assert decision.touch_weight == 1.0
    assert decision.fused_vector is not None
    assert decision.fused_vector[:16] == (0.0,) * 16


def test_two_unreliable_modalities_cause_abstention() -> None:
    """The system refuses unsupported low-reliability fusion."""

    decision = fuse_multisensory_view(
        view=make_view(
            olfactory_quality=0.20,
            tactile_quality=0.20,
        ),
        config=make_config(),
        method=FusionMethod.RELIABILITY_GATED,
    )

    assert decision.action is FusionAction.ABSTAIN
    assert decision.abstained is True
    assert decision.fused_vector is None
    assert decision.odor_weight == 0.0
    assert decision.touch_weight == 0.0


def test_missing_touch_receives_zero_weight() -> None:
    """An absent tactile vector can never influence the output."""

    view = make_view(
        condition=ConditionLabel.MISSING_TOUCH,
        tactile_vector=None,
        tactile_quality=0.0,
    )

    decision = fuse_multisensory_view(
        view=view,
        config=make_config(),
        method=FusionMethod.RELIABILITY_GATED,
    )

    assert decision.action is FusionAction.ODOR_ONLY
    assert decision.touch_reliability == 0.0
    assert decision.touch_weight == 0.0
    assert decision.fused_vector is not None
    assert decision.fused_vector[16:] == (0.0,) * 8


def test_missing_odor_receives_zero_weight() -> None:
    """An absent olfactory vector can never influence the output."""

    view = make_view(
        condition=ConditionLabel.MISSING_ODOR,
        olfactory_vector=None,
        olfactory_quality=0.0,
    )

    decision = fuse_multisensory_view(
        view=view,
        config=make_config(),
        method=FusionMethod.RELIABILITY_GATED,
    )

    assert decision.action is FusionAction.TOUCH_ONLY
    assert decision.odor_reliability == 0.0
    assert decision.odor_weight == 0.0
    assert decision.fused_vector is not None
    assert decision.fused_vector[:16] == (0.0,) * 16


def test_prespecified_conflict_causes_abstention() -> None:
    """Contradictory cross-family evidence is not fused confidently."""

    view = make_view(
        condition=ConditionLabel.CONTRADICTORY_MODALITIES,
        modality_conflict=True,
    )

    decision = fuse_multisensory_view(
        view=view,
        config=make_config(),
        method=FusionMethod.RELIABILITY_GATED,
    )

    assert decision.action is FusionAction.ABSTAIN
    assert decision.abstained is True
    assert decision.conflict_detected is True
    assert decision.fused_vector is None
    assert decision.odor_weight == 0.0
    assert decision.touch_weight == 0.0


def test_temporal_misalignment_causes_safe_abstention() -> None:
    """A locked nonzero temporal offset blocks confident fusion."""

    view = make_view(
        condition=ConditionLabel.TEMPORAL_MISALIGNMENT,
        temporal_offset_steps=2,
    )

    decision = fuse_multisensory_view(
        view=view,
        config=make_config(),
        method=FusionMethod.RELIABILITY_GATED,
    )

    assert decision.action is FusionAction.ABSTAIN
    assert decision.abstained is True
    assert decision.temporal_conflict_detected is True


def test_naive_concatenation_baseline_retains_raw_vectors() -> None:
    """Naive concatenation is an explicit non-gated comparison."""

    view = make_view(
        olfactory_quality=0.20,
        tactile_quality=0.20,
    )

    decision = fuse_multisensory_view(
        view=view,
        config=make_config(),
        method=FusionMethod.NAIVE_CONCATENATION,
    )

    assert decision.action is FusionAction.FUSED
    assert decision.fused_vector == ODOR + TOUCH
    assert decision.odor_weight == 1.0
    assert decision.touch_weight == 1.0
    assert decision.abstained is False


def test_fixed_equal_baseline_uses_half_weights() -> None:
    """The fixed baseline assigns 0.5/0.5 when both channels exist."""

    decision = fuse_multisensory_view(
        view=make_view(
            olfactory_quality=0.20,
            tactile_quality=0.90,
        ),
        config=make_config(),
        method=FusionMethod.FIXED_EQUAL,
    )

    assert decision.action is FusionAction.FUSED
    assert decision.odor_weight == 0.5
    assert decision.touch_weight == 0.5
    assert decision.fused_vector is not None
    assert len(decision.fused_vector) == 24


@pytest.mark.parametrize(
    "method",
    (
        FusionMethod.NAIVE_CONCATENATION,
        FusionMethod.FIXED_EQUAL,
    ),
)
def test_baselines_still_zero_missing_modality(
    method: FusionMethod,
) -> None:
    """Even non-gated baselines cannot fabricate unavailable evidence."""

    view = make_view(
        condition=ConditionLabel.MISSING_TOUCH,
        tactile_vector=None,
        tactile_quality=0.0,
    )

    decision = fuse_multisensory_view(
        view=view,
        config=make_config(),
        method=method,
    )

    assert decision.action is FusionAction.ODOR_ONLY
    assert decision.touch_weight == 0.0
    assert decision.fused_vector is not None
    assert decision.fused_vector[16:] == (0.0,) * 8


def test_decision_trace_is_complete_and_auditable() -> None:
    """Every proposed-method action explains its inputs and outcome."""

    view = make_view(
        olfactory_quality=0.80,
        tactile_quality=0.60,
    )

    decision = fuse_multisensory_view(
        view=view,
        config=make_config(),
        method=FusionMethod.RELIABILITY_GATED,
    )
    trace = decision.trace

    assert isinstance(trace, FusionTrace)
    assert trace.view_id == view.view_id
    assert trace.latent_event_id == view.latent_event_id
    assert trace.method is FusionMethod.RELIABILITY_GATED
    assert trace.odor_available is True
    assert trace.touch_available is True
    assert trace.odor_reliability == 0.80
    assert trace.touch_reliability == 0.60
    assert trace.conflict_detected is False
    assert trace.temporal_conflict_detected is False
    assert trace.minimum_reliability == 0.30
    assert trace.selected_action is FusionAction.FUSED
    assert trace.reason


def test_output_vector_is_finite() -> None:
    """Every non-abstaining output contains finite values."""

    for method in FusionMethod:
        decision = fuse_multisensory_view(
            view=make_view(),
            config=make_config(),
            method=method,
        )

        assert decision.fused_vector is not None
        assert all(
            math.isfinite(value)
            for value in decision.fused_vector
        )


def test_ground_truth_identifiers_are_preserved() -> None:
    """Fusion cannot redefine the latent event or target identity."""

    view = make_view()

    decision = fuse_multisensory_view(
        view=view,
        config=make_config(),
        method=FusionMethod.RELIABILITY_GATED,
    )

    assert decision.view_id == view.view_id
    assert decision.latent_event_id == view.latent_event_id
    assert decision.target_item_id == view.target_item_id
    assert decision.target_family_id == view.target_family_id


@pytest.mark.parametrize(
    "value",
    (
        0.0,
        -0.1,
        1.1,
        float("nan"),
        True,
    ),
)
def test_invalid_minimum_reliability_is_rejected(
    value: object,
) -> None:
    """The reliability threshold must be finite and inside (0, 1]."""

    with pytest.raises(
        FusionError,
        match="minimum_reliability",
    ):
        make_config(minimum_reliability=value)


def test_generator_version_is_required() -> None:
    """Every fusion decision carries versioned policy provenance."""

    with pytest.raises(
        FusionError,
        match="generator_version",
    ):
        make_config(generator_version=" ")


def test_unknown_method_is_rejected() -> None:
    """Only preregistered fusion methods are accepted."""

    with pytest.raises(
        FusionError,
        match="FusionMethod",
    ):
        fuse_multisensory_view(
            view=make_view(),
            config=make_config(),
            method="invented",  # type: ignore[arg-type]
        )


def test_wrong_input_type_is_rejected() -> None:
    """Fusion requires a validated condition-view record."""

    with pytest.raises(
        FusionError,
        match="MultisensoryConditionView",
    ):
        fuse_multisensory_view(
            view=object(),  # type: ignore[arg-type]
            config=make_config(),
            method=FusionMethod.RELIABILITY_GATED,
        )


def test_configuration_revalidates_after_replace() -> None:
    """Dataclass replacement cannot bypass policy validation."""

    config = make_config()

    with pytest.raises(FusionError):
        replace(
            config,
            minimum_reliability=0.0,
        )
