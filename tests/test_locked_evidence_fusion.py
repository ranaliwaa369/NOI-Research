"""Tests for metadata-blind validation-locked evidence fusion."""

from __future__ import annotations

from dataclasses import replace
import inspect
import math

import pytest

from src.evaluation.evidence_conflict import (
    EvidenceAssessment,
)
from src.evaluation.reliability_fusion import (
    FusionAction,
    FusionError,
    LockedFusionConfig,
    fuse_locked_evidence,
)


def assessment(
    *,
    odor_reliability: float = 0.8,
    touch_reliability: float = 0.7,
    conflict_score: float = 0.1,
    odor_available: bool = True,
    touch_available: bool = True,
) -> EvidenceAssessment:
    """Return one evidence-only assessment."""

    odor_distribution = (
        (0.7, 0.2, 0.1)
        if odor_available
        else ()
    )
    touch_distribution = (
        (0.65, 0.25, 0.10)
        if touch_available
        else ()
    )

    return EvidenceAssessment(
        family_ids=(0, 1, 2),
        odor_available=odor_available,
        touch_available=touch_available,
        odor_family_distribution=odor_distribution,
        touch_family_distribution=touch_distribution,
        odor_reliability=(
            odor_reliability if odor_available else 0.0
        ),
        touch_reliability=(
            touch_reliability if touch_available else 0.0
        ),
        conflict_available=(
            odor_available and touch_available
        ),
        conflict_score=(
            conflict_score
            if odor_available and touch_available
            else 0.0
        ),
    )


def config() -> LockedFusionConfig:
    """Return one validation-locked fusion policy."""

    return LockedFusionConfig(
        reliability_threshold=0.5,
        conflict_threshold=0.6,
        generator_version="0.3.1-lock-test",
    )


def test_public_interface_contains_no_condition_answers() -> None:
    """Locked fusion cannot receive a view, label, target, or conflict flag."""

    signature = inspect.signature(fuse_locked_evidence)

    assert set(signature.parameters) == {
        "event_id",
        "olfactory_vector",
        "tactile_vector",
        "temporal_offset_steps",
        "evidence",
        "config",
    }


def test_two_reliable_modalities_are_fused() -> None:
    """Reliable agreeing evidence receives normalized weights."""

    decision = fuse_locked_evidence(
        event_id="event-1",
        olfactory_vector=(1.0,) * 16,
        tactile_vector=(1.0,) * 8,
        temporal_offset_steps=0,
        evidence=assessment(),
        config=config(),
    )

    assert decision.action is FusionAction.FUSED
    assert decision.abstained is False
    assert decision.odor_weight + decision.touch_weight == pytest.approx(
        1.0
    )
    assert len(decision.fused_vector or ()) == 24


def test_evidence_conflict_causes_abstention() -> None:
    """Only the numeric evidence score may trigger modality conflict."""

    decision = fuse_locked_evidence(
        event_id="event-conflict",
        olfactory_vector=(1.0,) * 16,
        tactile_vector=(1.0,) * 8,
        temporal_offset_steps=0,
        evidence=assessment(conflict_score=0.8),
        config=config(),
    )

    assert decision.action is FusionAction.ABSTAIN
    assert decision.abstained is True
    assert decision.conflict_detected is True
    assert decision.fused_vector is None


def test_low_conflict_score_does_not_trigger_abstention() -> None:
    """A score below the validation lock cannot be relabeled conflict."""

    decision = fuse_locked_evidence(
        event_id="event-agree",
        olfactory_vector=(1.0,) * 16,
        tactile_vector=(1.0,) * 8,
        temporal_offset_steps=0,
        evidence=assessment(conflict_score=0.59),
        config=config(),
    )

    assert decision.conflict_detected is False
    assert decision.action is FusionAction.FUSED


def test_only_reliable_odor_selects_odor() -> None:
    """Touch below the locked reliability threshold receives zero weight."""

    decision = fuse_locked_evidence(
        event_id="event-odor",
        olfactory_vector=(1.0,) * 16,
        tactile_vector=(1.0,) * 8,
        temporal_offset_steps=0,
        evidence=assessment(
            odor_reliability=0.8,
            touch_reliability=0.3,
        ),
        config=config(),
    )

    assert decision.action is FusionAction.ODOR_ONLY
    assert decision.odor_weight == 1.0
    assert decision.touch_weight == 0.0


def test_only_reliable_touch_selects_touch() -> None:
    """Odor below the locked reliability threshold receives zero weight."""

    decision = fuse_locked_evidence(
        event_id="event-touch",
        olfactory_vector=(1.0,) * 16,
        tactile_vector=(1.0,) * 8,
        temporal_offset_steps=0,
        evidence=assessment(
            odor_reliability=0.3,
            touch_reliability=0.8,
        ),
        config=config(),
    )

    assert decision.action is FusionAction.TOUCH_ONLY
    assert decision.odor_weight == 0.0
    assert decision.touch_weight == 1.0


def test_no_reliable_modality_causes_abstention() -> None:
    """Insufficient evidence cannot produce a confident retrieval vector."""

    decision = fuse_locked_evidence(
        event_id="event-weak",
        olfactory_vector=(1.0,) * 16,
        tactile_vector=(1.0,) * 8,
        temporal_offset_steps=0,
        evidence=assessment(
            odor_reliability=0.3,
            touch_reliability=0.2,
        ),
        config=config(),
    )

    assert decision.action is FusionAction.ABSTAIN
    assert decision.fused_vector is None


def test_missing_touch_receives_zero_weight() -> None:
    """Unavailable touch is excluded even if the caller supplies no vector."""

    decision = fuse_locked_evidence(
        event_id="event-missing-touch",
        olfactory_vector=(1.0,) * 16,
        tactile_vector=None,
        temporal_offset_steps=0,
        evidence=assessment(
            touch_available=False,
        ),
        config=config(),
    )

    assert decision.action is FusionAction.ODOR_ONLY
    assert decision.touch_weight == 0.0


def test_missing_odor_receives_zero_weight() -> None:
    """Unavailable odor is excluded even if touch remains reliable."""

    decision = fuse_locked_evidence(
        event_id="event-missing-odor",
        olfactory_vector=None,
        tactile_vector=(1.0,) * 8,
        temporal_offset_steps=0,
        evidence=assessment(
            odor_available=False,
        ),
        config=config(),
    )

    assert decision.action is FusionAction.TOUCH_ONLY
    assert decision.odor_weight == 0.0


def test_availability_mismatch_is_rejected() -> None:
    """Evidence availability must match the actual supplied vectors."""

    with pytest.raises(
        FusionError,
        match="availability",
    ):
        fuse_locked_evidence(
            event_id="event-invalid",
            olfactory_vector=(1.0,) * 16,
            tactile_vector=None,
            temporal_offset_steps=0,
            evidence=assessment(),
            config=config(),
        )


def test_temporal_offset_causes_safe_abstention() -> None:
    """The locked temporal mismatch remains a direct safety control."""

    decision = fuse_locked_evidence(
        event_id="event-temporal",
        olfactory_vector=(1.0,) * 16,
        tactile_vector=(1.0,) * 8,
        temporal_offset_steps=3,
        evidence=assessment(),
        config=config(),
    )

    assert decision.action is FusionAction.ABSTAIN
    assert decision.temporal_conflict_detected is True


def test_trace_records_numeric_scores_and_thresholds() -> None:
    """Every locked decision must retain auditable evidence provenance."""

    decision = fuse_locked_evidence(
        event_id="event-trace",
        olfactory_vector=(1.0,) * 16,
        tactile_vector=(1.0,) * 8,
        temporal_offset_steps=0,
        evidence=assessment(),
        config=config(),
    )

    assert decision.trace.conflict_score == pytest.approx(0.1)
    assert decision.trace.conflict_threshold == pytest.approx(0.6)
    assert decision.trace.reliability_threshold == pytest.approx(0.5)
    assert decision.trace.condition_metadata_used is False
    assert decision.trace.target_labels_used is False
    assert decision.trace.final_test_labels_used is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reliability_threshold", -0.1),
        ("reliability_threshold", 1.1),
        ("conflict_threshold", -0.1),
        ("conflict_threshold", 1.1),
    ),
)
def test_invalid_locked_threshold_is_rejected(
    field: str,
    value: float,
) -> None:
    """Both validation-locked thresholds must remain in [0, 1]."""

    with pytest.raises(
        FusionError,
        match="between 0 and 1",
    ):
        replace(config(), **{field: value})


def test_locked_output_is_finite() -> None:
    """A non-abstaining locked vector contains only finite values."""

    decision = fuse_locked_evidence(
        event_id="event-finite",
        olfactory_vector=(1.0,) * 16,
        tactile_vector=(1.0,) * 8,
        temporal_offset_steps=0,
        evidence=assessment(),
        config=config(),
    )

    assert decision.fused_vector is not None
    assert all(
        math.isfinite(value)
        for value in decision.fused_vector
    )
