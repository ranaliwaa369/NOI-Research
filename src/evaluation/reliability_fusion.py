"""Reliability- and conflict-gated fusion for NOI v0.3.

The proposed method uses modality availability and recorded evidence quality
to select odor-only, touch-only, fused, or abstain behavior. Two explicit
non-gated baselines are retained for controlled comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

from src.evaluation.multisensory_records import (
    MultisensoryConditionView,
)


OLFACTORY_DIMENSION = 16
TACTILE_DIMENSION = 8
FUSED_DIMENSION = OLFACTORY_DIMENSION + TACTILE_DIMENSION


class FusionError(ValueError):
    """Raised when a fusion input or policy is invalid."""


class FusionMethod(str, Enum):
    """Proposed fusion method and two preregistered baselines."""

    RELIABILITY_GATED = "reliability_gated"
    NAIVE_CONCATENATION = "naive_concatenation"
    FIXED_EQUAL = "fixed_equal"


class FusionAction(str, Enum):
    """Operational action selected for one paired condition view."""

    ODOR_ONLY = "odor_only"
    TOUCH_ONLY = "touch_only"
    FUSED = "fused"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Locked reliability policy used by the proposed method."""

    minimum_reliability: float
    generator_version: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_reliability, bool)
            or not isinstance(self.minimum_reliability, Real)
            or not math.isfinite(float(self.minimum_reliability))
            or not 0.0 < float(self.minimum_reliability) <= 1.0
        ):
            raise FusionError(
                "minimum_reliability must be finite and between "
                "0 and 1."
            )

        if (
            not isinstance(self.generator_version, str)
            or not self.generator_version.strip()
        ):
            raise FusionError(
                "generator_version must be a nonempty string."
            )


@dataclass(frozen=True, slots=True)
class FusionTrace:
    """Auditable explanation of one fusion action."""

    view_id: str
    latent_event_id: str
    method: FusionMethod
    odor_available: bool
    touch_available: bool
    odor_reliability: float
    touch_reliability: float
    conflict_detected: bool
    temporal_conflict_detected: bool
    minimum_reliability: float
    selected_action: FusionAction
    odor_weight: float
    touch_weight: float
    reason: str
    generator_version: str


@dataclass(frozen=True, slots=True)
class FusionDecision:
    """One immutable fusion output and its decision provenance."""

    view_id: str
    latent_event_id: str
    target_item_id: str
    target_family_id: int
    method: FusionMethod
    action: FusionAction
    odor_reliability: float
    touch_reliability: float
    odor_weight: float
    touch_weight: float
    conflict_detected: bool
    temporal_conflict_detected: bool
    abstained: bool
    fused_vector: tuple[float, ...] | None
    trace: FusionTrace


def _scaled_or_zero(
    vector: tuple[float, ...] | None,
    *,
    dimension: int,
    weight: float,
) -> tuple[float, ...]:
    """Return a weighted vector or fixed-dimensional zeros."""

    if vector is None:
        return (0.0,) * dimension

    if len(vector) != dimension:
        raise FusionError(
            "Modality vector dimension violates the fusion contract."
        )

    values = tuple(
        float(weight) * float(value)
        for value in vector
    )

    if not all(math.isfinite(value) for value in values):
        raise FusionError(
            "Fusion output must contain only finite values."
        )

    return values


def _build_vector(
    view: MultisensoryConditionView,
    *,
    odor_weight: float,
    touch_weight: float,
) -> tuple[float, ...]:
    """Construct one fixed 24-dimensional weighted representation."""

    odor_part = _scaled_or_zero(
        view.olfactory_vector,
        dimension=OLFACTORY_DIMENSION,
        weight=odor_weight,
    )
    touch_part = _scaled_or_zero(
        view.tactile_vector,
        dimension=TACTILE_DIMENSION,
        weight=touch_weight,
    )

    output = odor_part + touch_part

    if len(output) != FUSED_DIMENSION:
        raise FusionError(
            "Fused output must contain exactly 24 values."
        )

    return output


def _make_decision(
    *,
    view: MultisensoryConditionView,
    config: FusionConfig,
    method: FusionMethod,
    action: FusionAction,
    odor_reliability: float,
    touch_reliability: float,
    odor_weight: float,
    touch_weight: float,
    conflict_detected: bool,
    temporal_conflict_detected: bool,
    reason: str,
) -> FusionDecision:
    """Build one immutable decision and its matching trace."""

    abstained = action is FusionAction.ABSTAIN

    fused_vector = (
        None
        if abstained
        else _build_vector(
            view,
            odor_weight=odor_weight,
            touch_weight=touch_weight,
        )
    )

    trace = FusionTrace(
        view_id=view.view_id,
        latent_event_id=view.latent_event_id,
        method=method,
        odor_available=view.olfactory_available,
        touch_available=view.tactile_available,
        odor_reliability=odor_reliability,
        touch_reliability=touch_reliability,
        conflict_detected=conflict_detected,
        temporal_conflict_detected=temporal_conflict_detected,
        minimum_reliability=float(
            config.minimum_reliability,
        ),
        selected_action=action,
        odor_weight=odor_weight,
        touch_weight=touch_weight,
        reason=reason,
        generator_version=config.generator_version,
    )

    return FusionDecision(
        view_id=view.view_id,
        latent_event_id=view.latent_event_id,
        target_item_id=view.target_item_id,
        target_family_id=view.target_family_id,
        method=method,
        action=action,
        odor_reliability=odor_reliability,
        touch_reliability=touch_reliability,
        odor_weight=odor_weight,
        touch_weight=touch_weight,
        conflict_detected=conflict_detected,
        temporal_conflict_detected=temporal_conflict_detected,
        abstained=abstained,
        fused_vector=fused_vector,
        trace=trace,
    )


def _reliability_gated_decision(
    *,
    view: MultisensoryConditionView,
    config: FusionConfig,
) -> FusionDecision:
    """Apply availability, reliability, conflict, and temporal gates."""

    odor_reliability = (
        float(view.olfactory_quality)
        if view.olfactory_available
        else 0.0
    )
    touch_reliability = (
        float(view.tactile_quality)
        if view.tactile_available
        else 0.0
    )

    conflict_detected = bool(view.modality_conflict)
    temporal_conflict_detected = (
        view.temporal_offset_steps != 0
    )

    if conflict_detected:
        return _make_decision(
            view=view,
            config=config,
            method=FusionMethod.RELIABILITY_GATED,
            action=FusionAction.ABSTAIN,
            odor_reliability=odor_reliability,
            touch_reliability=touch_reliability,
            odor_weight=0.0,
            touch_weight=0.0,
            conflict_detected=True,
            temporal_conflict_detected=(
                temporal_conflict_detected
            ),
            reason=(
                "Prespecified cross-family modality conflict "
                "requires abstention."
            ),
        )

    if temporal_conflict_detected:
        return _make_decision(
            view=view,
            config=config,
            method=FusionMethod.RELIABILITY_GATED,
            action=FusionAction.ABSTAIN,
            odor_reliability=odor_reliability,
            touch_reliability=touch_reliability,
            odor_weight=0.0,
            touch_weight=0.0,
            conflict_detected=False,
            temporal_conflict_detected=True,
            reason=(
                "Nonzero temporal offset requires safe abstention."
            ),
        )

    odor_reliable = (
        view.olfactory_available
        and odor_reliability >= config.minimum_reliability
    )
    touch_reliable = (
        view.tactile_available
        and touch_reliability >= config.minimum_reliability
    )

    if odor_reliable and touch_reliable:
        total = odor_reliability + touch_reliability
        odor_weight = odor_reliability / total
        touch_weight = touch_reliability / total

        return _make_decision(
            view=view,
            config=config,
            method=FusionMethod.RELIABILITY_GATED,
            action=FusionAction.FUSED,
            odor_reliability=odor_reliability,
            touch_reliability=touch_reliability,
            odor_weight=odor_weight,
            touch_weight=touch_weight,
            conflict_detected=False,
            temporal_conflict_detected=False,
            reason=(
                "Both available modalities satisfy the minimum "
                "reliability policy."
            ),
        )

    if odor_reliable:
        return _make_decision(
            view=view,
            config=config,
            method=FusionMethod.RELIABILITY_GATED,
            action=FusionAction.ODOR_ONLY,
            odor_reliability=odor_reliability,
            touch_reliability=touch_reliability,
            odor_weight=1.0,
            touch_weight=0.0,
            conflict_detected=False,
            temporal_conflict_detected=False,
            reason=(
                "Only olfactory evidence satisfies the minimum "
                "reliability policy."
            ),
        )

    if touch_reliable:
        return _make_decision(
            view=view,
            config=config,
            method=FusionMethod.RELIABILITY_GATED,
            action=FusionAction.TOUCH_ONLY,
            odor_reliability=odor_reliability,
            touch_reliability=touch_reliability,
            odor_weight=0.0,
            touch_weight=1.0,
            conflict_detected=False,
            temporal_conflict_detected=False,
            reason=(
                "Only tactile evidence satisfies the minimum "
                "reliability policy."
            ),
        )

    return _make_decision(
        view=view,
        config=config,
        method=FusionMethod.RELIABILITY_GATED,
        action=FusionAction.ABSTAIN,
        odor_reliability=odor_reliability,
        touch_reliability=touch_reliability,
        odor_weight=0.0,
        touch_weight=0.0,
        conflict_detected=False,
        temporal_conflict_detected=False,
        reason=(
            "No available modality satisfies the minimum "
            "reliability policy."
        ),
    )


def _baseline_decision(
    *,
    view: MultisensoryConditionView,
    config: FusionConfig,
    method: FusionMethod,
) -> FusionDecision:
    """Apply one explicit non-gated comparison baseline."""

    odor_available = view.olfactory_available
    touch_available = view.tactile_available
    odor_reliability = (
        float(view.olfactory_quality)
        if odor_available
        else 0.0
    )
    touch_reliability = (
        float(view.tactile_quality)
        if touch_available
        else 0.0
    )

    if odor_available and touch_available:
        action = FusionAction.FUSED

        if method is FusionMethod.NAIVE_CONCATENATION:
            odor_weight = 1.0
            touch_weight = 1.0
            reason = (
                "Naive baseline concatenates both available "
                "modalities without reliability gating."
            )
        else:
            odor_weight = 0.5
            touch_weight = 0.5
            reason = (
                "Fixed baseline assigns equal weights to both "
                "available modalities."
            )

    elif odor_available:
        action = FusionAction.ODOR_ONLY
        odor_weight = 1.0
        touch_weight = 0.0
        reason = (
            "Only olfactory evidence is available to the baseline."
        )

    elif touch_available:
        action = FusionAction.TOUCH_ONLY
        odor_weight = 0.0
        touch_weight = 1.0
        reason = (
            "Only tactile evidence is available to the baseline."
        )

    else:
        action = FusionAction.ABSTAIN
        odor_weight = 0.0
        touch_weight = 0.0
        reason = "No modality is available to the baseline."

    return _make_decision(
        view=view,
        config=config,
        method=method,
        action=action,
        odor_reliability=odor_reliability,
        touch_reliability=touch_reliability,
        odor_weight=odor_weight,
        touch_weight=touch_weight,
        conflict_detected=bool(view.modality_conflict),
        temporal_conflict_detected=(
            view.temporal_offset_steps != 0
        ),
        reason=reason,
    )


def fuse_multisensory_view(
    *,
    view: MultisensoryConditionView,
    config: FusionConfig,
    method: FusionMethod,
) -> FusionDecision:
    """Fuse one paired view under the selected registered method."""

    if not isinstance(view, MultisensoryConditionView):
        raise FusionError(
            "view must be a MultisensoryConditionView record."
        )

    if not isinstance(config, FusionConfig):
        raise FusionError(
            "config must be a FusionConfig record."
        )

    if not isinstance(method, FusionMethod):
        raise FusionError(
            "method must be a FusionMethod value."
        )

    if method is FusionMethod.RELIABILITY_GATED:
        return _reliability_gated_decision(
            view=view,
            config=config,
        )

    return _baseline_decision(
        view=view,
        config=config,
        method=method,
    )

from src.evaluation.evidence_conflict import EvidenceAssessment


@dataclass(frozen=True, slots=True)
class LockedFusionConfig:
    """Validation-locked metadata-blind fusion thresholds."""

    reliability_threshold: float
    conflict_threshold: float
    generator_version: str

    def __post_init__(self) -> None:
        _validate_locked_probability(
            "reliability_threshold",
            self.reliability_threshold,
        )
        _validate_locked_probability(
            "conflict_threshold",
            self.conflict_threshold,
        )

        if (
            not isinstance(self.generator_version, str)
            or not self.generator_version.strip()
        ):
            raise FusionError(
                "generator_version must be a nonempty string."
            )


@dataclass(frozen=True, slots=True)
class LockedFusionTrace:
    """Auditable trace with no condition or target-label inputs."""

    event_id: str
    odor_available: bool
    touch_available: bool
    odor_reliability: float
    touch_reliability: float
    reliability_threshold: float
    conflict_available: bool
    conflict_score: float
    conflict_threshold: float
    conflict_detected: bool
    temporal_offset_steps: int
    temporal_conflict_detected: bool
    selected_action: FusionAction
    odor_weight: float
    touch_weight: float
    reason: str
    generator_version: str
    condition_metadata_used: bool
    target_labels_used: bool
    final_test_labels_used: bool


@dataclass(frozen=True, slots=True)
class LockedFusionDecision:
    """One validation-locked evidence-only fusion decision."""

    event_id: str
    action: FusionAction
    odor_reliability: float
    touch_reliability: float
    odor_weight: float
    touch_weight: float
    conflict_score: float
    conflict_detected: bool
    temporal_conflict_detected: bool
    abstained: bool
    fused_vector: tuple[float, ...] | None
    trace: LockedFusionTrace


def _validate_locked_probability(
    name: str,
    value: object,
) -> float:
    """Require one finite locked probability in [0, 1]."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise FusionError(
            f"{name} must be finite and between 0 and 1."
        )

    return float(value)


def _locked_vector(
    vector: tuple[float, ...] | None,
    *,
    dimension: int,
    weight: float,
    label: str,
) -> tuple[float, ...]:
    """Return one validated weighted modality vector."""

    if vector is None:
        if weight != 0.0:
            raise FusionError(
                f"Unavailable {label} cannot receive nonzero weight."
            )
        return (0.0,) * dimension

    if (
        not isinstance(vector, tuple)
        or len(vector) != dimension
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in vector
        )
    ):
        raise FusionError(
            f"{label} must be a finite tuple with dimension "
            f"{dimension}."
        )

    return tuple(
        float(value) * weight
        for value in vector
    )


def _make_locked_decision(
    *,
    event_id: str,
    olfactory_vector: tuple[float, ...] | None,
    tactile_vector: tuple[float, ...] | None,
    temporal_offset_steps: int,
    evidence: EvidenceAssessment,
    config: LockedFusionConfig,
    action: FusionAction,
    odor_weight: float,
    touch_weight: float,
    conflict_detected: bool,
    temporal_conflict_detected: bool,
    reason: str,
) -> LockedFusionDecision:
    """Build one immutable metadata-blind locked decision."""

    abstained = action is FusionAction.ABSTAIN

    if abstained:
        fused_vector = None
    else:
        odor_part = _locked_vector(
            olfactory_vector,
            dimension=16,
            weight=odor_weight,
            label="olfactory_vector",
        )
        touch_part = _locked_vector(
            tactile_vector,
            dimension=8,
            weight=touch_weight,
            label="tactile_vector",
        )
        fused_vector = odor_part + touch_part

    trace = LockedFusionTrace(
        event_id=event_id,
        odor_available=evidence.odor_available,
        touch_available=evidence.touch_available,
        odor_reliability=float(
            evidence.odor_reliability
        ),
        touch_reliability=float(
            evidence.touch_reliability
        ),
        reliability_threshold=float(
            config.reliability_threshold
        ),
        conflict_available=evidence.conflict_available,
        conflict_score=float(evidence.conflict_score),
        conflict_threshold=float(
            config.conflict_threshold
        ),
        conflict_detected=conflict_detected,
        temporal_offset_steps=temporal_offset_steps,
        temporal_conflict_detected=(
            temporal_conflict_detected
        ),
        selected_action=action,
        odor_weight=odor_weight,
        touch_weight=touch_weight,
        reason=reason,
        generator_version=config.generator_version,
        condition_metadata_used=False,
        target_labels_used=False,
        final_test_labels_used=False,
    )

    return LockedFusionDecision(
        event_id=event_id,
        action=action,
        odor_reliability=float(
            evidence.odor_reliability
        ),
        touch_reliability=float(
            evidence.touch_reliability
        ),
        odor_weight=odor_weight,
        touch_weight=touch_weight,
        conflict_score=float(evidence.conflict_score),
        conflict_detected=conflict_detected,
        temporal_conflict_detected=(
            temporal_conflict_detected
        ),
        abstained=abstained,
        fused_vector=fused_vector,
        trace=trace,
    )


def fuse_locked_evidence(
    *,
    event_id: str,
    olfactory_vector: tuple[float, ...] | None,
    tactile_vector: tuple[float, ...] | None,
    temporal_offset_steps: int,
    evidence: EvidenceAssessment,
    config: LockedFusionConfig,
) -> LockedFusionDecision:
    """Fuse evidence without condition metadata or target labels."""

    if not isinstance(event_id, str) or not event_id.strip():
        raise FusionError(
            "event_id must be a nonempty string."
        )

    if (
        isinstance(temporal_offset_steps, bool)
        or not isinstance(temporal_offset_steps, int)
    ):
        raise FusionError(
            "temporal_offset_steps must be an integer."
        )

    if not isinstance(evidence, EvidenceAssessment):
        raise FusionError(
            "evidence must be an EvidenceAssessment record."
        )

    if not isinstance(config, LockedFusionConfig):
        raise FusionError(
            "config must be a LockedFusionConfig record."
        )

    odor_available = olfactory_vector is not None
    touch_available = tactile_vector is not None

    if (
        odor_available != evidence.odor_available
        or touch_available != evidence.touch_available
    ):
        raise FusionError(
            "Evidence availability must match supplied vectors."
        )

    if not odor_available and not touch_available:
        raise FusionError(
            "At least one modality must be available."
        )

    odor_reliability = _validate_locked_probability(
        "odor_reliability",
        evidence.odor_reliability,
    )
    touch_reliability = _validate_locked_probability(
        "touch_reliability",
        evidence.touch_reliability,
    )
    conflict_score = _validate_locked_probability(
        "conflict_score",
        evidence.conflict_score,
    )

    if not odor_available and odor_reliability != 0.0:
        raise FusionError(
            "Unavailable odor must have zero reliability."
        )

    if not touch_available and touch_reliability != 0.0:
        raise FusionError(
            "Unavailable touch must have zero reliability."
        )

    temporal_conflict = temporal_offset_steps != 0
    conflict_detected = (
        evidence.conflict_available
        and conflict_score >= config.conflict_threshold
    )

    if temporal_conflict:
        return _make_locked_decision(
            event_id=event_id,
            olfactory_vector=olfactory_vector,
            tactile_vector=tactile_vector,
            temporal_offset_steps=temporal_offset_steps,
            evidence=evidence,
            config=config,
            action=FusionAction.ABSTAIN,
            odor_weight=0.0,
            touch_weight=0.0,
            conflict_detected=conflict_detected,
            temporal_conflict_detected=True,
            reason=(
                "Nonzero temporal offset requires safe abstention."
            ),
        )

    if conflict_detected:
        return _make_locked_decision(
            event_id=event_id,
            olfactory_vector=olfactory_vector,
            tactile_vector=tactile_vector,
            temporal_offset_steps=temporal_offset_steps,
            evidence=evidence,
            config=config,
            action=FusionAction.ABSTAIN,
            odor_weight=0.0,
            touch_weight=0.0,
            conflict_detected=True,
            temporal_conflict_detected=False,
            reason=(
                "Evidence-derived conflict score meets the "
                "validation-locked threshold."
            ),
        )

    odor_reliable = (
        odor_available
        and odor_reliability
        >= config.reliability_threshold
    )
    touch_reliable = (
        touch_available
        and touch_reliability
        >= config.reliability_threshold
    )

    if odor_reliable and touch_reliable:
        total = odor_reliability + touch_reliability
        odor_weight = odor_reliability / total
        touch_weight = touch_reliability / total

        return _make_locked_decision(
            event_id=event_id,
            olfactory_vector=olfactory_vector,
            tactile_vector=tactile_vector,
            temporal_offset_steps=temporal_offset_steps,
            evidence=evidence,
            config=config,
            action=FusionAction.FUSED,
            odor_weight=odor_weight,
            touch_weight=touch_weight,
            conflict_detected=False,
            temporal_conflict_detected=False,
            reason=(
                "Both modalities satisfy the validation-locked "
                "reliability threshold."
            ),
        )

    if odor_reliable:
        return _make_locked_decision(
            event_id=event_id,
            olfactory_vector=olfactory_vector,
            tactile_vector=tactile_vector,
            temporal_offset_steps=temporal_offset_steps,
            evidence=evidence,
            config=config,
            action=FusionAction.ODOR_ONLY,
            odor_weight=1.0,
            touch_weight=0.0,
            conflict_detected=False,
            temporal_conflict_detected=False,
            reason=(
                "Only olfactory evidence satisfies the "
                "validation-locked reliability threshold."
            ),
        )

    if touch_reliable:
        return _make_locked_decision(
            event_id=event_id,
            olfactory_vector=olfactory_vector,
            tactile_vector=tactile_vector,
            temporal_offset_steps=temporal_offset_steps,
            evidence=evidence,
            config=config,
            action=FusionAction.TOUCH_ONLY,
            odor_weight=0.0,
            touch_weight=1.0,
            conflict_detected=False,
            temporal_conflict_detected=False,
            reason=(
                "Only tactile evidence satisfies the "
                "validation-locked reliability threshold."
            ),
        )

    return _make_locked_decision(
        event_id=event_id,
        olfactory_vector=olfactory_vector,
        tactile_vector=tactile_vector,
        temporal_offset_steps=temporal_offset_steps,
        evidence=evidence,
        config=config,
        action=FusionAction.ABSTAIN,
        odor_weight=0.0,
        touch_weight=0.0,
        conflict_detected=False,
        temporal_conflict_detected=False,
        reason=(
            "No modality satisfies the validation-locked "
            "reliability threshold."
        ),
    )
