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
