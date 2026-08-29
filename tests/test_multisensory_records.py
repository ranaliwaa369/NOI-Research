"""Tests for immutable NOI v0.3 multisensory records."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.evaluation.multisensory_records import (
    ConditionLabel,
    MultisensoryConditionView,
    MultisensoryDataset,
    MultisensoryRecordError,
    MultisensorySplit,
    MultisensoryTarget,
    SupportRegime,
    LatentMultisensoryEvent,
)


def make_target(
    *,
    item_id: str = "item-001",
    family_id: int = 1,
) -> MultisensoryTarget:
    """Return one valid multisensory target."""

    return MultisensoryTarget(
        item_id=item_id,
        family_id=family_id,
        olfactory_prototype=(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        tactile_prototype=(
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ),
    )


def make_event(
    *,
    latent_event_id: str = "latent-001",
    split: MultisensorySplit = MultisensorySplit.FINAL_TEST,
    target_item_id: str = "item-001",
    target_family_id: int = 1,
    support_regime: SupportRegime = SupportRegime.SEEN_ITEM,
) -> LatentMultisensoryEvent:
    """Return one valid clean latent event."""

    target = make_target(
        item_id=target_item_id,
        family_id=target_family_id,
    )

    return LatentMultisensoryEvent(
        latent_event_id=latent_event_id,
        split=split,
        template_id=2,
        target_item_id=target_item_id,
        target_family_id=target_family_id,
        support_regime=support_regime,
        olfactory_vector=target.olfactory_prototype,
        tactile_vector=target.tactile_prototype,
        generator_seed=1301,
    )


def make_view(
    *,
    view_id: str = "latent-001-clean",
    latent_event_id: str = "latent-001",
    condition: ConditionLabel = ConditionLabel.CLEAN,
    target_item_id: str = "item-001",
    target_family_id: int = 1,
) -> MultisensoryConditionView:
    """Return one valid clean condition view."""

    target = make_target(
        item_id=target_item_id,
        family_id=target_family_id,
    )

    return MultisensoryConditionView(
        view_id=view_id,
        latent_event_id=latent_event_id,
        condition=condition,
        target_item_id=target_item_id,
        target_family_id=target_family_id,
        olfactory_vector=target.olfactory_prototype,
        tactile_vector=target.tactile_prototype,
        olfactory_quality=1.0,
        tactile_quality=1.0,
        modality_conflict=False,
        temporal_offset_steps=0,
    )


def make_dataset(
    *,
    targets: tuple[MultisensoryTarget, ...] | None = None,
    events: tuple[LatentMultisensoryEvent, ...] | None = None,
    views: tuple[MultisensoryConditionView, ...] | None = None,
) -> MultisensoryDataset:
    """Return one minimal valid multisensory dataset."""

    return MultisensoryDataset(
        targets=targets or (make_target(),),
        latent_events=events or (make_event(),),
        condition_views=views or (make_view(),),
        generator_version="0.3.0-feasibility",
        generator_seed=1301,
    )


def test_valid_target_has_locked_dimensions() -> None:
    """A target contains sixteen odor and eight touch values."""

    target = make_target()

    assert len(target.olfactory_prototype) == 16
    assert len(target.tactile_prototype) == 8


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "olfactory_prototype",
            (1.0,) * 15,
            "exactly 16",
        ),
        (
            "tactile_prototype",
            (1.0,) * 7,
            "exactly 8",
        ),
        (
            "tactile_prototype",
            (1.0,) * 7 + (float("nan"),),
            "finite",
        ),
    ),
)
def test_invalid_target_vectors_are_rejected(
    field: str,
    value: tuple[float, ...],
    message: str,
) -> None:
    """Target vectors must retain dimensions and finite values."""

    values = {
        "item_id": "item-001",
        "family_id": 1,
        "olfactory_prototype": (1.0,) * 16,
        "tactile_prototype": (1.0,) * 8,
    }
    values[field] = value

    with pytest.raises(
        MultisensoryRecordError,
        match=message,
    ):
        MultisensoryTarget(**values)


def test_boolean_family_identifier_is_rejected() -> None:
    """Boolean values cannot masquerade as integer family IDs."""

    with pytest.raises(
        MultisensoryRecordError,
        match="family_id",
    ):
        make_target(family_id=True)


def test_training_event_requires_development_regime() -> None:
    """Training rows cannot be labeled as final evaluation regimes."""

    with pytest.raises(
        MultisensoryRecordError,
        match="training event",
    ):
        make_event(
            split=MultisensorySplit.TRAIN,
            support_regime=SupportRegime.SEEN_ITEM,
        )


def test_final_event_requires_evaluation_regime() -> None:
    """Final-test rows cannot use the development regime."""

    with pytest.raises(
        MultisensoryRecordError,
        match="final-test event",
    ):
        make_event(
            support_regime=SupportRegime.DEVELOPMENT,
        )


def test_valid_missing_touch_view() -> None:
    """A missing-touch view contains odor and zero touch quality."""

    view = replace(
        make_view(),
        view_id="latent-001-missing-touch",
        condition=ConditionLabel.MISSING_TOUCH,
        tactile_vector=None,
        tactile_quality=0.0,
    )

    assert view.olfactory_available is True
    assert view.tactile_available is False


def test_missing_vector_requires_zero_quality() -> None:
    """An absent modality cannot retain positive quality."""

    with pytest.raises(
        MultisensoryRecordError,
        match="quality must be zero",
    ):
        replace(
            make_view(),
            condition=ConditionLabel.MISSING_TOUCH,
            tactile_vector=None,
            tactile_quality=0.5,
        )


def test_present_vector_requires_positive_quality() -> None:
    """An available modality cannot have zero quality."""

    with pytest.raises(
        MultisensoryRecordError,
        match="quality must be positive",
    ):
        replace(
            make_view(),
            tactile_quality=0.0,
        )


def test_quality_outside_unit_interval_is_rejected() -> None:
    """Modality quality is bounded between zero and one."""

    with pytest.raises(
        MultisensoryRecordError,
        match="between 0 and 1",
    ):
        replace(
            make_view(),
            olfactory_quality=1.1,
        )


def test_conflict_condition_requires_conflict_flag() -> None:
    """Contradictory modalities must be marked as conflicting."""

    with pytest.raises(
        MultisensoryRecordError,
        match="conflict flag",
    ):
        replace(
            make_view(),
            condition=ConditionLabel.CONTRADICTORY_MODALITIES,
            modality_conflict=False,
        )


def test_nonconflict_condition_rejects_conflict_flag() -> None:
    """Only the contradictory condition can assert conflict."""

    with pytest.raises(
        MultisensoryRecordError,
        match="conflict flag",
    ):
        replace(
            make_view(),
            modality_conflict=True,
        )


def test_temporal_condition_requires_nonzero_offset() -> None:
    """Temporal misalignment must include an actual offset."""

    with pytest.raises(
        MultisensoryRecordError,
        match="nonzero temporal offset",
    ):
        replace(
            make_view(),
            condition=ConditionLabel.TEMPORAL_MISALIGNMENT,
            temporal_offset_steps=0,
        )


def test_non_temporal_condition_rejects_offset() -> None:
    """Other conditions cannot silently include temporal shift."""

    with pytest.raises(
        MultisensoryRecordError,
        match="zero temporal offset",
    ):
        replace(
            make_view(),
            temporal_offset_steps=3,
        )


def test_valid_dataset_preserves_ground_truth() -> None:
    """Targets, events, and views agree on item and family."""

    dataset = make_dataset()

    assert len(dataset.targets) == 1
    assert len(dataset.latent_events) == 1
    assert len(dataset.condition_views) == 1


def test_duplicate_target_identifier_is_rejected() -> None:
    """Target identifiers must be unique."""

    target = make_target()

    with pytest.raises(
        MultisensoryRecordError,
        match="target identifiers",
    ):
        make_dataset(targets=(target, target))


def test_duplicate_latent_event_identifier_is_rejected() -> None:
    """Latent event identifiers must be unique."""

    event = make_event()

    with pytest.raises(
        MultisensoryRecordError,
        match="latent event identifiers",
    ):
        make_dataset(events=(event, event))


def test_unknown_event_target_is_rejected() -> None:
    """Every latent event target must exist in the library."""

    with pytest.raises(
        MultisensoryRecordError,
        match="event target",
    ):
        make_dataset(
            events=(
                make_event(
                    target_item_id="unknown-item",
                ),
            ),
        )


def test_inconsistent_event_family_is_rejected() -> None:
    """Event family ground truth must match its target."""

    with pytest.raises(
        MultisensoryRecordError,
        match="event target-family",
    ):
        make_dataset(
            events=(
                make_event(
                    target_family_id=2,
                ),
            ),
        )


def test_unknown_view_event_is_rejected() -> None:
    """Every condition view must reference a latent event."""

    with pytest.raises(
        MultisensoryRecordError,
        match="view latent event",
    ):
        make_dataset(
            views=(
                make_view(
                    latent_event_id="unknown-latent",
                ),
            ),
        )


def test_view_ground_truth_must_match_latent_event() -> None:
    """A condition view cannot redefine target ground truth."""

    with pytest.raises(
        MultisensoryRecordError,
        match="view ground truth",
    ):
        make_dataset(
            views=(
                make_view(
                    target_family_id=2,
                ),
            ),
        )


def test_duplicate_condition_view_is_rejected() -> None:
    """One latent event cannot repeat the same condition."""

    first = make_view(
        view_id="view-one",
    )
    second = make_view(
        view_id="view-two",
    )

    with pytest.raises(
        MultisensoryRecordError,
        match="duplicate condition view",
    ):
        make_dataset(
            views=(first, second),
        )
