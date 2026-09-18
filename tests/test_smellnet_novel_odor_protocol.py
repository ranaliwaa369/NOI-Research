"""Tests for NOI v0.5 novel-family partitions and temporal evidence."""

from __future__ import annotations

from hashlib import sha256

import pytest

from src.evaluation.smellnet_adapter import (
    SMELLNET_SENSOR_COLUMNS,
    SmellNetRecording,
    SmellNetSplit,
)
from src.evaluation.smellnet_novel_odor_protocol import (
    LOCKED_FAMILY_ORDER,
    NovelOdorProtocolError,
    build_locked_novel_odor_folds,
    create_disjoint_repeat_sensing_pairs,
    create_nonoverlapping_windows,
)


FAMILY_INGREDIENTS = {
    "fruits": ("apple", "lemon"),
    "herbs": ("mint", "oregano"),
    "nuts": ("cashew", "walnuts"),
    "spices": ("cumin", "cinnamon"),
    "vegetables": ("potato", "tomato"),
}


def make_recording(
    ingredient: str,
    family: str,
    split: SmellNetSplit,
    index: int,
    *,
    row_count: int = 8,
) -> SmellNetRecording:
    """Create a deterministic, content-distinct recording fixture."""

    recording_id = f"{split.value}/{ingredient}/{index}.csv"
    base = float(
        sum(ord(character) for character in recording_id) + index * 100
    )
    rows = tuple(
        tuple(base + row * 10 + column for column in range(6))
        for row in range(row_count)
    )
    digest = sha256(recording_id.encode("utf-8")).hexdigest()
    return SmellNetRecording(
        recording_id=recording_id,
        split=split,
        ingredient=ingredient,
        family=family,
        source_path=f"/fixture/{recording_id}",
        sensor_columns=SMELLNET_SENSOR_COLUMNS,
        sensor_rows=rows,
        source_sha256=digest,
    )


def make_dataset() -> tuple[SmellNetRecording, ...]:
    """Create five families with grouped training and final recordings."""

    values: list[SmellNetRecording] = []
    for family, ingredients in FAMILY_INGREDIENTS.items():
        for ingredient in ingredients:
            values.extend(
                make_recording(
                    ingredient,
                    family,
                    SmellNetSplit.TRAINING,
                    index,
                )
                for index in range(3)
            )
            values.append(
                make_recording(
                    ingredient,
                    family,
                    SmellNetSplit.TESTING,
                    99,
                )
            )
    return tuple(values)


def test_windows_are_fixed_complete_and_nonoverlapping() -> None:
    recording = make_recording(
        "apple",
        "fruits",
        SmellNetSplit.TRAINING,
        0,
        row_count=10,
    )

    windows = create_nonoverlapping_windows(recording, window_size=4)

    assert tuple((value.start, value.stop) for value in windows) == (
        (0, 4),
        (4, 8),
    )
    assert all(len(value.sensor_rows) == 4 for value in windows)
    assert windows[0].stop <= windows[1].start


def test_partial_tail_is_not_reused_as_evidence() -> None:
    recording = make_recording(
        "apple",
        "fruits",
        SmellNetSplit.TRAINING,
        0,
        row_count=11,
    )

    windows = create_nonoverlapping_windows(recording, window_size=5)

    assert tuple((value.start, value.stop) for value in windows) == (
        (0, 5),
        (5, 10),
    )


@pytest.mark.parametrize("window_size", (True, 1, 100))
def test_invalid_or_unavailable_window_is_rejected(window_size: int) -> None:
    recording = make_recording(
        "apple",
        "fruits",
        SmellNetSplit.TRAINING,
        0,
        row_count=8,
    )

    with pytest.raises(NovelOdorProtocolError):
        create_nonoverlapping_windows(
            recording,
            window_size=window_size,
        )


def test_repeat_sensing_pairs_are_disjoint_and_do_not_reuse_windows() -> None:
    recording = make_recording(
        "apple",
        "fruits",
        SmellNetSplit.TESTING,
        99,
        row_count=16,
    )
    windows = create_nonoverlapping_windows(recording, window_size=4)

    pairs = create_disjoint_repeat_sensing_pairs(windows)

    assert len(pairs) == 2
    assert pairs[0].initial.start == 0
    assert pairs[0].repeat.start == 4
    assert pairs[1].initial.start == 8
    assert pairs[1].repeat.start == 12
    used_ids = [
        window.window_id
        for pair in pairs
        for window in (pair.initial, pair.repeat)
    ]
    assert len(used_ids) == len(set(used_ids))


def test_builds_five_deterministic_family_rotations() -> None:
    folds = build_locked_novel_odor_folds(make_dataset())

    assert len(folds) == 5
    assert tuple(value.final_unknown_family for value in folds) == (
        LOCKED_FAMILY_ORDER
    )
    assert tuple(value.validation_unknown_family for value in folds) == (
        "herbs",
        "nuts",
        "spices",
        "vegetables",
        "fruits",
    )


def test_final_unknown_family_is_absent_from_fold_development() -> None:
    recordings = make_dataset()
    by_id = {value.recording_id: value for value in recordings}

    for fold in build_locked_novel_odor_folds(recordings):
        development_ids = (
            *fold.model_train_ids,
            *fold.validation_known_ids,
            *fold.validation_unknown_ids,
        )
        assert all(
            by_id[value].family != fold.final_unknown_family
            for value in development_ids
        )
        assert all(
            by_id[value].family == fold.final_unknown_family
            for value in fold.final_unknown_ids
        )


def test_official_testing_is_used_only_for_final_roles() -> None:
    recordings = make_dataset()
    by_id = {value.recording_id: value for value in recordings}

    for fold in build_locked_novel_odor_folds(recordings):
        development_ids = (
            *fold.model_train_ids,
            *fold.validation_known_ids,
            *fold.validation_unknown_ids,
        )
        final_ids = (*fold.final_known_ids, *fold.final_unknown_ids)
        assert all(
            by_id[value].split is SmellNetSplit.TRAINING
            for value in development_ids
        )
        assert all(
            by_id[value].split is SmellNetSplit.TESTING
            for value in final_ids
        )


def test_each_recording_has_exactly_one_role_or_is_unused() -> None:
    recordings = make_dataset()
    all_ids = {value.recording_id for value in recordings}

    for fold in build_locked_novel_odor_folds(recordings):
        role_values = (
            fold.model_train_ids,
            fold.validation_known_ids,
            fold.validation_unknown_ids,
            fold.final_known_ids,
            fold.final_unknown_ids,
            fold.unused_ids,
        )
        flattened = [value for role in role_values for value in role]
        assert len(flattened) == len(set(flattened))
        assert set(flattened) == all_ids


def test_model_train_and_validation_known_are_recording_grouped() -> None:
    recordings = make_dataset()
    by_id = {value.recording_id: value for value in recordings}

    for fold in build_locked_novel_odor_folds(recordings):
        train_ingredients = {
            by_id[value].ingredient for value in fold.model_train_ids
        }
        validation_ingredients = {
            by_id[value].ingredient for value in fold.validation_known_ids
        }
        assert train_ingredients == validation_ingredients
        assert not set(fold.model_train_ids) & set(fold.validation_known_ids)


def test_missing_family_is_rejected() -> None:
    incomplete = tuple(
        value for value in make_dataset() if value.family != "vegetables"
    )

    with pytest.raises(NovelOdorProtocolError, match="five locked"):
        build_locked_novel_odor_folds(incomplete)


def test_known_ingredient_requires_grouped_validation_recording() -> None:
    recordings = tuple(
        value
        for value in make_dataset()
        if not (
            value.ingredient == "apple"
            and value.split is SmellNetSplit.TRAINING
            and not value.recording_id.endswith("/0.csv")
        )
    )

    with pytest.raises(NovelOdorProtocolError, match="at least two"):
        build_locked_novel_odor_folds(recordings)
