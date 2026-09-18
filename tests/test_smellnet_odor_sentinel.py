"""Behavioral tests for the executable NOI v0.5 odor sentinel."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.smellnet_adapter import SmellNetSplit
from src.evaluation.smellnet_novel_odor_protocol import (
    RepeatSensingPair,
    TemporalSensorWindow,
)
from src.evaluation.smellnet_odor_sentinel import (
    OdorSentinel,
    SentinelDecision,
    evaluate_sentinel_decisions,
    extract_temporal_features,
)


def make_window(
    family: str,
    base: float,
    index: int,
) -> TemporalSensorWindow:
    rows = tuple(
        tuple(
            base * (channel + 1)
            + 0.03 * step * (channel + 1)
            + 0.01 * index
            for channel in range(6)
        )
        for step in range(16)
    )
    recording_id = f"training/{family}/sample-{index}.csv"
    return TemporalSensorWindow(
        window_id=f"{recording_id}#0:16",
        recording_id=recording_id,
        split=SmellNetSplit.TRAINING,
        ingredient=f"{family}-ingredient",
        family=family,
        start=0,
        stop=16,
        sensor_rows=rows,
        source_sha256="a" * 64,
    )


def make_engine() -> tuple[OdorSentinel, tuple[TemporalSensorWindow, ...]]:
    training = tuple(
        make_window(family, base, index)
        for family, base in (("nuts", 1.0), ("spices", 4.0), ("herbs", 7.0))
        for index in range(8)
    )
    known = tuple(
        make_window(family, base, 100 + index)
        for family, base in (("nuts", 1.1), ("spices", 4.1), ("herbs", 7.1))
        for index in range(3)
    )
    unknown = tuple(make_window("fruits", 14.0, 200 + index) for index in range(8))
    engine = OdorSentinel().fit(training)
    engine.calibrate(known, unknown, required_known_acceptance=0.75)
    return engine, known + unknown


def test_temporal_feature_vector_uses_dynamics() -> None:
    window = make_window("nuts", 1.0, 0)
    features = extract_temporal_features(window)
    assert features.shape == (66,)
    assert np.isfinite(features).all()
    assert not np.allclose(features, 0)


def test_sentinel_trains_calibrates_and_emits_actions() -> None:
    engine, validation = make_engine()
    assert engine.calibration is not None
    decisions = tuple(
        engine.decide(window, allow_repeat=False) for window in validation
    )
    assert {value.action for value in decisions} <= {
        "recognize",
        "abstain_unknown",
    }
    assert all(0 <= value.combined_support <= 1 for value in decisions)
    assert all(0 <= value.novelty_score <= 1 for value in decisions)


def test_repeat_pair_produces_terminal_fused_decision() -> None:
    engine, validation = make_engine()
    first = validation[0]
    second = TemporalSensorWindow(
        window_id=f"{first.recording_id}#16:32",
        recording_id=first.recording_id,
        split=first.split,
        ingredient=first.ingredient,
        family=first.family,
        start=16,
        stop=32,
        sensor_rows=first.sensor_rows,
        source_sha256=first.source_sha256,
    )
    decision = engine.decide_pair(
        RepeatSensingPair(pair_id="pair-1", initial=first, repeat=second)
    )
    assert decision.used_repeat is True
    assert decision.action in {"recognize", "abstain_unknown"}


def test_asymmetric_repeat_never_promotes_abstention() -> None:
    engine, validation = make_engine()
    first = validation[-1]
    second = TemporalSensorWindow(
        window_id=f"{first.recording_id}#16:32",
        recording_id=first.recording_id,
        split=first.split,
        ingredient=first.ingredient,
        family=first.family,
        start=16,
        stop=32,
        sensor_rows=first.sensor_rows,
        source_sha256=first.source_sha256,
    )
    pair = RepeatSensingPair(
        pair_id="pair-asymmetric",
        initial=first,
        repeat=second,
    )
    initial = engine.decide(first, allow_repeat=False)
    decision = engine.decide_pair_asymmetric_veto(pair)
    if initial.action != "recognize":
        assert decision.action != "recognize"


def test_metrics_separate_false_known_from_safe_abstention() -> None:
    decisions = (
        SentinelDecision(
            "w1", "r1", "nuts", "nuts", "recognize",
            0.9, 0.9, 0.9, 0.1, True,
        ),
        SentinelDecision(
            "w2", "r2", "fruits", "nuts", "recognize",
            0.8, 0.7, 0.75, 0.25, False,
        ),
        SentinelDecision(
            "w3", "r3", "fruits", "spices", "abstain_unknown",
            0.4, 0.2, 0.3, 0.7, False,
        ),
    )
    metrics = evaluate_sentinel_decisions(
        decisions,
        unknown_family="fruits",
    )
    assert metrics.false_known_rate == pytest.approx(0.5)
    assert metrics.unknown_rejection_rate == pytest.approx(0.5)
    assert metrics.known_top1_accuracy == pytest.approx(1.0)
