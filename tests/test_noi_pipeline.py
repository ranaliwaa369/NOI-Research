"""Tests for the integrated NOI retrieval pipeline."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.evaluation.synthetic_generator import (
    generate_synthetic_pilot,
)
from src.models import (
    MultimodalContext,
    OutputRequest,
    PolicyOutcome,
)
from src.safety.policy_gate import load_policy_rules
from src.system.noi_pipeline import (
    HybridRetrievalCandidate,
    NOIPipeline,
    NOIPipelineError,
    NOIRetrievalResult,
    load_noi_system_configuration,
)


SYNTHETIC_CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"
SYSTEM_CONFIG_PATH = "configs/noi_system_v0.1.yaml"
POLICY_PATH = "configs/policy_rules.yaml"

PROTOCOL_HASH = (
    "e885f537b3209d7052d1517efc5be753"
    "24df39689cb4e54b38a241c95f22e512"
)

TRAINED_AT = datetime(
    2026,
    8,
    22,
    19,
    5,
    tzinfo=timezone.utc,
)


@pytest.fixture(scope="module")
def pilot_dataset():
    """Return the deterministic 200-event synthetic pilot."""

    return generate_synthetic_pilot(
        SYNTHETIC_CONFIG_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )


@pytest.fixture(scope="module")
def system_configuration():
    """Return the locked integrated-system configuration."""

    return load_noi_system_configuration(
        SYSTEM_CONFIG_PATH
    )


@pytest.fixture(scope="module")
def policy_configuration():
    """Return the locked simulated policy configuration."""

    return load_policy_rules(POLICY_PATH)


@pytest.fixture(scope="module")
def fitted_pipeline(
    pilot_dataset,
    system_configuration,
    policy_configuration,
) -> NOIPipeline:
    """Fit the integrated pipeline once for read-only tests."""

    pipeline = NOIPipeline(
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=PROTOCOL_HASH,
    )

    pipeline.fit(
        pilot_dataset,
        trained_at_utc=TRAINED_AT,
    )

    return pipeline


def context_from_event(
    event,
    *,
    timestamp_utc: datetime = TRAINED_AT,
) -> MultimodalContext:
    """Convert one synthetic event into a runtime context."""

    return MultimodalContext(
        event_id=event.event_id,
        timestamp_utc=timestamp_utc,
        text_vector=event.text_vector,
        image_vector=event.image_vector,
        audio_vector=event.audio_vector,
        metadata={},
    )


def first_validation_event(pilot_dataset):
    """Return the first validation event deterministically."""

    return sorted(
        (
            event
            for event in pilot_dataset.events
            if event.split.value == "validation"
        ),
        key=lambda event: event.event_id,
    )[0]


def build_fresh_pipeline(
    *,
    pilot_dataset,
    system_configuration,
    policy_configuration,
) -> NOIPipeline:
    """Return a separately fitted pipeline for mutation tests."""

    pipeline = NOIPipeline(
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=PROTOCOL_HASH,
    )
    pipeline.fit(
        pilot_dataset,
        trained_at_utc=TRAINED_AT,
    )
    return pipeline


def test_locked_configuration_loads(
    system_configuration,
) -> None:
    """The clarified score definition must remain locked."""

    hybrid = system_configuration["hybrid_retrieval"]

    assert hybrid["negative_cosine_handling"] == "clip_to_zero"
    assert hybrid["absent_memory_item_score"] == 0.0
    assert hybrid["memory_aggregation_per_odor"] == "maximum"
    assert hybrid["alpha_selection"]["candidates"] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]


def test_missing_configuration_is_rejected(
    tmp_path: Path,
) -> None:
    """A missing system definition must fail closed."""

    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(
        NOIPipelineError,
        match="not found",
    ):
        load_noi_system_configuration(missing_path)


def test_pipeline_is_fitted(
    fitted_pipeline: NOIPipeline,
) -> None:
    """Training and validation selection must complete."""

    assert fitted_pipeline.is_fitted is True
    assert 0.0 <= fitted_pipeline.selected_alpha <= 1.0
    assert fitted_pipeline.protocol_hash == PROTOCOL_HASH


def test_locked_split_counts_are_preserved(
    fitted_pipeline: NOIPipeline,
) -> None:
    """Only 140 train and 20 validation events may be used."""

    assert len(fitted_pipeline.training_event_ids) == 140
    assert len(fitted_pipeline.validation_event_ids) == 20
    assert fitted_pipeline.training_event_ids.isdisjoint(
        fitted_pipeline.validation_event_ids
    )


def test_ood_events_are_not_used_for_fit_or_selection(
    pilot_dataset,
    fitted_pipeline: NOIPipeline,
) -> None:
    """Locked OOD events cannot influence training or alpha."""

    ood_ids = {
        event.event_id
        for event in pilot_dataset.events
        if event.split.value == "ood_test"
    }

    used_ids = (
        fitted_pipeline.training_event_ids
        | fitted_pipeline.validation_event_ids
    )

    assert ood_ids
    assert ood_ids.isdisjoint(used_ids)


def test_retrieval_returns_ten_ranked_candidates(
    pilot_dataset,
    fitted_pipeline: NOIPipeline,
) -> None:
    """A default retrieval must return deterministic top ten."""

    event = first_validation_event(pilot_dataset)

    result = fitted_pipeline.retrieve(
        context_from_event(event)
    )

    assert isinstance(result, NOIRetrievalResult)
    assert len(result.candidates) == 10
    assert tuple(
        candidate.rank for candidate in result.candidates
    ) == tuple(range(1, 11))
    assert result.oracle_used is False


def test_retrieval_is_deterministic(
    pilot_dataset,
    fitted_pipeline: NOIPipeline,
) -> None:
    """Repeated retrievals must be exactly identical."""

    context = context_from_event(
        first_validation_event(pilot_dataset)
    )

    first = fitted_pipeline.retrieve(context)
    second = fitted_pipeline.retrieve(context)

    assert first == second


@pytest.mark.parametrize(
    "alpha",
    (0.0, 0.25, 0.5, 0.75, 1.0),
)
def test_hybrid_formula_is_exact(
    pilot_dataset,
    fitted_pipeline: NOIPipeline,
    alpha: float,
) -> None:
    """Each score must follow the locked convex combination."""

    context = context_from_event(
        first_validation_event(pilot_dataset)
    )

    result = fitted_pipeline.retrieve(
        context,
        alpha=alpha,
    )

    assert result.selected_alpha == alpha

    for candidate in result.candidates:
        expected = (
            alpha * candidate.library_score
            + (1.0 - alpha) * candidate.memory_score
        )
        assert candidate.hybrid_score == pytest.approx(
            expected,
            abs=1e-12,
        )


def test_all_component_scores_are_bounded(
    pilot_dataset,
    fitted_pipeline: NOIPipeline,
) -> None:
    """Clipped component and hybrid scores must remain valid."""

    context = context_from_event(
        first_validation_event(pilot_dataset)
    )

    for alpha in (0.0, 0.5, 1.0):
        result = fitted_pipeline.retrieve(
            context,
            alpha=alpha,
        )

        for candidate in result.candidates:
            assert 0.0 <= candidate.library_score <= 1.0
            assert 0.0 <= candidate.memory_score <= 1.0
            assert 0.0 <= candidate.hybrid_score <= 1.0


def test_result_is_immutable(
    pilot_dataset,
    fitted_pipeline: NOIPipeline,
) -> None:
    """A completed retrieval cannot be silently modified."""

    result = fitted_pipeline.retrieve(
        context_from_event(
            first_validation_event(pilot_dataset)
        )
    )

    with pytest.raises(FrozenInstanceError):
        result.selected_alpha = 0.0  # type: ignore[misc]


def test_candidate_is_immutable() -> None:
    """Ranked candidates must be immutable audit artifacts."""

    candidate = HybridRetrievalCandidate(
        item_id="odor-001",
        hybrid_score=0.5,
        library_score=0.5,
        memory_score=0.5,
        rank=1,
    )

    with pytest.raises(FrozenInstanceError):
        candidate.rank = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "invalid_alpha",
    (
        -0.1,
        1.1,
        float("inf"),
        float("nan"),
        True,
    ),
)
def test_invalid_retrieval_alpha_is_rejected(
    pilot_dataset,
    fitted_pipeline: NOIPipeline,
    invalid_alpha,
) -> None:
    """Runtime alpha overrides must stay inside [0, 1]."""

    context = context_from_event(
        first_validation_event(pilot_dataset)
    )

    with pytest.raises(
        NOIPipelineError,
        match="alpha",
    ):
        fitted_pipeline.retrieve(
            context,
            alpha=invalid_alpha,
        )


def test_use_before_fit_is_rejected(
    pilot_dataset,
    system_configuration,
    policy_configuration,
) -> None:
    """Retrieval before fitting must fail explicitly."""

    pipeline = NOIPipeline(
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=PROTOCOL_HASH,
    )

    context = context_from_event(
        first_validation_event(pilot_dataset)
    )

    with pytest.raises(
        NOIPipelineError,
        match="fitted",
    ):
        pipeline.retrieve(context)


def test_correction_is_audited(
    pilot_dataset,
    system_configuration,
    policy_configuration,
) -> None:
    """A correction must preserve identity and create an audit record."""

    pipeline = build_fresh_pipeline(
        pilot_dataset=pilot_dataset,
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
    )

    training_event = sorted(
        (
            event
            for event in pilot_dataset.events
            if event.split.value == "train"
        ),
        key=lambda event: event.event_id,
    )[0]

    alternative_target = next(
        target
        for target in pilot_dataset.odor_targets
        if target.item_id != training_event.target_item_id
    )

    audit = pipeline.correct_memory(
        correction_id="correction-test-001",
        memory_id=f"memory::{training_event.event_id}",
        corrected_at_utc=TRAINED_AT,
        reason="Locked synthetic correction test.",
        corrected_odor_item_id=alternative_target.item_id,
    )

    assert audit.memory_id == (
        f"memory::{training_event.event_id}"
    )
    assert audit.previous_odor_item_id == (
        training_event.target_item_id
    )
    assert audit.corrected_odor_item_id == (
        alternative_target.item_id
    )
    assert audit.protocol_hash == PROTOCOL_HASH
    assert audit.resulting_correction_count == 1


def test_unknown_correction_target_is_rejected(
    pilot_dataset,
    system_configuration,
    policy_configuration,
) -> None:
    """Corrections cannot introduce an unknown odor identifier."""

    pipeline = build_fresh_pipeline(
        pilot_dataset=pilot_dataset,
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
    )

    training_event = next(
        event
        for event in pilot_dataset.events
        if event.split.value == "train"
    )

    with pytest.raises(
        NOIPipelineError,
        match="odor library",
    ):
        pipeline.correct_memory(
            correction_id="correction-test-unknown",
            memory_id=f"memory::{training_event.event_id}",
            corrected_at_utc=TRAINED_AT,
            reason="Invalid target test.",
            corrected_odor_item_id="UNKNOWN-ODOR",
        )


def test_policy_gate_allows_conforming_simulated_request(
    fitted_pipeline: NOIPipeline,
) -> None:
    """A conforming locked simulated request may be allowed."""

    decision = fitted_pipeline.evaluate_output_request(
        OutputRequest(
            request_id="request-allow-001",
            item_id="SIM-CARTRIDGE-001",
            concentration_ppm=0.5,
            duration_seconds=10.0,
            environment_volume_m3=30.0,
            ventilation_ach=2.0,
            user_consent=True,
        )
    )

    assert decision.outcome is PolicyOutcome.ALLOW
    assert decision.protocol_hash == PROTOCOL_HASH


def test_policy_gate_blocks_absent_consent(
    fitted_pipeline: NOIPipeline,
) -> None:
    """The simulated gate must fail closed without consent."""

    decision = fitted_pipeline.evaluate_output_request(
        OutputRequest(
            request_id="request-block-001",
            item_id="SIM-CARTRIDGE-001",
            concentration_ppm=0.5,
            duration_seconds=10.0,
            environment_volume_m3=30.0,
            ventilation_ach=2.0,
            user_consent=False,
        )
    )

    assert decision.outcome is PolicyOutcome.BLOCK


def test_policy_gate_requires_missing_information(
    fitted_pipeline: NOIPipeline,
) -> None:
    """Missing required information cannot produce ALLOW."""

    decision = fitted_pipeline.evaluate_output_request(
        OutputRequest(
            request_id="request-missing-001",
            item_id="SIM-CARTRIDGE-001",
            concentration_ppm=None,
            duration_seconds=10.0,
            environment_volume_m3=30.0,
            ventilation_ach=2.0,
            user_consent=True,
        )
    )

    assert (
        decision.outcome
        is PolicyOutcome.REQUIRE_MISSING_INFORMATION
    )


def test_policy_evaluation_does_not_change_ranking(
    pilot_dataset,
    fitted_pipeline: NOIPipeline,
) -> None:
    """The downstream policy gate must remain ranking-independent."""

    context = context_from_event(
        first_validation_event(pilot_dataset)
    )
    before = fitted_pipeline.retrieve(context)

    fitted_pipeline.evaluate_output_request(
        OutputRequest(
            request_id="request-independent-001",
            item_id="SIM-CARTRIDGE-DISABLED",
            concentration_ppm=0.0,
            duration_seconds=0.0,
            environment_volume_m3=30.0,
            ventilation_ach=2.0,
            user_consent=True,
        )
    )

    after = fitted_pipeline.retrieve(context)

    assert before == after
