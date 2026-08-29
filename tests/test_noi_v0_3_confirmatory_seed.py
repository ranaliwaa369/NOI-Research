"""Tests for the NOI v0.3 confirmatory per-seed evaluator."""

import json
import math
from pathlib import Path

import pytest

from experiments.run_noi_v0_3_confirmatory import (
    ConfirmatoryExecutionError,
    build_confirmatory_seed_payload,
    evaluate_confirmatory_seed,
    export_confirmatory_seed_payload,
    score_to_confidence,
)
from src.evaluation.multisensory_conditions import (
    ConditionGenerationConfig,
    generate_multisensory_condition_views,
)
from src.evaluation.multisensory_records import (
    MultisensorySplit,
)
from src.evaluation.noi_v0_3_generator import (
    NOIV03GenerationConfig,
    generate_noi_v0_3_events,
)
from src.evaluation.noi_v0_3_retrieval import (
    NOIV03System,
)


LOCKED_VALUES = {
    "support_threshold": -20.0,
    "support_uncertainty_lower": -20.0,
    "support_uncertainty_upper": -20.0,
    "reliability_threshold": 0.10,
    "conflict_threshold": 0.20,
}


def generated_records():
    """Return a reduced nonconfirmatory integration allocation."""

    return generate_noi_v0_3_events(
        NOIV03GenerationConfig(
            seed=9901,
            train_event_count=70,
            validation_event_count=10,
            final_test_event_count=20,
            validation_seen_item_count=4,
            validation_known_family_unseen_item_count=3,
            validation_unseen_family_count=3,
            final_seen_item_count=8,
            final_known_family_unseen_item_count=6,
            final_unseen_family_count=6,
            known_family_count=4,
            training_items_per_family=4,
            withheld_items_per_known_family=2,
            validation_unknown_family_count=2,
            final_unknown_family_count=2,
            items_per_unknown_family=3,
            generator_version="0.3.1-confirmatory-test",
            feasibility_only=True,
        )
    )


def reduced_report():
    """Evaluate all systems on 20 latent events and seven views."""

    generated = generated_records()
    final_events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )

    conditions = generate_multisensory_condition_views(
        latent_events=final_events,
        targets=generated.targets,
        config=ConditionGenerationConfig(
            seed=9901,
            odor_noise_scale=0.10,
            tactile_noise_scale=0.10,
            degraded_quality=0.40,
            locked_temporal_offset_steps=3,
            generator_version="0.3.1-confirmatory-test",
        ),
    )

    return evaluate_confirmatory_seed(
        generated=generated,
        condition_result=conditions,
        locked_values=LOCKED_VALUES,
        top_k=10,
        false_confident_threshold=0.80,
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    (
        (-1.0, 0.0),
        (0.0, 0.5),
        (1.0, 1.0),
        (-2.0, 0.0),
        (2.0, 1.0),
    ),
)
def test_score_to_confidence_is_locked(
    score: float,
    expected: float,
) -> None:
    """Confidence uses the fixed clipped cosine transformation."""

    assert score_to_confidence(
        top_score=score,
        abstained=False,
    ) == expected


def test_abstention_confidence_is_zero() -> None:
    assert score_to_confidence(
        top_score=None,
        abstained=True,
    ) == 0.0


def test_reduced_report_has_all_paired_views_and_systems() -> None:
    report = reduced_report()

    assert report.seed == 9901
    assert report.final_latent_event_count == 20
    assert report.condition_view_count == 140
    assert report.system_count == 9
    assert report.system_evaluation_count == 1260


def test_every_registered_system_is_evaluated() -> None:
    report = reduced_report()

    assert set(report.system_summaries) == {
        system.value
        for system in NOIV03System
    }


def test_all_seven_conditions_are_reported() -> None:
    report = reduced_report()

    assert set(report.condition_summaries) == {
        "clean",
        "degraded_odor",
        "degraded_touch",
        "missing_touch",
        "missing_odor",
        "contradictory_modalities",
        "temporal_misalignment",
    }


def test_support_regimes_are_stratified() -> None:
    report = reduced_report()

    assert set(report.support_regime_summaries) == {
        "seen_item",
        "known_family_unseen_item",
        "unseen_family",
    }


def test_view_results_retain_latent_pairing() -> None:
    report = reduced_report()

    latent_ids = {
        result["latent_event_id"]
        for result in report.view_results
    }
    view_ids = {
        result["view_id"]
        for result in report.view_results
    }

    assert len(latent_ids) == 20
    assert len(view_ids) == 140
    assert all(
        result["statistical_unit"] == "latent_event_id"
        for result in report.view_results
    )


def test_results_contain_finite_metrics() -> None:
    report = reduced_report()

    for result in report.view_results:
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["reciprocal_rank"] <= 1.0
        assert all(
            math.isfinite(score)
            for score in result["scores"]
        )


def test_inference_audit_prohibits_oracle_inputs() -> None:
    report = reduced_report()

    assert report.integrity[
        "condition_metadata_used_as_model_input"
    ] is False
    assert report.integrity[
        "target_labels_used_as_inference_input"
    ] is False
    assert report.integrity[
        "final_test_labels_used_for_training"
    ] is False
    assert report.integrity[
        "final_test_labels_used_for_calibration"
    ] is False
    assert report.integrity[
        "final_test_labels_used_for_scoring_only"
    ] is True


def test_reduced_evaluation_is_deterministic() -> None:
    first = build_confirmatory_seed_payload(
        reduced_report()
    )
    second = build_confirmatory_seed_payload(
        reduced_report()
    )

    assert first == second


def test_payload_contains_no_runtime_or_timestamp() -> None:
    payload = build_confirmatory_seed_payload(
        reduced_report()
    )

    assert "runtime_seconds" not in payload
    assert "timestamp" not in payload


def test_payload_schema_is_confirmatory_seed_v1() -> None:
    payload = build_confirmatory_seed_payload(
        reduced_report()
    )

    assert payload["schema_version"] == (
        "noi-v0.3-confirmatory-seed-v1"
    )
    assert payload["study_phase"] == "confirmatory"
    assert payload["seed"] == 9901
    assert payload["confirmatory_execution"] is True


def test_export_is_canonical_and_round_trips(
    tmp_path: Path,
) -> None:
    payload = build_confirmatory_seed_payload(
        reduced_report()
    )
    output = tmp_path / "seed.json"

    export_confirmatory_seed_payload(
        payload,
        output,
        overwrite=False,
    )

    observed = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert observed == payload
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_export_refuses_overwrite(
    tmp_path: Path,
) -> None:
    payload = build_confirmatory_seed_payload(
        reduced_report()
    )
    output = tmp_path / "seed.json"

    export_confirmatory_seed_payload(
        payload,
        output,
        overwrite=False,
    )

    with pytest.raises(
        ConfirmatoryExecutionError,
        match="already exists",
    ):
        export_confirmatory_seed_payload(
            payload,
            output,
            overwrite=False,
        )


def test_invalid_locked_values_are_rejected() -> None:
    generated = generated_records()
    final_events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )
    conditions = generate_multisensory_condition_views(
        latent_events=final_events,
        targets=generated.targets,
        config=ConditionGenerationConfig(
            seed=9901,
            odor_noise_scale=0.10,
            tactile_noise_scale=0.10,
            degraded_quality=0.40,
            locked_temporal_offset_steps=3,
            generator_version="test",
        ),
    )

    with pytest.raises(
        ConfirmatoryExecutionError,
        match="locked",
    ):
        evaluate_confirmatory_seed(
            generated=generated,
            condition_result=conditions,
            locked_values={
                **LOCKED_VALUES,
                "conflict_threshold": float("nan"),
            },
            top_k=10,
            false_confident_threshold=0.80,
        )

def test_predicted_support_uses_each_system_decision() -> None:
    """False-known analysis must use each system's own abstention."""

    payload = build_confirmatory_seed_payload(
        reduced_report()
    )

    for record in payload["view_results"]:
        assert record["predicted_supported"] is (
            not record["abstained"]
        )
        assert isinstance(
            record["support_gate_predicted_supported"],
            bool,
        )
