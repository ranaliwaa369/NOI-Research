"""Tests for locked NOI v0.3 confirmatory aggregation."""

import json
from pathlib import Path

import pytest

from experiments.analyze_noi_v0_3_confirmatory import (
    ConfirmatoryAggregateError,
    analyze_confirmatory_payloads,
    export_confirmatory_aggregate,
)
from src.evaluation.noi_v0_3_retrieval import (
    NOIV03System,
)


SEEDS = tuple(range(1301, 1311))

NONCLEAN_CONDITIONS = (
    "degraded_odor",
    "degraded_touch",
    "missing_touch",
    "missing_odor",
    "contradictory_modalities",
    "temporal_misalignment",
)


def make_record(
    *,
    seed: int,
    latent_event_id: str,
    condition: str,
    support_regime: str,
    system: NOIV03System,
    reciprocal_rank: float,
    confidence: float,
    abstained: bool,
    correct: bool,
    predicted_supported: bool,
) -> dict:
    """Create one minimal canonical posthoc result record."""

    return {
        "view_id": (
            f"{seed}-{latent_event_id}-{condition}"
        ),
        "latent_event_id": latent_event_id,
        "statistical_unit": "latent_event_id",
        "condition": condition,
        "support_regime": support_regime,
        "system": system.value,
        "target_item_id": "target",
        "target_family_id": 0,
        "ranking": [] if abstained else ["candidate"],
        "scores": [] if abstained else [0.8],
        "abstained": abstained,
        "correct": correct,
        "reciprocal_rank": reciprocal_rank,
        "confidence": confidence,
        "odor_weight": 0.0 if abstained else 1.0,
        "touch_weight": 0.0,
        "touch_requested": False,
        "support_score": 0.0,
        "predicted_supported": predicted_supported,
        "true_supported": (
            support_regime == "seen_item"
        ),
        "support_uncertainty_status": (
            "certain_supported"
            if predicted_supported
            else "certain_unsupported"
        ),
        "conflict_score": 0.0,
        "conflict_detected": False,
        "temporal_conflict_detected": False,
    }


def positive_payload(seed: int) -> dict:
    """Create one seed where all registered success rules pass."""

    records = []

    for system in NOIV03System:
        clean_rr = 0.40

        if system is NOIV03System.ODOR_ONLY_COSINE:
            clean_rr = 0.80
        elif system is NOIV03System.SUPPORT_GATE_ODOR_ONLY:
            clean_rr = 0.79
        elif system in (
            NOIV03System.NAIVE_CONCATENATION,
            NOIV03System.FIXED_WEIGHT_FUSION,
        ):
            clean_rr = 0.80
        elif system is (
            NOIV03System
            .SUPPORT_GATE_RELIABILITY_FUSION_WITH_ABSTENTION
        ):
            clean_rr = 0.79

        records.append(
            make_record(
                seed=seed,
                latent_event_id="seen-clean",
                condition="clean",
                support_regime="seen_item",
                system=system,
                reciprocal_rank=clean_rr,
                confidence=0.90,
                abstained=False,
                correct=True,
                predicted_supported=True,
            )
        )

        records.append(
            make_record(
                seed=seed,
                latent_event_id="unknown-clean",
                condition="clean",
                support_regime="unseen_family",
                system=system,
                reciprocal_rank=0.0,
                confidence=(
                    0.0
                    if system
                    is NOIV03System.SUPPORT_GATE_ODOR_ONLY
                    else 0.90
                ),
                abstained=(
                    system
                    is NOIV03System.SUPPORT_GATE_ODOR_ONLY
                ),
                correct=False,
                predicted_supported=(
                    system
                    is not NOIV03System.SUPPORT_GATE_ODOR_ONLY
                ),
            )
        )

    for condition in NONCLEAN_CONDITIONS:
        for system in NOIV03System:
            reciprocal_rank = 0.30
            confidence = 0.60
            abstained = False
            correct = False

            if system in (
                NOIV03System.TOUCH_ONLY_COSINE,
                NOIV03System.FIXED_WEIGHT_FUSION,
            ):
                reciprocal_rank = 0.50

            if system is (
                NOIV03System
                .RELIABILITY_GATED_OLFACTORY_TACTILE_FUSION
            ):
                reciprocal_rank = (
                    0.70
                    if condition in (
                        "degraded_odor",
                        "missing_odor",
                    )
                    else 0.50
                )

            if system in (
                NOIV03System.NAIVE_CONCATENATION,
                NOIV03System.FIXED_WEIGHT_FUSION,
            ):
                confidence = 0.90
                correct = False

            if system is (
                NOIV03System
                .SUPPORT_GATE_RELIABILITY_FUSION_WITH_ABSTENTION
            ):
                reciprocal_rank = 0.0
                confidence = 0.0
                abstained = True
                correct = False

            records.append(
                make_record(
                    seed=seed,
                    latent_event_id=f"event-{condition}",
                    condition=condition,
                    support_regime="seen_item",
                    system=system,
                    reciprocal_rank=reciprocal_rank,
                    confidence=confidence,
                    abstained=abstained,
                    correct=correct,
                    predicted_supported=True,
                )
            )

    return {
        "schema_version": "noi-v0.3-confirmatory-seed-v1",
        "study_phase": "confirmatory",
        "confirmatory_execution": True,
        "seed": seed,
        "view_results": records,
        "integrity": {
            "final_test_labels_used_for_training": False,
            "final_test_labels_used_for_calibration": False,
            "final_test_labels_used_for_scoring_only": True,
            "condition_metadata_used_as_model_input": False,
            "target_labels_used_as_inference_input": False,
            "paired_views_treated_as_independent": False,
            "thresholds_changed_from_final_test": False,
        },
    }


def positive_payloads() -> tuple[dict, ...]:
    return tuple(
        positive_payload(seed)
        for seed in SEEDS
    )


def test_positive_mock_results_support_all_hypotheses() -> None:
    analysis = analyze_confirmatory_payloads(
        positive_payloads()
    )

    assert analysis["hypotheses"]["H6"]["status"] == "supported"
    assert analysis["hypotheses"]["H7"]["status"] == "supported"
    assert analysis["hypotheses"]["H8"]["status"] == "supported"


def test_h6_selects_strongest_clean_seen_baseline() -> None:
    analysis = analyze_confirmatory_payloads(
        positive_payloads()
    )

    assert analysis["hypotheses"]["H6"][
        "selected_comparator"
    ] == "fixed_weight_fusion"
    assert analysis["hypotheses"]["H6"][
        "false_known_reduction"
    ] >= 0.05
    assert analysis["hypotheses"]["H6"][
        "seen_item_clean_mrr_loss"
    ] <= 0.02


def test_h7_uses_locked_eligible_conditions() -> None:
    analysis = analyze_confirmatory_payloads(
        positive_payloads()
    )

    h7 = analysis["hypotheses"]["H7"]

    assert h7["eligible_conditions"] == [
        "degraded_odor",
        "missing_odor",
    ]
    assert h7["selected_comparator"] == (
        "fixed_weight_fusion"
    )
    assert h7["absolute_mrr_improvement"] >= 0.05
    assert h7["paired_bootstrap"][
        "confidence_interval_lower"
    ] > 0.0


def test_h8_must_pass_both_locked_comparators() -> None:
    analysis = analyze_confirmatory_payloads(
        positive_payloads()
    )

    h8 = analysis["hypotheses"]["H8"]

    assert set(h8["comparisons"]) == {
        "naive_concatenation",
        "fixed_weight_fusion",
    }
    assert all(
        comparison["status"] == "supported"
        for comparison in h8["comparisons"].values()
    )
    assert h8["status"] == "supported"


def test_secondary_tests_receive_holm_correction() -> None:
    analysis = analyze_confirmatory_payloads(
        positive_payloads()
    )

    holm = analysis["holm_correction"]

    assert holm["comparison_count"] == 3
    assert {
        comparison["name"]
        for comparison in holm["comparisons"]
    } == {
        "H7",
        "H8_vs_naive_concatenation",
        "H8_vs_fixed_weight_fusion",
    }


def test_all_registered_seeds_are_retained() -> None:
    analysis = analyze_confirmatory_payloads(
        positive_payloads()
    )

    assert analysis["completed_seeds"] == list(SEEDS)
    assert analysis["seed_count"] == 10
    assert analysis["integrity"][
        "silent_seed_removal_used"
    ] is False


def test_missing_seed_is_rejected() -> None:
    with pytest.raises(
        ConfirmatoryAggregateError,
        match="registered seeds",
    ):
        analyze_confirmatory_payloads(
            positive_payloads()[:-1]
        )


def test_duplicate_seed_is_rejected() -> None:
    payloads = positive_payloads()

    with pytest.raises(
        ConfirmatoryAggregateError,
        match="registered seeds",
    ):
        analyze_confirmatory_payloads(
            payloads[:-1] + (payloads[0],)
        )


def test_invalid_seed_schema_is_rejected() -> None:
    payloads = list(positive_payloads())
    payloads[0] = {
        **payloads[0],
        "schema_version": "wrong",
    }

    with pytest.raises(
        ConfirmatoryAggregateError,
        match="schema",
    ):
        analyze_confirmatory_payloads(
            tuple(payloads)
        )


def test_analysis_is_deterministic() -> None:
    assert analyze_confirmatory_payloads(
        positive_payloads()
    ) == analyze_confirmatory_payloads(
        positive_payloads()
    )


def test_aggregate_contains_no_runtime_or_timestamp() -> None:
    analysis = analyze_confirmatory_payloads(
        positive_payloads()
    )

    assert "runtime_seconds" not in analysis
    assert "timestamp" not in analysis


def test_export_round_trips_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    analysis = analyze_confirmatory_payloads(
        positive_payloads()
    )
    output = tmp_path / "aggregate.json"

    export_confirmatory_aggregate(
        analysis,
        output,
        overwrite=False,
    )

    observed = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert observed == analysis

    with pytest.raises(
        ConfirmatoryAggregateError,
        match="already exists",
    ):
        export_confirmatory_aggregate(
            analysis,
            output,
            overwrite=False,
        )
