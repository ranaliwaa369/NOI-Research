"""Tests for deterministic Track A result export."""

from hashlib import sha256
import json

from src.evaluation.seen_item_final_experiment import (
    SeenItemFinalExperiment,
)
from src.evaluation.seen_item_memory_experiment import (
    SeenItemEvaluation,
    SeenItemSystem,
)
from src.evaluation.seen_item_result_export import (
    export_seen_item_final_experiment,
)


def make_evaluation(
    system: SeenItemSystem,
    alpha: float,
) -> SeenItemEvaluation:
    return SeenItemEvaluation(
        system=system,
        alpha=alpha,
        event_ids=("validation-000001",),
        rankings=(("odor-a", "odor-b"),),
        relevant_items=(frozenset(("odor-a",)),),
        recall_at_1=1.0,
        recall_at_10=1.0,
        mean_reciprocal_rank=1.0,
        ndcg_at_10=1.0,
    )


def make_experiment() -> SeenItemFinalExperiment:
    return SeenItemFinalExperiment(
        training_event_count=10,
        calibration_event_count=2,
        final_test_event_count=1,
        raw_final_test_event_count=1,
        reachable_event_fraction=1.0,
        calibration_template_ids=(1,),
        final_test_template_ids=(2,),
        final_test_event_ids=("validation-000001",),
        selected_hybrid_alpha=0.5,
        evaluations=(
            make_evaluation(
                SeenItemSystem.MEMORY_ONLY,
                0.0,
            ),
            make_evaluation(
                SeenItemSystem.RIDGE_ONLY,
                1.0,
            ),
            make_evaluation(
                SeenItemSystem.HYBRID,
                0.5,
            ),
        ),
        oracle_used=False,
        final_test_tuning_used=False,
        protocol_hash="protocol-hash",
    )


def test_export_creates_json_and_sha256_files(
    tmp_path,
) -> None:
    result = export_seen_item_final_experiment(
        make_experiment(),
        tmp_path / "track_a_results.json",
    )

    assert result.json_path.is_file()
    assert result.sha256_path.is_file()

    observed_hash = sha256(
        result.json_path.read_bytes()
    ).hexdigest()

    assert result.sha256 == observed_hash
    assert (
        result.sha256_path.read_text(
            encoding="utf-8"
        ).strip()
        == observed_hash
    )


def test_export_contains_required_audit_fields(
    tmp_path,
) -> None:
    result = export_seen_item_final_experiment(
        make_experiment(),
        tmp_path / "track_a_results.json",
    )

    payload = json.loads(
        result.json_path.read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == "0.2.0"
    assert payload["evaluation_track"] == (
        "seen_item_episodic_retrieval"
    )
    assert payload["protocol_hash"] == "protocol-hash"
    assert payload["oracle_used"] is False
    assert payload["final_test_tuning_used"] is False
    assert payload["reachable_event_fraction"] == 1.0

    assert set(payload["systems"]) == {
        "memory_only",
        "ridge_only",
        "hybrid",
    }


def test_export_is_deterministic(tmp_path) -> None:
    experiment = make_experiment()

    first = export_seen_item_final_experiment(
        experiment,
        tmp_path / "first.json",
    )
    second = export_seen_item_final_experiment(
        experiment,
        tmp_path / "second.json",
    )

    assert first.json_path.read_bytes() == (
        second.json_path.read_bytes()
    )
    assert first.sha256 == second.sha256
