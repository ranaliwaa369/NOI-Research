"""Tests for repeated Track A aggregate export."""

from pathlib import Path

import pytest

from src.evaluation.seen_item_repeated_analysis import (
    analyze_repeated_track_a,
)
from src.evaluation.seen_item_repeated_analysis_export import (
    RepeatedTrackAAnalysisExportError,
    build_repeated_track_a_payload,
    export_repeated_track_a_analysis,
)
from src.evaluation.seen_item_repeated_config import (
    load_seen_item_repeated_config,
)


RESULTS_DIRECTORY = (
    "results/v0.2.1/repeated_track_a"
)
CONFIG_PATH = (
    "configs/"
    "seen_item_repeated_evaluation_v0.2.1.yaml"
)
HASH_PATH = (
    "configs/"
    "seen_item_repeated_evaluation_v0.2.1.sha256"
)


@pytest.fixture(scope="module")
def analysis():
    config = load_seen_item_repeated_config(
        CONFIG_PATH,
        HASH_PATH,
    )

    protocol_sha256 = (
        Path(HASH_PATH)
        .read_text(encoding="utf-8")
        .strip()
        .split()[0]
    )

    return analyze_repeated_track_a(
        RESULTS_DIRECTORY,
        repeated_config=config,
        repeated_protocol_sha256=(
            protocol_sha256
        ),
    )


def test_payload_records_locked_scope_and_controls(
    analysis,
) -> None:
    payload = build_repeated_track_a_payload(
        analysis
    )

    assert payload["schema_version"] == "0.2.1"
    assert (
        payload["artifact_type"]
        == "repeated_track_a_aggregate"
    )
    assert payload["run_count"] == 10
    assert payload["run_ids"] == [
        f"seed-{index:02d}"
        for index in range(1, 11)
    ]

    controls = payload["controls"]

    assert controls["all_targets_reachable"] is True
    assert controls["oracle_used"] is False
    assert (
        controls["final_test_tuning_used"]
        is False
    )
    assert (
        controls["independent_replication_unit"]
        == "seed"
    )
    assert controls["pilot_included"] is False


def test_system_summaries_are_exported(
    analysis,
) -> None:
    payload = build_repeated_track_a_payload(
        analysis
    )

    assert set(payload["systems"]) == {
        "memory_only",
        "ridge_only",
        "hybrid",
    }

    summary = payload["systems"][
        "memory_only"
    ]["mean_reciprocal_rank"]

    expected = analysis.summary_for(
        "memory_only",
        "mean_reciprocal_rank",
    )

    assert summary["values"] == list(
        expected.values
    )
    assert summary["count"] == 10
    assert summary["mean"] == expected.mean
    assert summary["median"] == expected.median
    assert (
        summary["standard_deviation"]
        == expected.standard_deviation
    )


def test_primary_comparison_and_inference_are_exported(
    analysis,
) -> None:
    payload = build_repeated_track_a_payload(
        analysis
    )
    primary = payload["primary_comparison"]

    assert (
        primary["direction"]
        == "memory_only minus ridge_only"
    )
    assert (
        primary["metric"]
        == "mean_reciprocal_rank"
    )
    assert (
        primary["confidence_interval"]["level"]
        == 0.95
    )
    assert (
        primary["confidence_interval"]["lower"]
        > 0.0
    )
    assert (
        primary["confirmatory_interval_excludes_zero"]
        is True
    )

    inference = payload["inference"]

    assert inference["p_values_reported"] is False
    assert (
        inference["multiple_testing_applied"]
        is False
    )
    assert (
        inference["reason"]
        == (
            "No null-hypothesis significance test "
            "was prespecified."
        )
    )


def test_limitations_prohibit_overclaiming(
    analysis,
) -> None:
    payload = build_repeated_track_a_payload(
        analysis
    )
    limitations = payload["limitations"]

    assert any(
        "human olfactory equivalence"
        in value.lower()
        for value in limitations
    )
    assert any(
        "real-world deployment"
        in value.lower()
        for value in limitations
    )
    assert any(
        "synthetic"
        in value.lower()
        for value in limitations
    )


def test_export_is_deterministic(
    analysis,
    tmp_path,
) -> None:
    first = export_repeated_track_a_analysis(
        analysis,
        tmp_path / "first.json",
    )
    second = export_repeated_track_a_analysis(
        analysis,
        tmp_path / "second.json",
    )

    assert (
        first.json_path.read_bytes()
        == second.json_path.read_bytes()
    )
    assert first.sha256 == second.sha256

    assert (
        first.sha256_path.read_text(
            encoding="utf-8"
        ).strip()
        == first.sha256
    )


def test_non_json_output_is_rejected(
    analysis,
    tmp_path,
) -> None:
    with pytest.raises(
        RepeatedTrackAAnalysisExportError,
        match=".json",
    ):
        export_repeated_track_a_analysis(
            analysis,
            tmp_path / "aggregate.txt",
        )
