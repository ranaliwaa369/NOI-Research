"""Tests for repeated Track B aggregate analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.track_b_analysis import (
    TrackBAnalysisError,
    analyze_track_b_results,
    export_track_b_analysis,
)
from src.evaluation.track_b_config import (
    load_track_b_configuration,
)


CONFIG_PATH = Path(
    "configs/"
    "track_b_unseen_family_evaluation_v0.2.2.yaml"
)
HASH_PATH = Path(
    "configs/"
    "track_b_unseen_family_evaluation_v0.2.2.sha256"
)
RESULTS_DIRECTORY = Path(
    "results/v0.2.2/track_b"
)


@pytest.fixture(scope="module")
def configuration():
    return load_track_b_configuration(
        CONFIG_PATH,
        HASH_PATH,
    )


@pytest.fixture(scope="module")
def analysis(configuration):
    return analyze_track_b_results(
        RESULTS_DIRECTORY,
        configuration=configuration,
    )


def test_all_locked_runs_are_analyzed(
    analysis,
) -> None:
    assert analysis["independent_runs"] == 10
    assert analysis["run_ids"] == [
        f"track-b-seed-{number:02d}"
        for number in range(1, 11)
    ]
    assert (
        analysis["governance"][
            "all_hashes_verified"
        ]
        is True
    )


def test_confirmatory_criterion_is_supported(
    analysis,
) -> None:
    criterion = analysis[
        "confirmatory_criterion"
    ]

    assert (
        criterion[
            "calibration_condition_met"
        ]
        is True
    )
    assert (
        criterion[
            "abstention_condition_met"
        ]
        is True
    )
    assert (
        criterion["criterion_supported"]
        is True
    )
    assert (
        criterion[
            "minimum_validation_coverage"
        ]
        == 0.95
    )
    assert (
        criterion[
            "minimum_ood_abstention_rate"
        ]
        == 0.80
    )


@pytest.mark.parametrize(
    ("tier", "expected_mean"),
    (
        ("mild", 0.96145),
        ("moderate", 0.9942),
        ("severe", 0.99915),
    ),
)
def test_abstention_summary(
    analysis,
    tier,
    expected_mean,
) -> None:
    tier_payload = analysis[
        "selective_safety"
    ][tier]

    assert (
        tier_payload[
            "criterion_met_run_count"
        ]
        == 10
    )
    assert (
        tier_payload[
            "criterion_met_all_runs"
        ]
        is True
    )
    assert (
        tier_payload["abstention_rate"][
            "mean"
        ]
        == pytest.approx(expected_mean)
    )
    assert (
        tier_payload["abstention_rate"][
            "minimum"
        ]
        >= 0.80
    )


def test_undefined_selective_errors_are_explicit(
    analysis,
) -> None:
    mild = analysis[
        "selective_safety"
    ]["mild"]["selective_error_rate"]
    moderate = analysis[
        "selective_safety"
    ]["moderate"]["selective_error_rate"]
    severe = analysis[
        "selective_safety"
    ]["severe"]["selective_error_rate"]

    assert mild["defined_count"] == 10
    assert mild["undefined_count"] == 0
    assert moderate["defined_count"] == 9
    assert moderate["undefined_count"] == 1
    assert severe["defined_count"] == 7
    assert severe["undefined_count"] == 3


def test_governance_is_preserved(
    analysis,
) -> None:
    governance = analysis["governance"]

    assert governance == {
        "all_hashes_verified": True,
        "all_ood_targets_unreachable": True,
        "strict_family_separation_verified": True,
        "oracle_used": False,
        "final_test_tuning_used": False,
        "ood_events_used_for_threshold": False,
        "seen_and_unseen_metrics_pooled": False,
    }


def test_analysis_is_deterministic(
    analysis,
    configuration,
) -> None:
    repeated = analyze_track_b_results(
        RESULTS_DIRECTORY,
        configuration=configuration,
    )

    assert repeated == analysis
    assert analysis["bootstrap"] == {
        "resamples": 10000,
        "seed": 4244,
        "confidence_level": 0.95,
    }


def test_export_is_deterministic(
    analysis,
    tmp_path,
) -> None:
    first = export_track_b_analysis(
        analysis,
        tmp_path / "first.json",
    )
    second = export_track_b_analysis(
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


def test_hash_mismatch_is_rejected(
    configuration,
    tmp_path,
) -> None:
    for run in configuration.runs:
        source_json = (
            RESULTS_DIRECTORY
            / f"{run.run_id}.json"
        )
        source_hash = source_json.with_suffix(
            ".json.sha256"
        )
        target_json = (
            tmp_path / source_json.name
        )
        target_hash = target_json.with_suffix(
            ".json.sha256"
        )

        target_json.symlink_to(
            source_json.resolve()
        )
        target_hash.write_text(
            source_hash.read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )

    bad_hash = (
        tmp_path
        / "track-b-seed-01.json.sha256"
    )
    bad_hash.write_text(
        "0" * 64 + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        TrackBAnalysisError,
        match="SHA-256 mismatch",
    ):
        analyze_track_b_results(
            tmp_path,
            configuration=configuration,
        )
