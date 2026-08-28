"""Tests for final robustness aggregate analysis."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from src.evaluation.final_robustness_analysis import (
    FinalRobustnessAnalysisError,
    analyze_final_robustness_results,
    export_final_robustness_analysis,
)


RESULTS_DIRECTORY = Path(
    "results/v0.2.3/final_robustness"
)
CONFIGURATION_PATH = Path(
    "configs/"
    "robustness_evaluation_v0.2.3.yaml"
)
CONFIGURATION_HASH_PATH = Path(
    "configs/"
    "robustness_evaluation_v0.2.3.sha256"
)


@pytest.fixture(scope="module")
def analysis():
    return analyze_final_robustness_results(
        RESULTS_DIRECTORY,
        configuration_path=(
            CONFIGURATION_PATH
        ),
        configuration_hash_path=(
            CONFIGURATION_HASH_PATH
        ),
    )


def test_all_runs_and_conditions_are_retained(
    analysis,
) -> None:
    assert analysis["independent_runs"] == 10
    assert len(analysis["run_ids"]) == 10
    assert len(
        analysis["condition_results"]
    ) == 36

    axes = [
        result["axis"]
        for result
        in analysis["condition_results"]
    ]

    assert axes.count(
        "missing_modality"
    ) == 21
    assert axes.count(
        "temporal_displacement"
    ) == 15


def test_strongest_baseline_is_selected_by_rule(
    analysis,
) -> None:
    assert {
        result["strongest_baseline"][
            "system"
        ]
        for result
        in analysis["condition_results"]
    } == {"ridge_only"}


def test_h4_negative_result_is_retained(
    analysis,
) -> None:
    h4 = analysis["hypothesis_h4"]

    assert (
        h4["total_condition_tier_tests"]
        == 36
    )
    assert (
        h4[
            "supported_condition_tier_tests"
        ]
        == 0
    )
    assert (
        h4[
            "unsupported_condition_tier_tests"
        ]
        == 36
    )
    assert (
        h4["hypothesis_supported"]
        is False
    )
    assert (
        h4[
            "null_and_negative_results_retained"
        ]
        is True
    )


def test_every_paired_interval_excludes_zero_negatively(
    analysis,
) -> None:
    for result in analysis[
        "condition_results"
    ]:
        advantage = result[
            "paired_mrr_advantage"
        ]
        interval = advantage[
            "confidence_interval"
        ]

        assert advantage["mean"] < 0.0
        assert interval["upper"] < 0.0
        assert (
            advantage["condition_supported"]
            is False
        )


def test_governance_is_preserved(
    analysis,
) -> None:
    governance = analysis["governance"]

    assert governance["oracle_used"] is False
    assert (
        governance["ood_tuning_used"]
        is False
    )
    assert (
        governance[
            "final_test_tuning_used"
        ]
        is False
    )
    assert (
        governance[
            "all_failures_retained"
        ]
        is True
    )


def test_analysis_is_deterministic(
    analysis,
) -> None:
    repeated = analyze_final_robustness_results(
        RESULTS_DIRECTORY,
        configuration_path=(
            CONFIGURATION_PATH
        ),
        configuration_hash_path=(
            CONFIGURATION_HASH_PATH
        ),
    )

    assert repeated == analysis


def test_export_writes_verified_hash(
    analysis,
    tmp_path,
) -> None:
    output = (
        tmp_path
        / "final_robustness_aggregate.json"
    )

    exported = export_final_robustness_analysis(
        analysis,
        output,
    )

    observed = sha256(
        output.read_bytes()
    ).hexdigest()
    recorded = (
        exported.sha256_path.read_text(
            encoding="utf-8"
        ).strip()
    )

    assert observed == recorded
    assert observed == exported.sha256


def test_corrupt_run_hash_is_rejected(
    tmp_path,
) -> None:
    for number in range(1, 11):
        stem = (
            f"robustness-seed-{number:02d}"
        )
        source_json = (
            RESULTS_DIRECTORY
            / f"{stem}.json"
        )
        source_hash = (
            RESULTS_DIRECTORY
            / f"{stem}.json.sha256"
        )
        target_json = (
            tmp_path
            / f"{stem}.json"
        )
        target_hash = (
            tmp_path
            / f"{stem}.json.sha256"
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

    (
        tmp_path
        / "robustness-seed-01.json.sha256"
    ).write_text(
        "0" * 64 + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FinalRobustnessAnalysisError,
        match="SHA-256 mismatch",
    ):
        analyze_final_robustness_results(
            tmp_path,
            configuration_path=(
                CONFIGURATION_PATH
            ),
            configuration_hash_path=(
                CONFIGURATION_HASH_PATH
            ),
        )
