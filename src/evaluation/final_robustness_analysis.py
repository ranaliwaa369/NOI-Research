"""Aggregate locked final robustness results across runs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class FinalRobustnessAnalysisError(
    ValueError
):
    """Raised when final robustness analysis is invalid."""


@dataclass(frozen=True)
class FinalRobustnessAnalysisExport:
    """Paths and digest for one aggregate export."""

    json_path: Path
    sha256_path: Path
    sha256: str


METRICS = (
    "mean_reciprocal_rank",
    "recall_at_1",
    "recall_at_10",
    "ndcg_at_10",
)


def analyze_final_robustness_results(
    results_directory: str | Path,
    *,
    configuration_path: str | Path,
    configuration_hash_path: str | Path,
) -> dict[str, Any]:
    """Analyze ten locked robustness runs."""

    results_directory = Path(
        results_directory
    )
    configuration_path = Path(
        configuration_path
    )
    configuration_hash_path = Path(
        configuration_hash_path
    )

    configuration_sha256 = (
        _verify_hash_pair(
            configuration_path,
            configuration_hash_path,
        )
    )

    configuration = yaml.safe_load(
        configuration_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(configuration, dict):
        raise FinalRobustnessAnalysisError(
            "Robustness configuration must "
            "be a mapping."
        )

    dataset = _mapping(
        configuration.get("dataset"),
        "dataset",
    )
    systems_payload = _mapping(
        configuration.get("systems"),
        "systems",
    )
    statistics = _mapping(
        configuration.get(
            "statistical_analysis"
        ),
        "statistical_analysis",
    )
    hypothesis = _mapping(
        configuration.get("hypothesis_h4"),
        "hypothesis_h4",
    )
    baseline_rule = _mapping(
        configuration.get(
            "strongest_baseline_rule"
        ),
        "strongest_baseline_rule",
    )

    runs = tuple(
        _mapping(value, "run")
        for value in _sequence(
            dataset.get("runs"),
            "dataset.runs",
        )
    )
    run_ids = tuple(
        _text(run.get("run_id"), "run_id")
        for run in runs
    )

    if len(runs) != 10:
        raise FinalRobustnessAnalysisError(
            "Exactly ten runs are required."
        )

    tiers = tuple(
        _text(value, "severity tier")
        for value in _sequence(
            dataset.get("severity_tiers"),
            "dataset.severity_tiers",
        )
    )

    if tiers != (
        "mild",
        "moderate",
        "severe",
    ):
        raise FinalRobustnessAnalysisError(
            "Severity tiers differ from "
            "the locked order."
        )

    system_names = tuple(systems_payload)

    if system_names != (
        "ridge_only",
        "memory_only",
        "hybrid_without_temporal_decay",
        "full_hybrid",
    ):
        raise FinalRobustnessAnalysisError(
            "Systems differ from the locked order."
        )

    eligible_baselines = tuple(
        _text(value, "eligible baseline")
        for value in _sequence(
            baseline_rule.get(
                "eligible_systems"
            ),
            "eligible_systems",
        )
    )
    tie_break_order = tuple(
        _text(value, "tie-break system")
        for value in _sequence(
            baseline_rule.get(
                "tie_break_order"
            ),
            "tie_break_order",
        )
    )

    if (
        eligible_baselines
        != tie_break_order
        or eligible_baselines
        != system_names[:3]
    ):
        raise FinalRobustnessAnalysisError(
            "Strongest-baseline rule is invalid."
        )

    missing_payload = _mapping(
        configuration.get("missing_modality"),
        "missing_modality",
    )
    temporal_payload = _mapping(
        configuration.get(
            "temporal_displacement"
        ),
        "temporal_displacement",
    )

    condition_specs: list[
        tuple[
            str,
            str,
            tuple[str, ...],
            int,
        ]
    ] = []

    for value in _sequence(
        missing_payload.get("conditions"),
        "missing_modality.conditions",
    ):
        condition = _mapping(
            value,
            "missing-modality condition",
        )
        condition_specs.append(
            (
                "missing_modality",
                _text(
                    condition.get(
                        "condition_id"
                    ),
                    "condition_id",
                ),
                tuple(
                    _text(item, "modality")
                    for item in _sequence(
                        condition.get(
                            "missing_modalities"
                        ),
                        "missing_modalities",
                    )
                ),
                0,
            )
        )

    for value in _sequence(
        temporal_payload.get("days"),
        "temporal_displacement.days",
    ):
        days = _integer(
            value,
            "temporal displacement",
        )
        condition_specs.append(
            (
                "temporal_displacement",
                f"day-{days}",
                (),
                days,
            )
        )

    if len(condition_specs) != 12:
        raise FinalRobustnessAnalysisError(
            "Exactly twelve axis conditions "
            "are required."
        )

    bootstrap_resamples = _integer(
        statistics.get("bootstrap_resamples"),
        "bootstrap_resamples",
    )
    bootstrap_seed = _integer(
        statistics.get("bootstrap_seed"),
        "bootstrap_seed",
    )
    confidence_level = _number(
        statistics.get("confidence_level"),
        "confidence_level",
    )

    if (
        bootstrap_resamples != 10_000
        or bootstrap_seed != 4245
        or confidence_level != 0.95
    ):
        raise FinalRobustnessAnalysisError(
            "Bootstrap configuration is not locked."
        )

    results_by_run: dict[
        str,
        dict[
            tuple[str, str, str, str],
            dict[str, Any],
        ],
    ] = {}

    for run in runs:
        run_id = _text(
            run.get("run_id"),
            "run_id",
        )
        json_path = (
            results_directory
            / f"{run_id}.json"
        )
        hash_path = Path(
            f"{json_path}.sha256"
        )

        _verify_hash_pair(
            json_path,
            hash_path,
        )

        payload = json.loads(
            json_path.read_text(
                encoding="utf-8"
            )
        )
        payload = _mapping(
            payload,
            f"result {run_id}",
        )

        if payload.get("run_id") != run_id:
            raise FinalRobustnessAnalysisError(
                f"Run ID mismatch: {run_id}"
            )

        expected_seeds = {
            "generator_seed": _integer(
                run.get("generator_seed"),
                "generator_seed",
            ),
            "ood_seed": _integer(
                run.get("ood_seed"),
                "ood_seed",
            ),
        }

        if payload.get("seeds") != expected_seeds:
            raise FinalRobustnessAnalysisError(
                f"Seed mismatch: {run_id}"
            )

        if payload.get(
            "protocol_sha256"
        ) != configuration_sha256:
            raise FinalRobustnessAnalysisError(
                f"Protocol mismatch: {run_id}"
            )

        counts = _mapping(
            payload.get("counts"),
            f"counts {run_id}",
        )

        if counts.get("evaluations") != 144:
            raise FinalRobustnessAnalysisError(
                f"Incomplete grid: {run_id}"
            )

        governance = _mapping(
            payload.get("governance"),
            f"governance {run_id}",
        )

        if (
            governance.get("oracle_used")
            is not False
            or governance.get(
                "ood_tuning_used"
            )
            is not False
            or governance.get(
                "final_test_tuning_used"
            )
            is not False
            or governance.get(
                "all_ood_targets_unreachable"
            )
            is not True
            or governance.get(
                "strict_family_separation_verified"
            )
            is not True
        ):
            raise FinalRobustnessAnalysisError(
                f"Governance failure: {run_id}"
            )

        evaluations = _sequence(
            payload.get("evaluations"),
            f"evaluations {run_id}",
        )
        indexed: dict[
            tuple[str, str, str, str],
            dict[str, Any],
        ] = {}

        for raw in evaluations:
            evaluation = _mapping(
                raw,
                "evaluation",
            )
            key = (
                _text(
                    evaluation.get("axis"),
                    "axis",
                ),
                _text(
                    evaluation.get(
                        "condition_id"
                    ),
                    "condition_id",
                ),
                _text(
                    evaluation.get("tier"),
                    "tier",
                ),
                _text(
                    evaluation.get("system"),
                    "system",
                ),
            )

            if key in indexed:
                raise FinalRobustnessAnalysisError(
                    f"Duplicate evaluation: "
                    f"{run_id} {key}"
                )

            if (
                evaluation.get("event_count")
                != 2000
            ):
                raise FinalRobustnessAnalysisError(
                    f"Event count mismatch: "
                    f"{run_id} {key}"
                )

            reciprocal_ranks = _sequence(
                evaluation.get(
                    "reciprocal_ranks"
                ),
                "reciprocal_ranks",
            )

            if len(reciprocal_ranks) != 2000:
                raise FinalRobustnessAnalysisError(
                    f"Paired event count mismatch: "
                    f"{run_id} {key}"
                )

            for metric in METRICS:
                _probability(
                    evaluation.get(metric),
                    metric,
                )

            indexed[key] = evaluation

        expected_keys = {
            (
                axis,
                condition_id,
                tier,
                system,
            )
            for (
                axis,
                condition_id,
                _,
                _,
            ) in condition_specs
            for tier in tiers
            for system in system_names
        }

        if set(indexed) != expected_keys:
            raise FinalRobustnessAnalysisError(
                f"Evaluation grid differs from "
                f"the protocol: {run_id}"
            )

        results_by_run[run_id] = indexed

    rng = np.random.default_rng(
        bootstrap_seed
    )
    condition_results = []
    supported_count = 0

    for (
        axis,
        condition_id,
        missing_modalities,
        displacement_days,
    ) in condition_specs:
        for tier in tiers:
            system_summaries = {}

            for system in system_names:
                metric_summaries = {}

                for metric in METRICS:
                    values = tuple(
                        _probability(
                            results_by_run[
                                run_id
                            ][
                                (
                                    axis,
                                    condition_id,
                                    tier,
                                    system,
                                )
                            ][metric],
                            metric,
                        )
                        for run_id in run_ids
                    )
                    metric_summaries[
                        metric
                    ] = _summary(
                        run_ids,
                        values,
                    )

                system_summaries[
                    system
                ] = metric_summaries

            baseline_means = {
                system: system_summaries[
                    system
                ][
                    "mean_reciprocal_rank"
                ]["mean"]
                for system
                in eligible_baselines
            }

            strongest_baseline = max(
                eligible_baselines,
                key=lambda system: (
                    baseline_means[system],
                    -tie_break_order.index(
                        system
                    ),
                ),
            )

            full_values = tuple(
                system_summaries[
                    "full_hybrid"
                ][
                    "mean_reciprocal_rank"
                ]["values_by_run"][
                    run_id
                ]
                for run_id in run_ids
            )
            baseline_values = tuple(
                system_summaries[
                    strongest_baseline
                ][
                    "mean_reciprocal_rank"
                ]["values_by_run"][
                    run_id
                ]
                for run_id in run_ids
            )
            differences = np.asarray(
                [
                    full - baseline
                    for full, baseline
                    in zip(
                        full_values,
                        baseline_values,
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )

            mean_advantage = float(
                np.mean(differences)
            )
            lower, upper = (
                _paired_bootstrap_interval(
                    differences,
                    rng=rng,
                    resamples=(
                        bootstrap_resamples
                    ),
                    confidence_level=(
                        confidence_level
                    ),
                )
            )
            supported = bool(
                mean_advantage > 0.0
                and lower > 0.0
            )

            if supported:
                supported_count += 1

            condition_results.append(
                {
                    "axis": axis,
                    "condition_id": (
                        condition_id
                    ),
                    "tier": tier,
                    "missing_modalities": list(
                        missing_modalities
                    ),
                    "temporal_displacement_days": (
                        displacement_days
                    ),
                    "systems": system_summaries,
                    "strongest_baseline": {
                        "system": (
                            strongest_baseline
                        ),
                        "selection_metric": (
                            "mean_reciprocal_rank"
                        ),
                        "across_run_means": (
                            baseline_means
                        ),
                    },
                    "paired_mrr_advantage": {
                        "direction": (
                            "full_hybrid minus "
                            "strongest_baseline"
                        ),
                        "differences_by_run": {
                            run_id: float(value)
                            for run_id, value
                            in zip(
                                run_ids,
                                differences,
                                strict=True,
                            )
                        },
                        "mean": mean_advantage,
                        "confidence_interval": {
                            "confidence_level": (
                                confidence_level
                            ),
                            "lower": lower,
                            "upper": upper,
                            "method": (
                                "paired percentile "
                                "bootstrap over run seeds"
                            ),
                            "resamples": (
                                bootstrap_resamples
                            ),
                            "seed": bootstrap_seed,
                        },
                        "mean_positive": bool(
                            mean_advantage > 0.0
                        ),
                        "lower_bound_above_zero": (
                            bool(lower > 0.0)
                        ),
                        "condition_supported": (
                            supported
                        ),
                    },
                }
            )

    total_conditions = len(
        condition_results
    )
    h4_supported = bool(
        supported_count == total_conditions
    )

    return {
        "artifact_type": (
            "final_robustness_aggregate"
        ),
        "schema_version": "1.0",
        "version": "0.2.3",
        "protocol_sha256": (
            configuration_sha256
        ),
        "independent_runs": len(run_ids),
        "run_ids": list(run_ids),
        "analysis_unit": (
            "independent run seed"
        ),
        "paired_analysis_unit": (
            "run_id and latent_event_id"
        ),
        "primary_metric": (
            "mean_reciprocal_rank"
        ),
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "confidence_level": (
                confidence_level
            ),
            "p_values_reported": False,
            "family_wise_error_control_claimed": (
                False
            ),
        },
        "strongest_baseline_rule": {
            "eligible_systems": list(
                eligible_baselines
            ),
            "selection_unit": (
                baseline_rule.get(
                    "selection_unit"
                )
            ),
            "selection_statistic": (
                baseline_rule.get(
                    "selection_statistic"
                )
            ),
            "tie_break_order": list(
                tie_break_order
            ),
        },
        "condition_results": (
            condition_results
        ),
        "hypothesis_h4": {
            "direction": hypothesis.get(
                "direction"
            ),
            "total_condition_tier_tests": (
                total_conditions
            ),
            "supported_condition_tier_tests": (
                supported_count
            ),
            "unsupported_condition_tier_tests": (
                total_conditions
                - supported_count
            ),
            "all_conditions_supported": (
                h4_supported
            ),
            "hypothesis_supported": (
                h4_supported
            ),
            "partial_support_reported": True,
            "null_and_negative_results_retained": (
                True
            ),
            "failure_does_not_invalidate_other_tracks": (
                True
            ),
        },
        "governance": {
            "oracle_used": False,
            "ood_tuning_used": False,
            "final_test_tuning_used": False,
            "all_runs_reported": True,
            "all_conditions_reported": True,
            "all_failures_retained": True,
        },
        "interpretation": {
            "evidence_type": (
                "Synthetic computational "
                "robustness evidence."
            ),
            "not_a_safety_certification": True,
            "not_human_or_animal_olfactory_equivalence": (
                True
            ),
            "track_a_and_track_b_remain_separate": (
                True
            ),
        },
    }


def export_final_robustness_analysis(
    analysis: dict[str, Any],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> FinalRobustnessAnalysisExport:
    """Export deterministic JSON and SHA-256."""

    output_path = Path(output_path)

    if output_path.suffix != ".json":
        raise FinalRobustnessAnalysisError(
            "Aggregate output must end in .json."
        )

    hash_path = Path(
        f"{output_path}.sha256"
    )

    if (
        not overwrite
        and (
            output_path.exists()
            or hash_path.exists()
        )
    ):
        raise FinalRobustnessAnalysisError(
            "Aggregate output already exists."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized = (
        json.dumps(
            analysis,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    output_path.write_text(
        serialized,
        encoding="utf-8",
    )

    digest = sha256(
        output_path.read_bytes()
    ).hexdigest()
    hash_path.write_text(
        digest + "\n",
        encoding="utf-8",
    )

    return FinalRobustnessAnalysisExport(
        json_path=output_path,
        sha256_path=hash_path,
        sha256=digest,
    )


def _verify_hash_pair(
    artifact_path: Path,
    hash_path: Path,
) -> str:
    if not artifact_path.is_file():
        raise FinalRobustnessAnalysisError(
            f"Missing artifact: {artifact_path}"
        )

    if not hash_path.is_file():
        raise FinalRobustnessAnalysisError(
            f"Missing hash: {hash_path}"
        )

    observed = sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    recorded = hash_path.read_text(
        encoding="utf-8"
    ).strip().split()[0]

    if observed != recorded:
        raise FinalRobustnessAnalysisError(
            f"SHA-256 mismatch: {artifact_path}"
        )

    return observed


def _summary(
    run_ids: tuple[str, ...],
    values: tuple[float, ...],
) -> dict[str, Any]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "values_by_run": {
            run_id: float(value)
            for run_id, value
            in zip(
                run_ids,
                array,
                strict=True,
            )
        },
        "mean": float(np.mean(array)),
        "sample_standard_deviation": float(
            np.std(array, ddof=1)
        ),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
    confidence_level: float,
) -> tuple[float, float]:
    indices = rng.integers(
        0,
        len(differences),
        size=(
            resamples,
            len(differences),
        ),
    )
    bootstrap_means = np.mean(
        differences[indices],
        axis=1,
    )
    tail = (
        1.0 - confidence_level
    ) / 2.0
    lower, upper = np.quantile(
        bootstrap_means,
        (tail, 1.0 - tail),
    )

    return float(lower), float(upper)


def _mapping(
    value: Any,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalRobustnessAnalysisError(
            f"{name} must be a mapping."
        )
    return value


def _sequence(
    value: Any,
    name: str,
) -> tuple[Any, ...]:
    if (
        not isinstance(value, (list, tuple))
    ):
        raise FinalRobustnessAnalysisError(
            f"{name} must be a sequence."
        )
    return tuple(value)


def _text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise FinalRobustnessAnalysisError(
            f"{name} must be nonempty text."
        )
    return value


def _integer(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise FinalRobustnessAnalysisError(
            f"{name} must be an integer."
        )
    return value


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise FinalRobustnessAnalysisError(
            f"{name} must be numeric."
        )
    return float(value)


def _probability(
    value: Any,
    name: str,
) -> float:
    converted = _number(value, name)

    if not 0.0 <= converted <= 1.0:
        raise FinalRobustnessAnalysisError(
            f"{name} must be in [0, 1]."
        )

    return converted
