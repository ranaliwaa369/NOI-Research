"""Aggregate analysis for repeated Track B results."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any, Iterable

from src.evaluation.track_b_config import (
    TrackBConfiguration,
)


class TrackBAnalysisError(ValueError):
    """Raised when Track B results fail validation."""


@dataclass(frozen=True)
class TrackBAnalysisExport:
    """Exported aggregate artifact metadata."""

    json_path: Path
    sha256_path: Path
    sha256: str


METRICS = (
    "recall_at_1",
    "recall_at_10",
    "mean_reciprocal_rank",
    "ndcg_at_10",
)


def analyze_track_b_results(
    results_directory: str | Path,
    *,
    configuration: TrackBConfiguration,
) -> dict[str, Any]:
    """Verify and aggregate all locked Track B runs."""

    directory = Path(results_directory)
    payloads: list[dict[str, Any]] = []

    for run in configuration.runs:
        json_path = directory / f"{run.run_id}.json"
        hash_path = json_path.with_suffix(
            ".json.sha256"
        )

        if not json_path.is_file():
            raise TrackBAnalysisError(
                f"Missing result: {run.run_id}."
            )

        if not hash_path.is_file():
            raise TrackBAnalysisError(
                f"Missing hash: {run.run_id}."
            )

        observed = hashlib.sha256(
            json_path.read_bytes()
        ).hexdigest()
        recorded = hash_path.read_text(
            encoding="utf-8"
        ).strip()

        if observed != recorded:
            raise TrackBAnalysisError(
                f"SHA-256 mismatch: {run.run_id}."
            )

        try:
            payload = json.loads(
                json_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise TrackBAnalysisError(
                f"Invalid JSON: {run.run_id}."
            ) from exc

        _validate_run(
            payload,
            run_id=run.run_id,
            generator_seed=run.generator_seed,
            ood_seed=run.ood_seed,
            configuration=configuration,
        )
        payloads.append(payload)

    if len(payloads) != (
        configuration.independent_run_count
    ):
        raise TrackBAnalysisError(
            "Independent run count does not match."
        )

    calibration_coverages = tuple(
        _number(
            payload["calibration"][
                "achieved_reachable_coverage"
            ],
            "calibration coverage",
        )
        for payload in payloads
    )
    thresholds = tuple(
        _number(
            payload["calibration"]["threshold"],
            "calibration threshold",
        )
        for payload in payloads
    )

    selective: dict[str, Any] = {}
    full_noi: dict[str, Any] = {}
    memory_only: dict[str, Any] = {}

    for tier in configuration.severity_tiers:
        tier_selective = tuple(
            _item_for_tier(
                payload["selective_evaluations"],
                tier,
            )
            for payload in payloads
        )
        tier_full = tuple(
            _item_for_tier(
                payload["full_noi_evaluations"],
                tier,
            )
            for payload in payloads
        )
        tier_memory = tuple(
            _item_for_tier(
                payload["memory_only_evaluations"],
                tier,
            )
            for payload in payloads
        )

        selective[tier] = {
            metric: _summary(
                tuple(
                    _number(item[metric], metric)
                    for item in tier_selective
                ),
                configuration=configuration,
                label=f"selective:{tier}:{metric}",
            )
            for metric in (
                "coverage",
                "abstention_rate",
                "false_support_rate",
            )
        }
        selective[tier][
            "selective_error_rate"
        ] = _optional_summary(
            tuple(
                item["selective_error_rate"]
                for item in tier_selective
            ),
            configuration=configuration,
            label=(
                f"selective:{tier}:"
                "selective_error_rate"
            ),
        )
        selective[tier][
            "criterion_met_run_count"
        ] = sum(
            item["abstention_criterion_met"]
            is True
            for item in tier_selective
        )
        selective[tier][
            "criterion_met_all_runs"
        ] = all(
            item["abstention_criterion_met"]
            is True
            for item in tier_selective
        )

        full_noi[tier] = {
            metric: _summary(
                tuple(
                    _number(item[metric], metric)
                    for item in tier_full
                ),
                configuration=configuration,
                label=f"full:{tier}:{metric}",
            )
            for metric in METRICS
        }
        memory_only[tier] = {
            metric: _summary(
                tuple(
                    _number(item[metric], metric)
                    for item in tier_memory
                ),
                configuration=configuration,
                label=f"memory:{tier}:{metric}",
            )
            for metric in METRICS
        }

    baselines: dict[str, Any] = {}

    baseline_names = sorted({
        item["baseline"]
        for payload in payloads
        for item in payload[
            "graded_baselines"
        ]["evaluations"]
    })

    for baseline in baseline_names:
        baselines[baseline] = {}

        for tier in configuration.severity_tiers:
            items = tuple(
                _baseline_item(
                    payload[
                        "graded_baselines"
                    ]["evaluations"],
                    baseline,
                    tier,
                )
                for payload in payloads
            )
            baselines[baseline][tier] = {
                metric: _summary(
                    tuple(
                        _number(
                            item[metric],
                            metric,
                        )
                        for item in items
                    ),
                    configuration=configuration,
                    label=(
                        f"baseline:{baseline}:"
                        f"{tier}:{metric}"
                    ),
                )
                for metric in METRICS
            }

    calibration_met = all(
        value
        >= configuration.minimum_reachable_coverage
        for value in calibration_coverages
    )
    abstention_met = all(
        selective[tier][
            "criterion_met_all_runs"
        ]
        for tier in configuration.severity_tiers
    )

    return {
        "artifact_type": (
            "track_b_unseen_family_aggregate"
        ),
        "schema_version": "1.0",
        "protocol_sha256": (
            configuration.configuration_sha256
        ),
        "version": configuration.version,
        "track": configuration.track,
        "independent_runs": len(payloads),
        "run_ids": [
            payload["run_id"]
            for payload in payloads
        ],
        "bootstrap": {
            "resamples": (
                configuration.bootstrap_resamples
            ),
            "seed": configuration.bootstrap_seed,
            "confidence_level": 0.95,
        },
        "governance": {
            "all_hashes_verified": True,
            "all_ood_targets_unreachable": True,
            "strict_family_separation_verified": True,
            "oracle_used": False,
            "final_test_tuning_used": False,
            "ood_events_used_for_threshold": False,
            "seen_and_unseen_metrics_pooled": False,
        },
        "calibration": {
            "minimum_reachable_coverage": (
                configuration
                .minimum_reachable_coverage
            ),
            "achieved_coverage": _summary(
                calibration_coverages,
                configuration=configuration,
                label="calibration:coverage",
            ),
            "threshold": _summary(
                thresholds,
                configuration=configuration,
                label="calibration:threshold",
            ),
            "condition_met_all_runs": (
                calibration_met
            ),
        },
        "selective_safety": selective,
        "retrieval": {
            "full_noi": full_noi,
            "memory_only": memory_only,
            "baselines": baselines,
        },
        "selected_alpha_counts": _alpha_counts(
            payloads
        ),
        "confirmatory_criterion": {
            "minimum_validation_coverage": (
                configuration
                .minimum_reachable_coverage
            ),
            "minimum_ood_abstention_rate": (
                configuration
                .minimum_ood_abstention_rate
            ),
            "calibration_condition_met": (
                calibration_met
            ),
            "abstention_condition_met": (
                abstention_met
            ),
            "require_both_conditions": True,
            "criterion_supported": (
                calibration_met
                and abstention_met
            ),
            "interpretation": (
                "Prespecified synthetic engineering "
                "criterion only; this is not a "
                "clinical, biological, or deployment-"
                "safety certification."
            ),
        },
    }


def export_track_b_analysis(
    analysis: dict[str, Any],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> TrackBAnalysisExport:
    """Write deterministic aggregate JSON and hash."""

    json_path = Path(output_path)

    if json_path.suffix.lower() != ".json":
        raise TrackBAnalysisError(
            "Aggregate output must use .json."
        )

    hash_path = json_path.with_suffix(
        ".json.sha256"
    )

    if not overwrite and (
        json_path.exists()
        or hash_path.exists()
    ):
        raise TrackBAnalysisError(
            "Aggregate artifact already exists."
        )

    encoded = (
        json.dumps(
            analysis,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    json_path.write_bytes(encoded)
    hash_path.write_text(
        digest + "\n",
        encoding="utf-8",
    )

    return TrackBAnalysisExport(
        json_path=json_path,
        sha256_path=hash_path,
        sha256=digest,
    )


def _validate_run(
    payload: dict[str, Any],
    *,
    run_id: str,
    generator_seed: int,
    ood_seed: int,
    configuration: TrackBConfiguration,
) -> None:
    if payload.get("run_id") != run_id:
        raise TrackBAnalysisError(
            f"Run ID mismatch: {run_id}."
        )

    if payload.get("protocol_sha256") != (
        configuration.configuration_sha256
    ):
        raise TrackBAnalysisError(
            f"Protocol SHA-256 mismatch: {run_id}."
        )

    if payload.get("seeds") != {
        "generator_seed": generator_seed,
        "ood_seed": ood_seed,
    }:
        raise TrackBAnalysisError(
            f"Seed mismatch: {run_id}."
        )

    governance = payload.get("governance", {})

    required = {
        "oracle_used": False,
        "final_test_tuning_used": False,
        "target_identifier_used_in_support": False,
        "family_identifier_used_in_support": False,
        "strict_family_separation_verified": True,
        "all_ood_targets_unreachable": True,
    }

    for key, expected in required.items():
        if governance.get(key) is not expected:
            raise TrackBAnalysisError(
                f"Governance failure "
                f"{key}: {run_id}."
            )

    reachability = payload.get(
        "reachability",
        {},
    )

    if (
        reachability.get(
            "reachable_target_fraction"
        )
        != 0.0
        or reachability.get(
            "reachable_event_fraction"
        )
        != 0.0
    ):
        raise TrackBAnalysisError(
            f"Reachability failure: {run_id}."
        )


def _summary(
    values: tuple[float, ...],
    *,
    configuration: TrackBConfiguration,
    label: str,
) -> dict[str, Any]:
    if not values:
        raise TrackBAnalysisError(
            f"No values for {label}."
        )

    boot_seed = (
        configuration.bootstrap_seed
        + int.from_bytes(
            hashlib.sha256(
                label.encode("utf-8")
            ).digest()[:8],
            "big",
        )
    )
    rng = random.Random(boot_seed)
    size = len(values)
    means = sorted(
        fmean(
            values[
                rng.randrange(size)
            ]
            for _ in range(size)
        )
        for _ in range(
            configuration.bootstrap_resamples
        )
    )

    return {
        "values": list(values),
        "mean": fmean(values),
        "median": median(values),
        "standard_deviation": (
            stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "minimum": min(values),
        "maximum": max(values),
        "confidence_interval": {
            "level": 0.95,
            "lower": _percentile(
                means,
                0.025,
            ),
            "upper": _percentile(
                means,
                0.975,
            ),
            "method": (
                "percentile bootstrap of "
                "the across-run mean"
            ),
        },
    }


def _optional_summary(
    values: tuple[Any, ...],
    *,
    configuration: TrackBConfiguration,
    label: str,
) -> dict[str, Any]:
    """Summarize defined values without imputing None."""

    defined = tuple(
        _number(value, label)
        for value in values
        if value is not None
    )
    undefined_count = (
        len(values) - len(defined)
    )

    return {
        "defined_count": len(defined),
        "undefined_count": undefined_count,
        "undefined_reason": (
            "No supported events were available "
            "for a selective-error calculation."
        ),
        "summary": (
            _summary(
                defined,
                configuration=configuration,
                label=label,
            )
            if defined
            else None
        ),
    }


def _percentile(
    values: list[float],
    probability: float,
) -> float:
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(
        lower + 1,
        len(values) - 1,
    )
    fraction = position - lower

    return (
        values[lower] * (1.0 - fraction)
        + values[upper] * fraction
    )


def _item_for_tier(
    items: Iterable[dict[str, Any]],
    tier: str,
) -> dict[str, Any]:
    matches = [
        item
        for item in items
        if item.get("tier") == tier
    ]

    if len(matches) != 1:
        raise TrackBAnalysisError(
            f"Expected one result for tier {tier}."
        )

    return matches[0]


def _baseline_item(
    items: Iterable[dict[str, Any]],
    baseline: str,
    tier: str,
) -> dict[str, Any]:
    matches = [
        item
        for item in items
        if (
            item.get("baseline") == baseline
            and item.get("tier") == tier
        )
    ]

    if len(matches) != 1:
        raise TrackBAnalysisError(
            "Expected one baseline/tier result."
        )

    return matches[0]


def _number(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TrackBAnalysisError(
            f"{name} must be numeric."
        )

    converted = float(value)

    if not isfinite(converted):
        raise TrackBAnalysisError(
            f"{name} must be finite."
        )

    return converted


def _alpha_counts(
    payloads: Iterable[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for payload in payloads:
        key = str(
            payload["selected_hybrid_alpha"]
        )
        counts[key] = counts.get(key, 0) + 1

    return dict(sorted(counts.items()))
