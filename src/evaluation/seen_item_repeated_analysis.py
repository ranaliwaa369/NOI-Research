"""Aggregate analysis for repeated NOI Track A results."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any, Mapping

import numpy as np

from src.evaluation.seen_item_repeated_config import (
    RepeatedSeedSpec,
    SeenItemRepeatedConfig,
)


SYSTEMS = (
    "memory_only",
    "ridge_only",
    "hybrid",
)

METRICS = (
    "recall_at_1",
    "recall_at_10",
    "mean_reciprocal_rank",
    "ndcg_at_10",
)


class RepeatedTrackAAnalysisError(ValueError):
    """Raised when repeated Track A results are invalid."""


@dataclass(frozen=True, slots=True)
class SystemMetricSummary:
    """Seed-level descriptive statistics for one metric."""

    system: str
    metric: str
    values: tuple[float, ...]
    count: int
    mean: float
    median: float
    standard_deviation: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.system not in SYSTEMS:
            raise RepeatedTrackAAnalysisError(
                "Unknown system in metric summary."
            )

        if self.metric not in METRICS:
            raise RepeatedTrackAAnalysisError(
                "Unknown metric in metric summary."
            )

        if self.count != len(self.values):
            raise RepeatedTrackAAnalysisError(
                "Summary count is inconsistent."
            )

        if self.count < 2:
            raise RepeatedTrackAAnalysisError(
                "At least two seeds are required."
            )


@dataclass(frozen=True, slots=True)
class PairedMetricComparison:
    """Prespecified paired seed-level comparison."""

    left_system: str
    right_system: str
    direction: str
    metric: str
    differences: tuple[float, ...]
    mean_difference: float
    median_difference: float
    standard_deviation: float
    minimum_difference: float
    maximum_difference: float
    confidence_level: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    wins: int
    ties: int
    losses: int

    def __post_init__(self) -> None:
        if (
            self.left_system not in SYSTEMS
            or self.right_system not in SYSTEMS
        ):
            raise RepeatedTrackAAnalysisError(
                "Comparison contains an unknown system."
            )

        if self.left_system == self.right_system:
            raise RepeatedTrackAAnalysisError(
                "Comparison systems must differ."
            )

        if self.metric not in METRICS:
            raise RepeatedTrackAAnalysisError(
                "Comparison metric is unknown."
            )

        if not self.differences:
            raise RepeatedTrackAAnalysisError(
                "Comparison requires paired differences."
            )

        if (
            self.wins
            + self.ties
            + self.losses
            != len(self.differences)
        ):
            raise RepeatedTrackAAnalysisError(
                "Win, tie, and loss counts are inconsistent."
            )


@dataclass(frozen=True, slots=True)
class RepeatedTrackAAnalysis:
    """Verified aggregate analysis across locked runs."""

    run_ids: tuple[str, ...]
    source_sha256: tuple[str, ...]
    reachable_event_fractions: tuple[float, ...]
    selected_hybrid_alphas: tuple[float, ...]
    system_summaries: tuple[SystemMetricSummary, ...]
    primary_comparison: PairedMetricComparison
    oracle_used: bool
    final_test_tuning_used: bool
    confidence_level: float
    bootstrap_seed: int
    bootstrap_resamples: int
    repeated_protocol_sha256: str

    def __post_init__(self) -> None:
        run_count = len(self.run_ids)

        if run_count < 2:
            raise RepeatedTrackAAnalysisError(
                "At least two runs are required."
            )

        sequence_lengths = (
            len(self.source_sha256),
            len(self.reachable_event_fractions),
            len(self.selected_hybrid_alphas),
        )

        if any(
            length != run_count
            for length in sequence_lengths
        ):
            raise RepeatedTrackAAnalysisError(
                "Run-level sequences are inconsistent."
            )

        if self.oracle_used is not False:
            raise RepeatedTrackAAnalysisError(
                "Oracle use is prohibited."
            )

        if self.final_test_tuning_used is not False:
            raise RepeatedTrackAAnalysisError(
                "Final-test tuning is prohibited."
            )

    def summary_for(
        self,
        system: str,
        metric: str,
    ) -> SystemMetricSummary:
        """Return one system-metric summary."""

        for summary in self.system_summaries:
            if (
                summary.system == system
                and summary.metric == metric
            ):
                return summary

        raise RepeatedTrackAAnalysisError(
            f"No summary exists for {system}/{metric}."
        )


def analyze_repeated_track_a(
    results_directory: str | Path,
    *,
    repeated_config: SeenItemRepeatedConfig,
    repeated_protocol_sha256: str,
) -> RepeatedTrackAAnalysis:
    """Verify and aggregate all prespecified seed runs."""

    if not isinstance(
        repeated_config,
        SeenItemRepeatedConfig,
    ):
        raise RepeatedTrackAAnalysisError(
            "repeated_config has an invalid type."
        )

    _validate_digest(
        repeated_protocol_sha256,
        name="Repeated protocol SHA-256",
    )

    results_dir = Path(results_directory)

    if not results_dir.is_dir():
        raise RepeatedTrackAAnalysisError(
            f"Results directory not found: {results_dir}"
        )

    run_ids = []
    source_hashes = []
    reachable_fractions = []
    hybrid_alphas = []

    metric_values = {
        (system, metric): []
        for system in SYSTEMS
        for metric in METRICS
    }

    for run_spec in repeated_config.runs:
        payload, source_digest = (
            _load_verified_run(
                results_dir,
                run_spec=run_spec,
                repeated_protocol_sha256=(
                    repeated_protocol_sha256
                ),
            )
        )

        run_ids.append(run_spec.run_id)
        source_hashes.append(source_digest)

        reachable_fraction = _metric_value(
            payload.get(
                "reachable_event_fraction"
            ),
            name=(
                f"{run_spec.run_id} "
                "reachable_event_fraction"
            ),
        )

        if reachable_fraction != 1.0:
            raise RepeatedTrackAAnalysisError(
                f"{run_spec.run_id} is not fully reachable."
            )

        reachable_fractions.append(
            reachable_fraction
        )

        alpha = _metric_value(
            payload.get("selected_hybrid_alpha"),
            name=(
                f"{run_spec.run_id} "
                "selected_hybrid_alpha"
            ),
        )
        hybrid_alphas.append(alpha)

        if payload.get("oracle_used") is not False:
            raise RepeatedTrackAAnalysisError(
                f"{run_spec.run_id} used an oracle."
            )

        if (
            payload.get("final_test_tuning_used")
            is not False
        ):
            raise RepeatedTrackAAnalysisError(
                f"{run_spec.run_id} used final-test tuning."
            )

        systems = _mapping(
            payload,
            "systems",
            run_id=run_spec.run_id,
        )

        if set(systems) != set(SYSTEMS):
            raise RepeatedTrackAAnalysisError(
                f"{run_spec.run_id} has an invalid system set."
            )

        for system in SYSTEMS:
            system_payload = _mapping(
                systems,
                system,
                run_id=run_spec.run_id,
            )
            metrics = _mapping(
                system_payload,
                "metrics",
                run_id=run_spec.run_id,
            )

            if set(metrics) != set(METRICS):
                raise RepeatedTrackAAnalysisError(
                    f"{run_spec.run_id}/{system} "
                    "has an invalid metric set."
                )

            for metric in METRICS:
                metric_values[
                    (system, metric)
                ].append(
                    _metric_value(
                        metrics.get(metric),
                        name=(
                            f"{run_spec.run_id}/"
                            f"{system}/{metric}"
                        ),
                    )
                )

    expected_run_ids = tuple(
        run.run_id
        for run in repeated_config.runs
    )

    if tuple(run_ids) != expected_run_ids:
        raise RepeatedTrackAAnalysisError(
            "Analyzed runs differ from the locked schedule."
        )

    summaries = tuple(
        _summarize(
            system,
            metric,
            tuple(
                metric_values[(system, metric)]
            ),
        )
        for system in SYSTEMS
        for metric in METRICS
    )

    summary_map = {
        (summary.system, summary.metric): summary
        for summary in summaries
    }

    memory_mrr = summary_map[
        ("memory_only", "mean_reciprocal_rank")
    ].values
    ridge_mrr = summary_map[
        ("ridge_only", "mean_reciprocal_rank")
    ].values

    differences = tuple(
        left - right
        for left, right in zip(
            memory_mrr,
            ridge_mrr,
            strict=True,
        )
    )

    interval_lower, interval_upper = (
        _paired_bootstrap_interval(
            differences,
            confidence_level=(
                repeated_config.confidence_level
            ),
            bootstrap_seed=(
                repeated_config.bootstrap_seed
            ),
            bootstrap_resamples=(
                repeated_config.bootstrap_resamples
            ),
        )
    )

    comparison = PairedMetricComparison(
        left_system="memory_only",
        right_system="ridge_only",
        direction=(
            "memory_only minus ridge_only"
        ),
        metric="mean_reciprocal_rank",
        differences=differences,
        mean_difference=fmean(differences),
        median_difference=float(
            median(differences)
        ),
        standard_deviation=stdev(differences),
        minimum_difference=min(differences),
        maximum_difference=max(differences),
        confidence_level=(
            repeated_config.confidence_level
        ),
        confidence_interval_lower=interval_lower,
        confidence_interval_upper=interval_upper,
        wins=sum(
            difference > 0.0
            for difference in differences
        ),
        ties=sum(
            difference == 0.0
            for difference in differences
        ),
        losses=sum(
            difference < 0.0
            for difference in differences
        ),
    )

    return RepeatedTrackAAnalysis(
        run_ids=tuple(run_ids),
        source_sha256=tuple(source_hashes),
        reachable_event_fractions=tuple(
            reachable_fractions
        ),
        selected_hybrid_alphas=tuple(
            hybrid_alphas
        ),
        system_summaries=summaries,
        primary_comparison=comparison,
        oracle_used=False,
        final_test_tuning_used=False,
        confidence_level=(
            repeated_config.confidence_level
        ),
        bootstrap_seed=(
            repeated_config.bootstrap_seed
        ),
        bootstrap_resamples=(
            repeated_config.bootstrap_resamples
        ),
        repeated_protocol_sha256=(
            repeated_protocol_sha256
        ),
    )


def _load_verified_run(
    results_directory: Path,
    *,
    run_spec: RepeatedSeedSpec,
    repeated_protocol_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    json_path = (
        results_directory
        / f"{run_spec.run_id}.json"
    )
    hash_path = json_path.with_suffix(
        json_path.suffix + ".sha256"
    )

    if not json_path.is_file():
        raise RepeatedTrackAAnalysisError(
            f"Result file not found: {json_path}"
        )

    if not hash_path.is_file():
        raise RepeatedTrackAAnalysisError(
            f"Result hash not found: {hash_path}"
        )

    raw_bytes = json_path.read_bytes()
    observed_digest = sha256(
        raw_bytes
    ).hexdigest()

    recorded_text = hash_path.read_text(
        encoding="utf-8"
    ).strip()

    if not recorded_text:
        raise RepeatedTrackAAnalysisError(
            f"Empty result hash: {hash_path}"
        )

    recorded_digest = recorded_text.split()[0]

    _validate_digest(
        recorded_digest,
        name=f"{run_spec.run_id} SHA-256",
    )

    if observed_digest != recorded_digest:
        raise RepeatedTrackAAnalysisError(
            f"{run_spec.run_id} failed SHA-256 verification."
        )

    try:
        payload = json.loads(
            raw_bytes.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise RepeatedTrackAAnalysisError(
            f"{run_spec.run_id} is not valid JSON."
        ) from exc

    if not isinstance(payload, Mapping):
        raise RepeatedTrackAAnalysisError(
            f"{run_spec.run_id} root must be a mapping."
        )

    if payload.get("schema_version") != "0.2.1":
        raise RepeatedTrackAAnalysisError(
            f"{run_spec.run_id} has an invalid schema."
        )

    if (
        payload.get("repeated_protocol_sha256")
        != repeated_protocol_sha256
    ):
        raise RepeatedTrackAAnalysisError(
            f"{run_spec.run_id} protocol SHA-256 "
            "does not match the locked protocol."
        )

    repeated_run = _mapping(
        payload,
        "repeated_run",
        run_id=run_spec.run_id,
    )

    expected_run = {
        "run_id": run_spec.run_id,
        "generator_seed": (
            run_spec.generator_seed
        ),
        "ood_seed": run_spec.ood_seed,
        "partition_seed": (
            run_spec.partition_seed
        ),
    }

    if dict(repeated_run) != expected_run:
        raise RepeatedTrackAAnalysisError(
            f"{run_spec.run_id} seed metadata "
            "does not match the locked schedule."
        )

    if (
        payload.get("evaluation_track")
        != "seen_item_episodic_retrieval"
    ):
        raise RepeatedTrackAAnalysisError(
            f"{run_spec.run_id} has the wrong track."
        )

    if (
        payload.get("evidence_scope")
        != "synthetic_computational_evaluation_only"
    ):
        raise RepeatedTrackAAnalysisError(
            f"{run_spec.run_id} has the wrong evidence scope."
        )

    return payload, observed_digest


def _summarize(
    system: str,
    metric: str,
    values: tuple[float, ...],
) -> SystemMetricSummary:
    return SystemMetricSummary(
        system=system,
        metric=metric,
        values=values,
        count=len(values),
        mean=fmean(values),
        median=float(median(values)),
        standard_deviation=stdev(values),
        minimum=min(values),
        maximum=max(values),
    )


def _paired_bootstrap_interval(
    differences: tuple[float, ...],
    *,
    confidence_level: float,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> tuple[float, float]:
    if len(differences) < 2:
        raise RepeatedTrackAAnalysisError(
            "Paired bootstrap requires at least two seeds."
        )

    rng = np.random.default_rng(
        bootstrap_seed
    )
    values = np.asarray(
        differences,
        dtype=np.float64,
    )

    indices = rng.integers(
        0,
        len(values),
        size=(
            bootstrap_resamples,
            len(values),
        ),
    )

    bootstrap_means = values[indices].mean(
        axis=1
    )

    tail_probability = (
        1.0 - confidence_level
    ) / 2.0

    lower, upper = np.quantile(
        bootstrap_means,
        (
            tail_probability,
            1.0 - tail_probability,
        ),
        method="linear",
    )

    return float(lower), float(upper)


def _mapping(
    parent: Mapping[str, Any],
    key: str,
    *,
    run_id: str,
) -> Mapping[str, Any]:
    value = parent.get(key)

    if not isinstance(value, Mapping):
        raise RepeatedTrackAAnalysisError(
            f"{run_id}/{key} must be a mapping."
        )

    return value


def _metric_value(
    value: Any,
    *,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise RepeatedTrackAAnalysisError(
            f"{name} must be numeric."
        )

    converted = float(value)

    if (
        not isfinite(converted)
        or not 0.0 <= converted <= 1.0
    ):
        raise RepeatedTrackAAnalysisError(
            f"{name} must be in [0, 1]."
        )

    return converted


def _validate_digest(
    value: str,
    *,
    name: str,
) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise RepeatedTrackAAnalysisError(
            f"{name} is invalid."
        )
