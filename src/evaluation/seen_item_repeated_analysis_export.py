"""Deterministic aggregate export for repeated Track A."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from src.evaluation.seen_item_repeated_analysis import (
    METRICS,
    SYSTEMS,
    RepeatedTrackAAnalysis,
)


class RepeatedTrackAAnalysisExportError(
    ValueError
):
    """Raised when aggregate results cannot export."""


@dataclass(frozen=True, slots=True)
class RepeatedTrackAAnalysisExport:
    """Paths and digest for an aggregate artifact."""

    json_path: Path
    sha256_path: Path
    sha256: str

    def __post_init__(self) -> None:
        if self.json_path.suffix.lower() != ".json":
            raise RepeatedTrackAAnalysisExportError(
                "json_path must use the .json extension."
            )

        if (
            len(self.sha256) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character in self.sha256
            )
        ):
            raise RepeatedTrackAAnalysisExportError(
                "sha256 must be a lowercase digest."
            )


def build_repeated_track_a_payload(
    analysis: RepeatedTrackAAnalysis,
) -> dict[str, Any]:
    """Build the deterministic aggregate payload."""

    if not isinstance(
        analysis,
        RepeatedTrackAAnalysis,
    ):
        raise RepeatedTrackAAnalysisExportError(
            "analysis must be a "
            "RepeatedTrackAAnalysis."
        )

    systems: dict[str, dict[str, Any]] = {
        system: {}
        for system in SYSTEMS
    }

    for system in SYSTEMS:
        for metric in METRICS:
            summary = analysis.summary_for(
                system,
                metric,
            )

            systems[system][metric] = {
                "values": list(summary.values),
                "count": summary.count,
                "mean": summary.mean,
                "median": summary.median,
                "standard_deviation": (
                    summary.standard_deviation
                ),
                "minimum": summary.minimum,
                "maximum": summary.maximum,
            }

    primary = analysis.primary_comparison

    interval_excludes_zero = (
        primary.confidence_interval_lower > 0.0
        or primary.confidence_interval_upper < 0.0
    )

    memory_mrr = analysis.summary_for(
        "memory_only",
        "mean_reciprocal_rank",
    ).values
    ridge_mrr = analysis.summary_for(
        "ridge_only",
        "mean_reciprocal_rank",
    ).values
    hybrid_mrr = analysis.summary_for(
        "hybrid",
        "mean_reciprocal_rank",
    ).values

    hybrid_minus_memory = tuple(
        hybrid - memory
        for hybrid, memory in zip(
            hybrid_mrr,
            memory_mrr,
            strict=True,
        )
    )
    hybrid_minus_ridge = tuple(
        hybrid - ridge
        for hybrid, ridge in zip(
            hybrid_mrr,
            ridge_mrr,
            strict=True,
        )
    )

    source_artifacts = [
        {
            "run_id": run_id,
            "json_sha256": digest,
            "reachable_event_fraction": (
                reachable_fraction
            ),
            "selected_hybrid_alpha": alpha,
        }
        for (
            run_id,
            digest,
            reachable_fraction,
            alpha,
        ) in zip(
            analysis.run_ids,
            analysis.source_sha256,
            analysis.reachable_event_fractions,
            analysis.selected_hybrid_alphas,
            strict=True,
        )
    ]

    return {
        "schema_version": "0.2.1",
        "artifact_type": (
            "repeated_track_a_aggregate"
        ),
        "evaluation_track": (
            "seen_item_episodic_retrieval"
        ),
        "evidence_scope": (
            "synthetic_computational_evaluation_only"
        ),
        "run_count": len(analysis.run_ids),
        "run_ids": list(analysis.run_ids),
        "repeated_protocol_sha256": (
            analysis.repeated_protocol_sha256
        ),
        "source_artifacts": source_artifacts,
        "controls": {
            "all_targets_reachable": all(
                value == 1.0
                for value in (
                    analysis.reachable_event_fractions
                )
            ),
            "oracle_used": analysis.oracle_used,
            "final_test_tuning_used": (
                analysis.final_test_tuning_used
            ),
            "independent_replication_unit": (
                "seed"
            ),
            "pilot_included": False,
            "systems_use_paired_seed_runs": True,
        },
        "bootstrap": {
            "method": (
                "paired percentile bootstrap "
                "of seed-level mean differences"
            ),
            "confidence_level": (
                analysis.confidence_level
            ),
            "seed": analysis.bootstrap_seed,
            "resamples": (
                analysis.bootstrap_resamples
            ),
        },
        "selected_hybrid_alphas": list(
            analysis.selected_hybrid_alphas
        ),
        "systems": systems,
        "primary_comparison": {
            "status": "prespecified_confirmatory",
            "left_system": primary.left_system,
            "right_system": primary.right_system,
            "direction": primary.direction,
            "metric": primary.metric,
            "differences": list(
                primary.differences
            ),
            "mean_difference": (
                primary.mean_difference
            ),
            "median_difference": (
                primary.median_difference
            ),
            "standard_deviation": (
                primary.standard_deviation
            ),
            "minimum_difference": (
                primary.minimum_difference
            ),
            "maximum_difference": (
                primary.maximum_difference
            ),
            "confidence_interval": {
                "level": primary.confidence_level,
                "lower": (
                    primary.confidence_interval_lower
                ),
                "upper": (
                    primary.confidence_interval_upper
                ),
                "method": (
                    "paired percentile bootstrap"
                ),
            },
            "wins": primary.wins,
            "ties": primary.ties,
            "losses": primary.losses,
            "confirmatory_interval_excludes_zero": (
                interval_excludes_zero
            ),
        },
        "secondary_descriptive_comparisons": {
            "hybrid_minus_memory_mrr": (
                _descriptive_difference(
                    hybrid_minus_memory
                )
            ),
            "hybrid_minus_ridge_mrr": (
                _descriptive_difference(
                    hybrid_minus_ridge
                )
            ),
        },
        "inference": {
            "p_values_reported": False,
            "multiple_testing_applied": False,
            "configured_multiple_testing_method": (
                "holm"
            ),
            "reason": (
                "No null-hypothesis significance test "
                "was prespecified."
            ),
            "secondary_comparisons_are": (
                "descriptive_not_confirmatory"
            ),
        },
        "interpretation": {
            "primary_supported": (
                primary.confidence_interval_lower
                > 0.0
            ),
            "statement": (
                "Across the ten prespecified synthetic "
                "seed runs, memory-only achieved a "
                "positive mean MRR difference relative "
                "to ridge-only, and the prespecified "
                "paired 95% bootstrap confidence "
                "interval excluded zero."
            ),
            "hybrid_statement": (
                "The hybrid system had the highest "
                "descriptive mean MRR, but hybrid "
                "comparisons are not the prespecified "
                "primary confirmatory comparison."
            ),
        },
        "limitations": [
            (
                "Synthetic computational evidence does "
                "not establish human olfactory "
                "equivalence or human perceptual "
                "validity."
            ),
            (
                "These results do not establish "
                "real-world deployment readiness or "
                "performance with physical chemical "
                "sensors."
            ),
            (
                "Track A evaluates retrieval only when "
                "the correct target is represented in "
                "training memory."
            ),
            (
                "The held-out-family OOD question "
                "remains a separate evaluation track."
            ),
            (
                "Hybrid comparisons are secondary "
                "descriptive analyses in this artifact."
            ),
        ],
    }


def export_repeated_track_a_analysis(
    analysis: RepeatedTrackAAnalysis,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> RepeatedTrackAAnalysisExport:
    """Export deterministic JSON and SHA-256 files."""

    if not isinstance(
        analysis,
        RepeatedTrackAAnalysis,
    ):
        raise RepeatedTrackAAnalysisExportError(
            "analysis must be a "
            "RepeatedTrackAAnalysis."
        )

    if not isinstance(overwrite, bool):
        raise RepeatedTrackAAnalysisExportError(
            "overwrite must be boolean."
        )

    json_path = Path(output_path)

    if json_path.suffix.lower() != ".json":
        raise RepeatedTrackAAnalysisExportError(
            "output_path must use the .json extension."
        )

    sha256_path = json_path.with_suffix(
        json_path.suffix + ".sha256"
    )

    if (
        not overwrite
        and (
            json_path.exists()
            or sha256_path.exists()
        )
    ):
        raise RepeatedTrackAAnalysisExportError(
            f"Aggregate artifact already exists: "
            f"{json_path}"
        )

    payload = build_repeated_track_a_payload(
        analysis
    )

    try:
        serialized = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RepeatedTrackAAnalysisExportError(
            "Aggregate payload is not serializable."
        ) from exc

    digest = sha256(serialized).hexdigest()

    try:
        json_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        json_path.write_bytes(serialized)
        sha256_path.write_text(
            digest + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RepeatedTrackAAnalysisExportError(
            "Unable to write aggregate artifacts."
        ) from exc

    return RepeatedTrackAAnalysisExport(
        json_path=json_path,
        sha256_path=sha256_path,
        sha256=digest,
    )


def _descriptive_difference(
    differences: tuple[float, ...],
) -> dict[str, Any]:
    return {
        "differences": list(differences),
        "mean_difference": fmean(differences),
        "wins": sum(
            difference > 0.0
            for difference in differences
        ),
        "ties": sum(
            difference == 0.0
            for difference in differences
        ),
        "losses": sum(
            difference < 0.0
            for difference in differences
        ),
        "status": "secondary_descriptive",
    }
