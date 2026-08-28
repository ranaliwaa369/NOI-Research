"""Deterministic final robustness seed export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.evaluation.final_robustness_experiment import (
    FinalRobustnessExperiment,
)


class FinalRobustnessExportError(ValueError):
    """Raised when robustness export fails."""


@dataclass(frozen=True)
class FinalRobustnessExport:
    json_path: Path
    sha256_path: Path
    sha256: str


def export_final_robustness_experiment(
    experiment: FinalRobustnessExperiment,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> FinalRobustnessExport:
    """Export one compact deterministic seed result."""

    if not isinstance(
        experiment,
        FinalRobustnessExperiment,
    ):
        raise FinalRobustnessExportError(
            "experiment must be a final "
            "robustness experiment."
        )

    json_path = Path(output_path)

    if json_path.suffix.lower() != ".json":
        raise FinalRobustnessExportError(
            "Output must use a .json suffix."
        )

    if json_path.stem != experiment.run_id:
        raise FinalRobustnessExportError(
            "Output filename must match run ID."
        )

    hash_path = json_path.with_suffix(
        ".json.sha256"
    )

    if not overwrite and (
        json_path.exists()
        or hash_path.exists()
    ):
        raise FinalRobustnessExportError(
            "Robustness artifact already exists."
        )

    payload = _payload(experiment)

    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    digest = sha256(encoded).hexdigest()

    try:
        json_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        json_path.write_bytes(encoded)
        hash_path.write_text(
            digest + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise FinalRobustnessExportError(
            "Unable to write robustness artifacts."
        ) from exc

    return FinalRobustnessExport(
        json_path=json_path,
        sha256_path=hash_path,
        sha256=digest,
    )


def _payload(
    experiment: FinalRobustnessExperiment,
) -> dict[str, Any]:
    ids_by_tier: dict[str, list[str]] = {}

    for evaluation in experiment.evaluations:
        existing = ids_by_tier.get(
            evaluation.tier
        )
        current = list(
            evaluation.latent_event_ids
        )

        if existing is None:
            ids_by_tier[
                evaluation.tier
            ] = current
        elif existing != current:
            raise FinalRobustnessExportError(
                "Latent event pairing differs "
                "within a severity tier."
            )

    evaluations = [
        {
            "axis": item.axis,
            "condition_id": item.condition_id,
            "system": item.system,
            "tier": item.tier,
            "missing_modalities": list(
                item.missing_modalities
            ),
            "temporal_displacement_days": (
                item.temporal_displacement_days
            ),
            "selected_alpha": (
                item.selected_alpha
            ),
            "apply_temporal_decay": (
                item.apply_temporal_decay
            ),
            "event_count": item.event_count,
            "reciprocal_ranks": list(
                item.reciprocal_ranks
            ),
            "recall_at_1": item.recall_at_1,
            "recall_at_10": (
                item.recall_at_10
            ),
            "mean_reciprocal_rank": (
                item.mean_reciprocal_rank
            ),
            "ndcg_at_10": item.ndcg_at_10,
        }
        for item in experiment.evaluations
    ]

    return {
        "artifact_type": (
            "final_robustness_seed_result"
        ),
        "schema_version": "1.0",
        "run_id": experiment.run_id,
        "seeds": {
            "generator_seed": (
                experiment.generator_seed
            ),
            "ood_seed": experiment.ood_seed,
        },
        "protocol_sha256": (
            experiment.protocol_hash
        ),
        "counts": {
            "training_events": (
                experiment.training_event_count
            ),
            "validation_events": (
                experiment.validation_event_count
            ),
            "latent_ood_events": (
                experiment.latent_event_count
            ),
            "observed_ood_events": (
                experiment.observed_event_count
            ),
            "odor_library_size": (
                experiment.odor_library_size
            ),
            "evaluations": len(
                experiment.evaluations
            ),
        },
        "selected_validation_alpha": (
            experiment.selected_validation_alpha
        ),
        "paired_analysis_unit": (
            experiment.paired_analysis_unit
        ),
        "paired_latent_event_ids_by_tier": (
            ids_by_tier
        ),
        "governance": {
            "all_ood_targets_unreachable": (
                experiment
                .all_ood_targets_unreachable
            ),
            "strict_family_separation_verified": (
                experiment
                .strict_family_separation_verified
            ),
            "oracle_used": (
                experiment.oracle_used
            ),
            "ood_tuning_used": (
                experiment.ood_tuning_used
            ),
            "final_test_tuning_used": (
                experiment
                .final_test_tuning_used
            ),
        },
        "evaluations": evaluations,
    }
