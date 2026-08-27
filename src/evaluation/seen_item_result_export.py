"""Deterministic JSON and SHA-256 export for NOI Track A."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.evaluation.seen_item_final_experiment import (
    SeenItemFinalExperiment,
)


class SeenItemResultExportError(ValueError):
    """Raised when Track A results cannot be exported."""


@dataclass(frozen=True, slots=True)
class SeenItemResultExport:
    """Paths and checksum for one exported result artifact."""

    json_path: Path
    sha256_path: Path
    sha256: str

    def __post_init__(self) -> None:
        if self.json_path.suffix.lower() != ".json":
            raise SeenItemResultExportError(
                "json_path must use the .json extension."
            )

        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.sha256
            )
        ):
            raise SeenItemResultExportError(
                "sha256 must be a lowercase SHA-256 digest."
            )


def export_seen_item_final_experiment(
    experiment: SeenItemFinalExperiment,
    output_path: str | Path,
) -> SeenItemResultExport:
    """Export deterministic, auditable Track A results."""

    if not isinstance(
        experiment,
        SeenItemFinalExperiment,
    ):
        raise SeenItemResultExportError(
            "experiment must be a SeenItemFinalExperiment."
        )

    json_path = Path(output_path)

    if json_path.suffix.lower() != ".json":
        raise SeenItemResultExportError(
            "output_path must use the .json extension."
        )

    payload = _build_payload(experiment)

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
        raise SeenItemResultExportError(
            "Result payload is not JSON serializable."
        ) from exc

    digest = sha256(serialized).hexdigest()
    sha256_path = json_path.with_suffix(
        json_path.suffix + ".sha256"
    )

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
        raise SeenItemResultExportError(
            "Unable to write result artifacts."
        ) from exc

    return SeenItemResultExport(
        json_path=json_path,
        sha256_path=sha256_path,
        sha256=digest,
    )


def build_seen_item_final_payload(
    experiment: SeenItemFinalExperiment,
) -> dict[str, Any]:
    """Return the deterministic Track A result payload."""

    if not isinstance(
        experiment,
        SeenItemFinalExperiment,
    ):
        raise SeenItemResultExportError(
            "experiment must be a SeenItemFinalExperiment."
        )

    return _build_payload(experiment)


def _build_payload(
    experiment: SeenItemFinalExperiment,
) -> dict[str, Any]:
    systems = {}

    for evaluation in experiment.evaluations:
        event_results = []

        for (
            event_id,
            ranking,
            relevant,
        ) in zip(
            evaluation.event_ids,
            evaluation.rankings,
            evaluation.relevant_items,
            strict=True,
        ):
            event_results.append(
                {
                    "event_id": event_id,
                    "memory_reachable": True,
                    "ranking": list(ranking),
                    "relevant_items": sorted(relevant),
                }
            )

        systems[evaluation.system.value] = {
            "alpha": evaluation.alpha,
            "metrics": {
                "recall_at_1": (
                    evaluation.recall_at_1
                ),
                "recall_at_10": (
                    evaluation.recall_at_10
                ),
                "mean_reciprocal_rank": (
                    evaluation.mean_reciprocal_rank
                ),
                "ndcg_at_10": (
                    evaluation.ndcg_at_10
                ),
            },
            "event_results": event_results,
        }

    return {
        "schema_version": "0.2.0",
        "evaluation_track": (
            "seen_item_episodic_retrieval"
        ),
        "evidence_scope": (
            "synthetic_computational_evaluation_only"
        ),
        "protocol_hash": experiment.protocol_hash,
        "oracle_used": experiment.oracle_used,
        "final_test_tuning_used": (
            experiment.final_test_tuning_used
        ),
        "selected_hybrid_alpha": (
            experiment.selected_hybrid_alpha
        ),
        "counts": {
            "training_events": (
                experiment.training_event_count
            ),
            "calibration_events": (
                experiment.calibration_event_count
            ),
            "raw_final_test_events": (
                experiment.raw_final_test_event_count
            ),
            "reachable_final_test_events": (
                experiment.final_test_event_count
            ),
        },
        "reachable_event_fraction": (
            experiment.reachable_event_fraction
        ),
        "templates": {
            "calibration": list(
                experiment.calibration_template_ids
            ),
            "final_test": list(
                experiment.final_test_template_ids
            ),
            "overlap": [],
        },
        "final_test_events": [
            {
                "event_id": event_id,
                "memory_reachable": True,
            }
            for event_id in experiment.final_test_event_ids
        ],
        "systems": systems,
        "limitations": [
            (
                "Synthetic performance does not establish "
                "human perceptual validity."
            ),
            (
                "Synthetic performance does not establish "
                "real-world olfactory performance."
            ),
            (
                "One deterministic run does not establish "
                "statistical generalizability."
            ),
        ],
    }
