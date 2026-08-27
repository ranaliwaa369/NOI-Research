"""Deterministic export for one Track B seed result."""

from __future__ import annotations

import json
from dataclasses import (
    dataclass,
    fields,
    is_dataclass,
)
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.track_b_seed_experiment import (
    TrackBSeedExperiment,
)


class TrackBSeedExportError(ValueError):
    """Raised when a Track B result cannot be exported."""


@dataclass(frozen=True)
class TrackBSeedExport:
    """Paths and digest for one exported result."""

    json_path: Path
    sha256_path: Path
    sha256: str


def export_track_b_seed_experiment(
    experiment: TrackBSeedExperiment,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> TrackBSeedExport:
    """Export one Track B experiment and its SHA-256."""

    if not isinstance(
        experiment,
        TrackBSeedExperiment,
    ):
        raise TrackBSeedExportError(
            "experiment must be a "
            "TrackBSeedExperiment."
        )

    json_path = Path(output_path)

    if json_path.suffix.lower() != ".json":
        raise TrackBSeedExportError(
            "Track B output must use a .json suffix."
        )

    if json_path.stem != experiment.run_id:
        raise TrackBSeedExportError(
            "Output filename must match the run ID."
        )

    sha256_path = json_path.with_suffix(
        json_path.suffix + ".sha256"
    )

    if not overwrite and (
        json_path.exists()
        or sha256_path.exists()
    ):
        raise TrackBSeedExportError(
            "Track B output artifact already exists."
        )

    payload = _build_payload(experiment)

    try:
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
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TrackBSeedExportError(
            "Track B result is not JSON serializable."
        ) from exc

    digest = sha256(encoded).hexdigest()

    try:
        json_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        json_path.write_bytes(encoded)
        sha256_path.write_text(
            digest + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise TrackBSeedExportError(
            "Unable to write Track B artifacts."
        ) from exc

    return TrackBSeedExport(
        json_path=json_path,
        sha256_path=sha256_path,
        sha256=digest,
    )


def _build_payload(
    experiment: TrackBSeedExperiment,
) -> dict[str, Any]:
    return {
        "artifact_type": (
            "track_b_unseen_family_seed_result"
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
        },
        "reachability": {
            "reachable_target_fraction": (
                experiment.reachable_target_fraction
            ),
            "reachable_event_fraction": (
                experiment.reachable_event_fraction
            ),
        },
        "selected_hybrid_alpha": (
            experiment.selected_hybrid_alpha
        ),
        "governance": {
            "oracle_used": experiment.oracle_used,
            "final_test_tuning_used": (
                experiment.final_test_tuning_used
            ),
            "target_identifier_used_in_support": (
                experiment
                .target_identifier_used_in_support
            ),
            "family_identifier_used_in_support": (
                experiment
                .family_identifier_used_in_support
            ),
            "strict_family_separation_verified": (
                experiment
                .strict_family_separation_verified
            ),
            "all_ood_targets_unreachable": (
                experiment
                .all_ood_targets_unreachable
            ),
        },
        "calibration": _json_value(
            experiment.calibration
        ),
        "graded_baselines": _json_value(
            experiment.graded_baselines
        ),
        "full_noi_evaluations": _json_value(
            experiment.full_noi_evaluations
        ),
        "memory_only_evaluations": _json_value(
            experiment.memory_only_evaluations
        ),
        "selective_evaluations": _json_value(
            experiment.selective_evaluations
        ),
    }


def _json_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
        return _json_value(value.value)

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not isfinite(value):
            raise TrackBSeedExportError(
                "Non-finite numeric value encountered."
            )
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if is_dataclass(value) and not isinstance(
        value,
        type,
    ):
        return {
            field.name: _json_value(
                getattr(value, field.name)
            )
            for field in fields(value)
        }

    if isinstance(value, Mapping):
        return {
            _json_key(key): _json_value(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (tuple, list),
    ):
        return [
            _json_value(item)
            for item in value
        ]

    if isinstance(
        value,
        (set, frozenset),
    ):
        converted = [
            _json_value(item)
            for item in value
        ]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

    item_method = getattr(value, "item", None)

    if callable(item_method):
        try:
            scalar = item_method()
        except (TypeError, ValueError):
            scalar = value
        else:
            if scalar is not value:
                return _json_value(scalar)

    raise TrackBSeedExportError(
        "Unsupported value in Track B result: "
        f"{type(value).__name__}."
    )


def _json_key(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float, bool)):
        return str(value)

    raise TrackBSeedExportError(
        "Unsupported mapping key in Track B result."
    )
