"""Deterministic per-seed checkpoint export for repeated Track A."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from src.evaluation.seen_item_final_experiment import (
    SeenItemFinalExperiment,
)
from src.evaluation.seen_item_repeated_runner import (
    RepeatedSeedRunResult,
)
from src.evaluation.seen_item_result_export import (
    build_seen_item_final_payload,
)


class RepeatedSeedExportError(ValueError):
    """Raised when a repeated-run checkpoint cannot be exported."""


@dataclass(frozen=True, slots=True)
class RepeatedSeedExport:
    """Paths and checksum for one seed checkpoint."""

    json_path: Path
    sha256_path: Path
    sha256: str


def export_repeated_seed_result(
    result: RepeatedSeedRunResult,
    output_directory: str | Path,
    *,
    repeated_protocol_sha256: str,
    overwrite: bool = False,
) -> RepeatedSeedExport:
    """Export one completed seed without overwriting by default."""

    if not isinstance(result, RepeatedSeedRunResult):
        raise RepeatedSeedExportError(
            "result must be a RepeatedSeedRunResult."
        )

    if not isinstance(
        result.experiment,
        SeenItemFinalExperiment,
    ):
        raise RepeatedSeedExportError(
            "result must contain a final experiment."
        )

    if (
        not isinstance(repeated_protocol_sha256, str)
        or len(repeated_protocol_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in repeated_protocol_sha256
        )
    ):
        raise RepeatedSeedExportError(
            "repeated_protocol_sha256 is invalid."
        )

    if not isinstance(overwrite, bool):
        raise RepeatedSeedExportError(
            "overwrite must be boolean."
        )

    output_dir = Path(output_directory)
    json_path = output_dir / (
        f"{result.run_id}.json"
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
        raise RepeatedSeedExportError(
            f"Checkpoint already exists: {json_path}"
        )

    payload = build_seen_item_final_payload(
        result.experiment
    )
    payload["schema_version"] = "0.2.1"
    payload["repeated_protocol_sha256"] = (
        repeated_protocol_sha256
    )
    payload["repeated_run"] = {
        "run_id": result.run_id,
        "generator_seed": result.generator_seed,
        "ood_seed": result.ood_seed,
        "partition_seed": result.partition_seed,
    }

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

    digest = sha256(serialized).hexdigest()

    try:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        json_path.write_bytes(serialized)
        sha256_path.write_text(
            digest + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RepeatedSeedExportError(
            "Unable to write checkpoint files."
        ) from exc

    return RepeatedSeedExport(
        json_path=json_path,
        sha256_path=sha256_path,
        sha256=digest,
    )
