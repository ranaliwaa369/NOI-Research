"""Deterministic export of validated synthetic NOI datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.evaluation.leakage_audit import audit_synthetic_dataset
from src.evaluation.synthetic_records import SyntheticDataset


class DatasetExportError(ValueError):
    """Raised when a synthetic dataset cannot be exported safely."""


@dataclass(frozen=True)
class DatasetExportResult:
    """Paths and cryptographic hashes produced by one export."""

    output_directory: Path
    targets_path: Path
    events_path: Path
    manifest_path: Path
    targets_sha256: str
    events_sha256: str
    manifest_sha256: str


def export_synthetic_dataset(
    dataset: SyntheticDataset,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> DatasetExportResult:
    """Export a leakage-audited dataset in deterministic JSONL form.

    The output includes separate odor-target and event files plus a manifest
    containing record counts, generator provenance, split counts, and SHA-256
    hashes.

    Existing export files are not overwritten unless ``overwrite=True``.
    Passing this export process does not establish perceptual or clinical
    validity.
    """

    if not isinstance(dataset, SyntheticDataset):
        raise DatasetExportError(
            "dataset must be an instance of SyntheticDataset."
        )

    if not isinstance(overwrite, bool):
        raise DatasetExportError(
            "overwrite must be boolean."
        )

    output_path = Path(output_directory)

    if output_path.exists() and not output_path.is_dir():
        raise DatasetExportError(
            "output_directory exists but is not a directory."
        )

    # Export is prohibited unless the dataset passes the locked leakage audit.
    audit_report = audit_synthetic_dataset(
        dataset,
        raise_on_failure=True,
    )

    output_path.mkdir(parents=True, exist_ok=True)

    targets_path = output_path / "odor_targets.jsonl"
    events_path = output_path / "events.jsonl"
    manifest_path = output_path / "dataset_manifest.json"

    output_files = (
        targets_path,
        events_path,
        manifest_path,
    )

    existing_files = tuple(
        path for path in output_files if path.exists()
    )

    if existing_files and not overwrite:
        names = ", ".join(path.name for path in existing_files)
        raise DatasetExportError(
            f"Refusing to overwrite existing export files: {names}"
        )

    target_records = [
        {
            "item_id": target.item_id,
            "family_id": target.family_id,
            "odor_vector": list(target.odor_vector),
        }
        for target in sorted(
            dataset.odor_targets,
            key=lambda value: value.item_id,
        )
    ]

    event_records = [
        {
            "event_id": event.event_id,
            "split": event.split.value,
            "template_id": event.template_id,
            "target_item_id": event.target_item_id,
            "target_family_id": event.target_family_id,
            "text_vector": list(event.text_vector),
            "image_vector": list(event.image_vector),
            "audio_vector": list(event.audio_vector),
        }
        for event in sorted(
            dataset.events,
            key=lambda value: value.event_id,
        )
    ]

    targets_content = _encode_json_lines(target_records)
    events_content = _encode_json_lines(event_records)

    targets_hash = _sha256_bytes(targets_content)
    events_hash = _sha256_bytes(events_content)

    split_counts = {
        split_name: count
        for split_name, count in audit_report.split_counts
    }

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "dataset_type": "synthetic_implementation_evaluation",
        "project": "Neuro-Olfactive Interface",
        "owner": "GUARDIANX LLC",
        "generator": {
            "version": dataset.generator_version,
            "generator_seed": dataset.generator_seed,
            "ood_seed": dataset.ood_seed,
        },
        "counts": {
            "odor_targets": len(dataset.odor_targets),
            "events": len(dataset.events),
            "splits": split_counts,
        },
        "leakage_audit": {
            "passed": audit_report.passed,
            "duplicate_event_ids": list(
                audit_report.duplicate_event_ids
            ),
            "cross_split_feature_duplicates": list(
                audit_report.cross_split_feature_duplicates
            ),
            "ood_family_overlap": list(
                audit_report.ood_family_overlap
            ),
            "template_overlap": list(
                audit_report.template_overlap
            ),
            "inconsistent_target_families": list(
                audit_report.inconsistent_target_families
            ),
            "missing_splits": list(
                audit_report.missing_splits
            ),
        },
        "files": {
            "odor_targets.jsonl": {
                "sha256": targets_hash,
                "records": len(target_records),
            },
            "events.jsonl": {
                "sha256": events_hash,
                "records": len(event_records),
            },
        },
        "scope_limitations": [
            "Synthetic implementation evaluation only.",
            "No human perceptual validity claim.",
            "No clinical or diagnostic claim.",
            "No physical odor-emission safety claim.",
        ],
    }

    manifest_content = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    manifest_hash = _sha256_bytes(manifest_content)

    temporary_paths = {
        targets_path: output_path / ".odor_targets.jsonl.tmp",
        events_path: output_path / ".events.jsonl.tmp",
        manifest_path: output_path / ".dataset_manifest.json.tmp",
    }

    try:
        _write_bytes(
            temporary_paths[targets_path],
            targets_content,
        )
        _write_bytes(
            temporary_paths[events_path],
            events_content,
        )
        _write_bytes(
            temporary_paths[manifest_path],
            manifest_content,
        )

        for final_path, temporary_path in temporary_paths.items():
            temporary_path.replace(final_path)

    except OSError as error:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)

        raise DatasetExportError(
            f"Dataset export failed: {error}"
        ) from error

    return DatasetExportResult(
        output_directory=output_path,
        targets_path=targets_path,
        events_path=events_path,
        manifest_path=manifest_path,
        targets_sha256=targets_hash,
        events_sha256=events_hash,
        manifest_sha256=manifest_hash,
    )


def verify_exported_dataset(
    result: DatasetExportResult,
) -> bool:
    """Verify that exported files still match their recorded hashes."""

    if not isinstance(result, DatasetExportResult):
        raise DatasetExportError(
            "result must be a DatasetExportResult."
        )

    required_files = (
        result.targets_path,
        result.events_path,
        result.manifest_path,
    )

    if any(not path.is_file() for path in required_files):
        return False

    return (
        _sha256_file(result.targets_path)
        == result.targets_sha256
        and _sha256_file(result.events_path)
        == result.events_sha256
        and _sha256_file(result.manifest_path)
        == result.manifest_sha256
    )


def _encode_json_lines(
    records: list[dict[str, Any]],
) -> bytes:
    """Encode records as deterministic UTF-8 JSON Lines."""

    lines = [
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for record in records
    ]

    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_bytes(
    path: Path,
    content: bytes,
) -> None:
    """Write bytes to one temporary export file."""

    path.write_bytes(content)


def _sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 hexadecimal digest of bytes."""

    return sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hexadecimal digest of one file."""

    digest = sha256()

    with path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()
