"""Generate, audit, export, and verify the NOI synthetic pilot dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.dataset_export import (
    DatasetExportResult,
    export_synthetic_dataset,
    verify_exported_dataset,
)
from src.evaluation.leakage_audit import audit_synthetic_dataset
from src.evaluation.synthetic_generator import generate_synthetic_pilot


DEFAULT_SYNTHETIC_CONFIG = Path("configs/synthetic_data.yaml")
DEFAULT_RESEARCH_PROTOCOL = Path("configs/research_protocol.yaml")
DEFAULT_OUTPUT_DIRECTORY = Path("data/processed/pilot-v0.1")
DEFAULT_EVENT_COUNT = 200


class PilotExportError(RuntimeError):
    """Raised when the pilot export or its verification fails."""


def run_pilot_export(
    *,
    synthetic_config_path: str | Path = DEFAULT_SYNTHETIC_CONFIG,
    research_protocol_path: str | Path = DEFAULT_RESEARCH_PROTOCOL,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    event_count: int = DEFAULT_EVENT_COUNT,
    overwrite: bool = False,
) -> DatasetExportResult:
    """Generate, audit, export, and verify one synthetic pilot dataset."""

    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 1
    ):
        raise PilotExportError(
            "event_count must be a positive integer."
        )

    if not isinstance(overwrite, bool):
        raise PilotExportError(
            "overwrite must be boolean."
        )

    dataset = generate_synthetic_pilot(
        synthetic_config_path,
        research_protocol_path,
        event_count=event_count,
    )

    audit_report = audit_synthetic_dataset(
        dataset,
        raise_on_failure=True,
    )

    if not audit_report.passed:
        raise PilotExportError(
            "The pilot dataset failed its leakage audit."
        )

    result = export_synthetic_dataset(
        dataset,
        output_directory,
        overwrite=overwrite,
    )

    if not verify_exported_dataset(result):
        raise PilotExportError(
            "The exported pilot failed SHA-256 verification."
        )

    return result


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for reproducible pilot export."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate, leakage-audit, export, and verify the "
            "synthetic NOI pilot dataset."
        )
    )

    parser.add_argument(
        "--synthetic-config",
        type=Path,
        default=DEFAULT_SYNTHETIC_CONFIG,
        help="Path to the locked synthetic-data configuration.",
    )

    parser.add_argument(
        "--research-protocol",
        type=Path,
        default=DEFAULT_RESEARCH_PROTOCOL,
        help="Path to the preregistered research protocol.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for deterministic pilot artifacts.",
    )

    parser.add_argument(
        "--event-count",
        type=int,
        default=DEFAULT_EVENT_COUNT,
        help="Number of synthetic pilot events.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly authorize replacement of an existing export.",
    )

    return parser


def main() -> int:
    """Run the pilot workflow from the command line."""

    parser = build_argument_parser()
    arguments = parser.parse_args()

    result = run_pilot_export(
        synthetic_config_path=arguments.synthetic_config,
        research_protocol_path=arguments.research_protocol,
        output_directory=arguments.output_directory,
        event_count=arguments.event_count,
        overwrite=arguments.overwrite,
    )

    print("NOI synthetic pilot exported successfully")
    print(f"Output directory: {result.output_directory}")
    print(f"Targets file: {result.targets_path}")
    print(f"Targets SHA-256: {result.targets_sha256}")
    print(f"Events file: {result.events_path}")
    print(f"Events SHA-256: {result.events_sha256}")
    print(f"Manifest file: {result.manifest_path}")
    print(f"Manifest SHA-256: {result.manifest_sha256}")
    print("Leakage audit: PASSED")
    print("Export verification: PASSED")
    print(
        "Scope: synthetic implementation evaluation only; "
        "no human perceptual, clinical, device, or comprehensive "
        "safety claim."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())